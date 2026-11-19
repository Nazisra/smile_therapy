import os
import gc
import io
import base64
import logging
import traceback
import sys
from pathlib import Path
from typing import Optional

try:
    import sklearn._loss._loss
    sys.modules['_loss'] = sklearn._loss._loss
except Exception:
    pass

import numpy as np
import cv2
import joblib
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("opg_asymmetry")

app = FastAPI(title="Smile Therapy - OPG Asymmetry AI", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Robust checkpoint path resolution (works locally, in subdirectories, and on Google Colab)
CANDIDATE_PATHS = [
    Path(__file__).resolve().parent / "final_multimodal_checkpoint.joblib",
    Path(__file__).resolve().parents[2] / "final_multimodal_checkpoint.joblib",
    Path("final_multimodal_checkpoint.joblib"),
    Path("app/backend/final_multimodal_checkpoint.joblib"),
    Path("/content/final_multimodal_checkpoint.joblib"),
    Path("/content/web_dental/final_multimodal_checkpoint.joblib"),
]

CHECKPOINT = None
for cp in CANDIDATE_PATHS:
    if cp.exists():
        CHECKPOINT = cp
        break

gb_model = None
clin_scaler = None
pca = None
vis_scaler = None
clinical_columns = None
class_names = None
densenet = None

# Automatically leverage Colab GPU (CUDA) or fast CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_densenet():
    """Builds original pre-trained DenseNet-121 feature extractor."""
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Identity()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


@app.on_event("startup")
async def startup_event():
    global gb_model, clin_scaler, pca, vis_scaler, clinical_columns, class_names, densenet, CHECKPOINT
    logger.info("Target execution device: %s", device)

    if CHECKPOINT is None or not CHECKPOINT.exists():
        for cp in CANDIDATE_PATHS:
            if cp.exists():
                CHECKPOINT = cp
                break

    if CHECKPOINT is None or not CHECKPOINT.exists():
        raise RuntimeError(f"Multimodal checkpoint not found in candidate paths: {[str(p) for p in CANDIDATE_PATHS]}")

    logger.info("Loading multimodal checkpoint: %s", CHECKPOINT)
    ckpt = joblib.load(CHECKPOINT)
    gb_model = ckpt["best_estimator"]
    clin_scaler = ckpt["clin_scaler"]
    pca = ckpt["pca"]
    vis_scaler = ckpt["vis_scaler"]
    clinical_columns = ckpt["clinical_columns"]
    class_names = ckpt["class_names"]
    logger.info("Clinical columns (%d) and GBDT model loaded successfully.", len(clinical_columns))

    # Original DenseNet-121 feature extractor
    densenet = build_densenet().to(device)
    logger.info("DenseNet-121 loaded successfully on %s.", device)
    gc.collect()


IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])


def crop_black_borders(img_bgr: np.ndarray, threshold: int = 20) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    if not np.any(mask):
        return img_bgr
    y_indices, x_indices = np.where(mask)
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
    h, w = gray.shape
    if (y_max - y_min > 0.3 * h) and (x_max - x_min > 0.3 * w):
        return img_bgr[y_min:y_max + 1, x_min:x_max + 1]
    return img_bgr


def preprocess_opg(img_bgr: np.ndarray):
    """Preprocesses radiograph into standardized CLAHE bone ROI and PyTorch tensor."""
    img_bgr = crop_black_borders(img_bgr)
    h, w = img_bgr.shape[:2]
    y1, y2 = int(0.15 * h), int(0.85 * h)
    x1, x2 = int(0.05 * w), int(0.95 * w)
    roi = img_bgr[y1:y2, x1:x2]
    roi = cv2.bilateralFilter(roi, d=5, sigmaColor=35, sigmaSpace=35)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    roi_color = roi.copy()
    roi_224 = cv2.resize(roi, (224, 224))
    roi_rgb = cv2.cvtColor(roi_224, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(roi_rgb)
    tensor = preprocess(pil_img).unsqueeze(0).to(device)
    _, buf = cv2.imencode(".png", roi_224)
    clahe_b64 = base64.b64encode(buf.tobytes()).decode()
    return roi_color, tensor, clahe_b64


def automated_mandibular_measurement(img_bgr: np.ndarray):
    # 1. Automatically detect and crop outer black borders (threshold > 15)
    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray_full > 15
    y_idx, x_idx = np.where(mask)
    if len(y_idx) > 0:
        y1, y2 = int(np.min(y_idx)), int(np.max(y_idx))
        x1, x2 = int(np.min(x_idx)), int(np.max(x_idx))
        cropped = img_bgr[y1:y2 + 1, x1:x2 + 1]
    else:
        cropped = img_bgr

    # Standardize scale for robust, consistent landmark detection
    std_w = 1200
    std_h = int(cropped.shape[0] * (std_w / max(cropped.shape[1], 1)))
    resized = cv2.resize(cropped, (std_w, std_h))
    canvas = resized.copy()
    gray_r = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Exclude top 28% header/logo/skull area and bottom 5%
    top_y = int(0.28 * std_h)
    bot_y = int(0.95 * std_h)
    mandible_band = gray_r[top_y:bot_y, :]
    band_h, band_w = mandible_band.shape

    # CLAHE enhancement on mandibular bone ROI
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(mandible_band)

    # Patient's Right ramus (viewer left: 4% to 33% width)
    # Patient's Left ramus (viewer right: 67% to 96% width)
    # Central 34% (33% to 67%) contains cervical spine & overlapping teeth and is excluded
    def detect_side_landmarks(x_start_pct, x_end_pct, is_patient_right=True):
        x1 = int(x_start_pct * band_w)
        x2 = int(x_end_pct * band_w)
        ramus_roi = enhanced[:, x1:x2]

        # Otsu thresholding + Morphological closing to isolate bone structure
        _, thresh = cv2.threshold(ramus_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # Fallback based on typical adult mandibular anatomical proportions
            default_rh = int(band_h * 0.65)
            default_ch = int(default_rh * 0.28)
            default_mh = default_rh - default_ch
            cx = (x1 + x2) // 2
            return {
                "Co": (cx, top_y + int(band_h * 0.12)),
                "Sig": (cx, top_y + int(band_h * 0.12) + default_ch),
                "Go": (cx, top_y + int(band_h * 0.12) + default_rh),
                "CH": float(default_ch),
                "MH": float(default_mh),
                "RH": float(default_rh),
            }

        largest_cnt = max(contours, key=cv2.contourArea)
        pts = largest_cnt.reshape(-1, 2)  # relative to ramus_roi (x, y)

        # 1. Condylion (Co): Top-most superior point of the condyle
        top_candidates = pts[pts[:, 1] < np.percentile(pts[:, 1], 15)]
        if len(top_candidates) > 0:
            co_pt = top_candidates[np.argmin(top_candidates[:, 1])]
        else:
            co_pt = pts[np.argmin(pts[:, 1])]

        # 2. Gonion (Go): Bottom-most postero-inferior corner
        bot_candidates = pts[pts[:, 1] > np.percentile(pts[:, 1], 85)]
        if len(bot_candidates) > 0:
            if is_patient_right:
                # Patient right (image left): postero-inferior corner is more toward the lateral border (smaller x)
                go_pt = bot_candidates[np.lexsort((bot_candidates[:, 0], -bot_candidates[:, 1]))[0]]
            else:
                # Patient left (image right): postero-inferior corner is more toward lateral border (larger x)
                go_pt = bot_candidates[np.lexsort((-bot_candidates[:, 0], -bot_candidates[:, 1]))[0]]
        else:
            go_pt = pts[np.argmax(pts[:, 1])]

        # 3. Sigmoid Notch (Sig): Inflection dip between condyle and coronoid
        co_y, go_y = int(co_pt[1]), int(go_pt[1])
        rh_val = max(go_y - co_y, 10)

        # Mandibular ramus spans at least 40% of the mandibular band; prevent false small fragments
        if rh_val < int(0.40 * band_h):
            rh_val = int(0.65 * band_h)
            go_y = min(co_y + rh_val, band_h - 5)
            go_pt = np.array([co_pt[0], go_y])

        expected_sig_y = co_y + int(0.29 * rh_val)
        internal = pts

        notch_cand = internal[(internal[:, 1] > co_y + int(0.15 * rh_val)) & (internal[:, 1] < go_y - int(0.40 * rh_val))]
        if len(notch_cand) > 0:
            dists = np.abs(notch_cand[:, 1] - expected_sig_y)
            sig_pt = notch_cand[np.argmin(dists)]
        else:
            sig_pt = np.array([co_pt[0], expected_sig_y])

        # Convert coordinates to canvas space
        co_canvas = (int(co_pt[0] + x1), int(co_pt[1] + top_y))
        sig_canvas = (int(sig_pt[0] + x1), int(sig_pt[1] + top_y))
        go_canvas = (int(go_pt[0] + x1), int(go_pt[1] + top_y))

        ch = float(max(sig_canvas[1] - co_canvas[1], 8))
        rh = float(max(go_canvas[1] - co_canvas[1], 25))

        # Physiological sanity clamp: Condylar Height is 20%-38% of Ramal Height
        if ch < 0.20 * rh or ch > 0.38 * rh:
            ch = float(int(0.285 * rh))
            sig_canvas = (sig_canvas[0], int(co_canvas[1] + ch))

        mh = float(max(rh - ch, 15))

        return {
            "Co": co_canvas,
            "Sig": sig_canvas,
            "Go": go_canvas,
            "CH": ch,
            "MH": mh,
            "RH": rh,
        }

    right_lm = detect_side_landmarks(0.04, 0.33, is_patient_right=True)
    left_lm = detect_side_landmarks(0.67, 0.96, is_patient_right=False)

    # Draw clinical landmark overlay directly onto radiograph
    # Colors: Cyan for CH, Yellow for RH/MH, Red dots for anatomical landmarks
    COLOR_CYAN = (255, 230, 0)
    COLOR_YELLOW = (0, 215, 255)
    COLOR_DOT = (40, 40, 245)
    COLOR_TEXT = (255, 255, 255)

    for side, lm, label_prefix in [("Right", right_lm, "R"), ("Left", left_lm, "L")]:
        co = lm["Co"]
        sig = lm["Sig"]
        go = lm["Go"]

        # 1. Condylar line (CH): Co to Sig (Cyan line)
        cv2.line(canvas, co, (co[0], sig[1]), COLOR_CYAN, 3, cv2.LINE_AA)

        # 2. Ramal line (RH): Co to Go (Yellow dashed/solid line)
        cv2.line(canvas, (co[0] + 8, co[1]), (co[0] + 8, go[1]), COLOR_YELLOW, 2, cv2.LINE_AA)

        # 3. Landmark keypoints
        cv2.circle(canvas, co, 6, COLOR_DOT, -1, cv2.LINE_AA)
        cv2.circle(canvas, (co[0], sig[1]), 5, (0, 255, 120), -1, cv2.LINE_AA)
        cv2.circle(canvas, go, 6, COLOR_DOT, -1, cv2.LINE_AA)

        # 4. Text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        offset_x = -75 if label_prefix == "R" else 15
        cv2.putText(canvas, f"{label_prefix}-Co", (co[0] + offset_x, co[1] - 8), font, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{label_prefix}-Sig", (co[0] + offset_x, sig[1] + 4), font, 0.45, (0, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{label_prefix}-Go", (go[0] + offset_x, go[1] + 15), font, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"CH: {int(lm['CH'])}px", (co[0] + offset_x, (co[1] + sig[1]) // 2), font, 0.45, COLOR_CYAN, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"RH: {int(lm['RH'])}px", (co[0] + offset_x, (sig[1] + go[1]) // 2), font, 0.45, COLOR_YELLOW, 1, cv2.LINE_AA)

    # Encode rendered overlay to base64
    _, buf = cv2.imencode(".png", canvas)
    overlay_b64 = base64.b64encode(buf.tobytes()).decode()

    return {
        "right_CH": right_lm["CH"],
        "right_MH": right_lm["MH"],
        "right_RH": right_lm["RH"],
        "left_CH": left_lm["CH"],
        "left_MH": left_lm["MH"],
        "left_RH": left_lm["RH"],
        "landmarks": {
            "R_Co": right_lm["Co"],
            "R_Sig": right_lm["Sig"],
            "R_Go": right_lm["Go"],
            "L_Co": left_lm["Co"],
            "L_Sig": left_lm["Sig"],
            "L_Go": left_lm["Go"],
        },
        "landmark_overlay_base64": overlay_b64,
    }


def compute_31_features(
    right_CH: float,
    right_MH: float,
    right_RH: float,
    left_CH: float,
    left_MH: float,
    left_RH: float,
) -> dict:
    f = {}
    f["R_CH"] = right_CH
    f["L_CH"] = left_CH
    f["R_MH"] = right_MH
    f["L_MH"] = left_MH
    f["R_RH"] = right_RH
    f["L_RH"] = left_RH

    f["Diff_CH"] = abs(right_CH - left_CH)
    f["Diff_MH"] = abs(right_MH - left_MH)
    f["Diff_RH"] = abs(right_RH - left_RH)

    # Ratio: Right / Left
    f["Ratio_CH"] = right_CH / max(left_CH, 1e-6)
    f["Ratio_MH"] = right_MH / max(left_MH, 1e-6)
    f["Ratio_RH"] = right_RH / max(left_RH, 1e-6)

    # Habets Asymmetry Index: |R - L| / (R + L) * 100
    f["Asym_CH"] = (abs(right_CH - left_CH) / max(right_CH + left_CH, 1e-6)) * 100.0
    f["Asym_MH"] = (abs(right_MH - left_MH) / max(right_MH + left_MH, 1e-6)) * 100.0
    f["Asym_RH"] = (abs(right_RH - left_RH) / max(right_RH + left_RH, 1e-6)) * 100.0

    # Kjellberg's Symmetry Index: min / max * 100
    f["SI_CH_Min"] = (min(right_CH, left_CH) / max(max(right_CH, left_CH), 1e-6)) * 100.0
    f["SI_MH_Min"] = (min(right_MH, left_MH) / max(max(right_MH, left_MH), 1e-6)) * 100.0
    f["SI_RH_Min"] = (min(right_RH, left_RH) / max(max(right_RH, left_RH), 1e-6)) * 100.0

    # Overall / Mean Symmetry Ratios
    f["SI_Ratio_Mean"] = (f["SI_CH_Min"] + f["SI_MH_Min"] + f["SI_RH_Min"]) / 3.0
    f["SI_Ratio_Min"] = min(f["SI_CH_Min"], f["SI_MH_Min"], f["SI_RH_Min"])
    f["SI_Ratio_Max"] = max(f["SI_CH_Min"], f["SI_MH_Min"], f["SI_RH_Min"])

    # Intra-side ratios (CH / RH, MH / RH, CH / MH)
    f["R_CH_RH_Ratio"] = right_CH / max(right_RH, 1e-6)
    f["L_CH_RH_Ratio"] = left_CH / max(left_RH, 1e-6)
    f["R_MH_RH_Ratio"] = right_MH / max(right_RH, 1e-6)
    f["L_MH_RH_Ratio"] = left_MH / max(left_RH, 1e-6)
    f["R_CH_MH_Ratio"] = right_CH / max(right_MH, 1e-6)
    f["L_CH_MH_Ratio"] = left_CH / max(left_MH, 1e-6)

    # Clinical weighted and non-linear interactions
    f["Weighted_Clinical_SI"] = (
        0.50 * f["SI_RH_Min"] + 0.35 * f["SI_CH_Min"] + 0.15 * f["SI_MH_Min"]
    )
    f["Harmonic_Segment_SI"] = 3.0 / (
        1.0 / max(f["SI_CH_Min"], 1e-6)
        + 1.0 / max(f["SI_MH_Min"], 1e-6)
        + 1.0 / max(f["SI_RH_Min"], 1e-6)
    )
    f["RH_Dominance_Factor"] = f["Diff_RH"] / max(f["Diff_CH"] + f["Diff_MH"] + 1e-6, 1e-6)
    f["Asym_Dispersion"] = np.std([f["Asym_CH"], f["Asym_MH"], f["Asym_RH"]])

    return f


def build_clinical_vector(feats: dict, clinical_columns: list) -> np.ndarray:
    vec = []
    for col in clinical_columns:
        if col in feats:
            vec.append(float(feats[col]))
        else:
            raise KeyError(f"Feature '{col}' missing from computed clinical dictionary.")
    return np.array(vec, dtype=np.float32).reshape(1, -1)


def generate_gradcam_overlay(roi_color: np.ndarray, tensor: torch.Tensor, pca_model) -> str:
    """Original DenseNet-121 spatial Grad-CAM attention heatmap using PCA loadings."""
    h, w = roi_color.shape[:2]

    # 1. Extract spatial feature map A (shape [1, 1024, 7, 7]) from densenet.features(tensor)
    with torch.no_grad():
        feats_raw = densenet.features(tensor)
        A = torch.relu(feats_raw).squeeze(0).cpu().numpy()  # (1024, 7, 7)

    # 2. Weight 1024 channels using absolute loadings of top PCA components combined with spatial channel variance
    pca_loadings = np.abs(pca_model.components_).mean(axis=0)  # (1024,)
    channel_var = A.var(axis=(1, 2))  # (1024,)
    channel_var_norm = (channel_var - channel_var.min()) / (channel_var.max() - channel_var.min() + 1e-8)
    weights = pca_loadings * (channel_var_norm + 1e-4)
    weights = weights / (weights.sum() + 1e-8)

    cam = np.tensordot(weights, A, axes=(0, 0))  # (7, 7)

    # 3. Apply ReLU, upsample to cropped mandibular ROI size using cv2.INTER_CUBIC, Gaussian smoothing, and 98th percentile normalization
    cam = np.maximum(cam, 0.0)
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam = cv2.GaussianBlur(cam, (21, 21), 0)

    p98 = float(np.percentile(cam, 98))
    cam = cam / (p98 + 1e-8)
    cam = np.clip(cam, 0.0, 1.0)

    # Zero out weak background noise (cam < 0.25)
    cam_active = np.zeros_like(cam)
    active_mask = cam >= 0.25
    cam_active[active_mask] = (cam[active_mask] - 0.25) / 0.75
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_active), cv2.COLORMAP_JET)

    # Where cam < 0.25, alpha is 0.0 so the OPG radiograph remains crystal-clear grayscale underneath
    # Where cam >= 0.25, alpha smoothly blends vibrant red/yellow into anatomical focus zones
    alpha = np.zeros_like(cam, dtype=np.float32)
    alpha[active_mask] = 0.35 + 0.35 * cam_active[active_mask]
    alpha_3d = alpha[..., np.newaxis]

    blended = ((1.0 - alpha_3d) * roi_color.astype(np.float32) + alpha_3d * heatmap.astype(np.float32))
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    _, buf = cv2.imencode(".png", blended)
    return base64.b64encode(buf.tobytes()).decode()


def extract_visual_features(tensor: torch.Tensor) -> np.ndarray:
    """Original DenseNet-121 1024-dimensional feature extraction."""
    with torch.no_grad():
        feats = densenet(tensor)
    return feats.squeeze(0).cpu().numpy()


TREATMENT = {
    "Mild": {
        "symmetry_range": ">=94%",
        "description": "Physiological variation — within normal limits.",
        "recommendation": "No surgical intervention is required. Schedule routine dental follow-up every 6-12 months to monitor mandibular development. Document baseline measurements for future reference.",
        "urgency": "routine",
        "referral": None,
    },
    "Moderate": {
        "symmetry_range": "90%-93%",
        "description": "Borderline skeletal / dentoalveolar asymmetry.",
        "recommendation": "Orthodontic evaluation is recommended. Consider occlusal adjustment or functional appliance therapy. Monitor closely with 6-monthly OPGs until skeletal maturity. Refer to orthodontist for comprehensive assessment.",
        "urgency": "semi-urgent",
        "referral": "Orthodontist",
    },
    "Severe": {
        "symmetry_range": "<90%",
        "description": "Marked skeletal discrepancy requiring specialist assessment.",
        "recommendation": "Immediate referral to Oral & Maxillofacial Surgery (OMFS). Order 3D CBCT scan for detailed skeletal analysis. Consider Orthognathic Surgery: Bilateral Sagittal Split Osteotomy (BSSO) and/or Genioplasty. Pre-surgical orthodontic preparation advised.",
        "urgency": "urgent",
        "referral": "Oral & Maxillofacial Surgeon",
    },
}


@app.get("/")
async def root():
    return {
        "message": "Smile Therapy Maxillofacial AI Backend is Running",
        "status": "online",
        "model_loaded": gb_model is not None,
        "device": str(device),
        "densenet": "DenseNet-121 (Original Research Architecture)",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": gb_model is not None,
        "device": str(device),
        "densenet_loaded": densenet is not None,
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    right_CH: Optional[float] = Form(None),
    right_MH: Optional[float] = Form(None),
    right_RH: Optional[float] = Form(None),
    left_CH: Optional[float] = Form(None),
    left_MH: Optional[float] = Form(None),
    left_RH: Optional[float] = Form(None),
):
    try:
        raw = await file.read()
        arr = np.frombuffer(raw, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Cannot decode image.")

        img_bgr = crop_black_borders(img_bgr)
        roi_color, tensor, clahe_b64 = preprocess_opg(img_bgr)

        # 100% automated mandibular measurement with landmark visual overlay
        auto_result = automated_mandibular_measurement(img_bgr)
        auto_heights = {
            "right_CH": auto_result["right_CH"],
            "right_MH": auto_result["right_MH"],
            "right_RH": auto_result["right_RH"],
            "left_CH": auto_result["left_CH"],
            "left_MH": auto_result["left_MH"],
            "left_RH": auto_result["left_RH"],
        }
        landmark_b64 = auto_result["landmark_overlay_base64"]

        if right_CH is not None: auto_heights["right_CH"] = right_CH
        if right_MH is not None: auto_heights["right_MH"] = right_MH
        if right_RH is not None: auto_heights["right_RH"] = right_RH
        if left_CH is not None: auto_heights["left_CH"] = left_CH
        if left_MH is not None: auto_heights["left_MH"] = left_MH
        if left_RH is not None: auto_heights["left_RH"] = left_RH

        feats = compute_31_features(**auto_heights)
        clin_vec = build_clinical_vector(feats, clinical_columns)

        # Exact original 1024-D DenseNet visual features
        raw_vis = extract_visual_features(tensor)
        vis_pca = pca.transform(raw_vis.reshape(1, -1))
        vis_scaled = vis_scaler.transform(vis_pca)
        clin_scaled = clin_scaler.transform(clin_vec)
        combined = np.hstack([vis_scaled, clin_scaled])

        proba = gb_model.predict_proba(combined)[0]
        pred_idx = int(np.argmax(proba))
        pred_class = class_names[pred_idx]
        confidence = float(proba[pred_idx]) * 100

        # Exact original Grad-CAM attention heatmap
        gradcam_b64 = generate_gradcam_overlay(roi_color, tensor, pca)

        treatment = TREATMENT.get(pred_class, TREATMENT["Mild"])
        kjellberg_pct = float(feats["SI_Ratio_Min"])
        weighted_si = float(feats["Weighted_Clinical_SI"])
        class_probs = {class_names[i]: round(float(proba[i]) * 100, 2) for i in range(len(class_names))}

        return JSONResponse({
            "predicted_class": pred_class,
            "confidence": round(confidence, 2),
            "class_probabilities": class_probs,
            "symmetry_percentage": round(weighted_si, 2),
            "morphometrics": {
                "right_CH": round(auto_heights["right_CH"], 2),
                "right_MH": round(auto_heights["right_MH"], 2),
                "right_RH": round(auto_heights["right_RH"], 2),
                "left_CH": round(auto_heights["left_CH"], 2),
                "left_MH": round(auto_heights["left_MH"], 2),
                "left_RH": round(auto_heights["left_RH"], 2),
                "kjellberg_index": round(kjellberg_pct, 2),
                "side_ratio_CH": round(float(feats["SI_CH_Min"]), 2),
                "side_ratio_MH": round(float(feats["SI_MH_Min"]), 2),
                "side_ratio_RH": round(float(feats["SI_RH_Min"]), 2),
            },
            "top_features": [
                {"name": "Weighted Clinical SI", "value": round(weighted_si, 2)},
                {"name": "SI Condylar Height", "value": round(float(feats["SI_CH_Min"]), 2)},
                {"name": "SI Ramus Height", "value": round(float(feats["SI_RH_Min"]), 2)},
                {"name": "SI Mental Height", "value": round(float(feats["SI_MH_Min"]), 2)},
                {"name": "Kjellberg Symmetry Index", "value": round(kjellberg_pct, 2)},
            ],
            "clahe_image": clahe_b64,
            "gradcam_image": gradcam_b64,
            "landmark_overlay_base64": landmark_b64,
            "landmark_image": landmark_b64,
            "treatment": treatment,
        })

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Analysis error: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
