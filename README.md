# OPG-AsymmetryNet 🦷

**Automated Mandibular Asymmetry Severity Classification from Panoramic OPG Dental X-rays**

A production-ready clinical AI web application that classifies mandibular asymmetry as **Mild**, **Moderate**, or **Severe** using a multimodal pipeline combining DenseNet-121 visual features with 31 scale-invariant morphometric features.

---

## 🏗️ Architecture

```
OPG-AsymmetryNet/
├── final_multimodal_checkpoint.joblib   # Pre-trained model artifact
├── Dockerfile                           # Single-container Docker deployment
├── README.md
└── app/
    ├── backend/                         # FastAPI Python backend
    │   ├── main.py                      # Full inference & API server
    │   └── requirements.txt
    └── frontend/                        # React + Vite + Tailwind frontend
        ├── src/
        │   ├── App.jsx                  # Main dashboard UI
        │   └── main.jsx
        ├── vite.config.js
        └── package.json
```

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- Node.js 18+
- The `final_multimodal_checkpoint.joblib` file in the project root

### 1. Backend

```bash
cd app/backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

### 2. Frontend

```bash
cd app/frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## 🔬 AI Pipeline

### Step A — Image Preprocessing
- Grayscale conversion → crop mandibular ROI (height 15–85%, width 5–95%)
- Bilateral filter (d=5, σ_color=35, σ_space=35) → CLAHE (clipLimit=3.0, 8×8 grid)
- Resize to 224×224 (no horizontal flip)

### Step B — Automated Morphometric Estimation
- Adaptive thresholding + morphological operations to segment mandible
- Estimates bilateral **Condylar Height (CH)**, **Mental Height (MH)**, **Ramus Height (RH)**
- Computes **31 scale-invariant features** including Kjellberg Symmetry Index, weighted SIs, side ratios, compensation gaps

### Step C — Multimodal Prediction
- DenseNet-121 → 1024-D embedding → PCA (6 components) → vis_scaler
- Concatenate [visual_6D ‖ clinical_31D] → GradientBoostingClassifier
- Outputs: Mild / Moderate / Severe with class probabilities

### Step D — Grad-CAM Explainability
- Hooks on `denseblock4.denselayer16.conv2`
- Heatmap overlaid (60% original + 40% JET colormap) on ROI

### Step E — Clinical Recommendations
| Class | Symmetry | Recommendation |
|-------|----------|----------------|
| **Mild** | ≥94% | Routine follow-up every 6–12 months |
| **Moderate** | 90–93% | Orthodontic evaluation, functional appliance |
| **Severe** | <90% | OMFS referral, 3D CBCT, Orthognathic Surgery |

---

## 🐳 Docker Deployment

```bash
# Build and run
docker build -t opg-asymmetrynet .
docker run -p 8000:8000 opg-asymmetrynet
```

---

## ☁️ Cloud Deployment

### Option A: Render (Backend) + Vercel (Frontend)

**Backend on Render:**
1. Connect your GitHub repo to [Render](https://render.com)
2. Create a **Web Service** with:
   - Build Command: `pip install -r app/backend/requirements.txt`
   - Start Command: `uvicorn app.backend.main:app --host 0.0.0.0 --port $PORT`
   - Environment: Python 3.11

**Frontend on Vercel:**
1. Import repo to [Vercel](https://vercel.com)
2. Set Root Directory: `app/frontend`
3. Add Environment Variable: `VITE_API_URL=https://your-render-service.onrender.com`
4. Deploy

### Option B: Hugging Face Spaces (Docker)

1. Create a new Space with **Docker** SDK
2. Upload all project files
3. The `Dockerfile` handles everything — HF Spaces maps port 8000 automatically

### Option C: Railway

```bash
railway login
railway new
railway up
```

Set `PORT=8000` environment variable.

---

## 📡 API Reference

### `POST /api/analyze`

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | OPG X-ray image (PNG/JPG/TIFF) |
| `right_CH` | float (optional) | Manual Condylar Height override (right) |
| `right_MH` | float (optional) | Manual Mental Height override (right) |
| `right_RH` | float (optional) | Manual Ramus Height override (right) |
| `left_CH` | float (optional) | Manual Condylar Height override (left) |
| `left_MH` | float (optional) | Manual Mental Height override (left) |
| `left_RH` | float (optional) | Manual Ramus Height override (left) |

**Response:**
```json
{
  "predicted_class": "Moderate",
  "confidence": 87.3,
  "class_probabilities": {"Mild": 8.2, "Moderate": 87.3, "Severe": 4.5},
  "symmetry_percentage": 92.4,
  "morphometrics": { "right_CH": 45.2, "left_CH": 41.8, "..." : "..." },
  "top_features": ["..."],
  "clahe_image": "<base64 PNG>",
  "gradcam_image": "<base64 PNG>",
  "treatment": { "recommendation": "...", "urgency": "semi-urgent" }
}
```

### `GET /health`

Returns `{"status": "ok", "model_loaded": true}`

---

## ⚠️ Medical Disclaimer

This software is intended for **clinical decision support only**. All AI-generated analyses must be reviewed and confirmed by a qualified dental or medical professional. This tool does not constitute a medical diagnosis.

---

## 📄 License

For research and educational use. Contact the authors for commercial licensing.
