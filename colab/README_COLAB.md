# 🦷 Google Colab Backend Deployment Guide (Smile Therapy)

This guide walks you through running the **full original research model** (DenseNet-121 + 31 Morphometrics + GBDT + Grad-CAM) on Google Colab with **free GPU acceleration** (T4 GPU) and connecting it to your Vercel frontend.

---

## 🚀 Quick Start (Option 1: Using the Jupyter Notebook)

1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **File** -> **Upload notebook** and upload `colab/Smile_Therapy_Colab_Backend.ipynb` (or upload it to your Google Drive).
3. Set Runtime to GPU: Click **Runtime** -> **Change runtime type** -> Select **T4 GPU** -> Click **Save**.
4. In the files sidebar on the left, upload:
   - `main.py` (from `app/backend/main.py`)
   - `final_multimodal_checkpoint.joblib`
   *(Or simply run `!git clone https://github.com/YOUR_USERNAME/web_dental.git`)*
5. Click **Runtime** -> **Run all** (or run each cell one by one).
6. In **Step 4**, Cloudflare Tunnel will start and print your public URL:
   ```
   =================================================================
     🎉 SMILE THERAPY AI BACKEND IS LIVE ON GOOGLE COLAB! 🎉
   =================================================================

     👉 PUBLIC API URL:   https://your-unique-id.trycloudflare.com
     👉 HEALTH CHECK:     https://your-unique-id.trycloudflare.com/health

     📋 Copy the URL above and paste it in your Vercel frontend!
   =================================================================
   ```
7. Copy `https://your-unique-id.trycloudflare.com`.
8. Open your Vercel-deployed frontend website (or localhost), click the **"Backend Server"** / **"Colab API"** button in the header, paste this URL, and click **"Test & Save"**!

---

## ⚡ Option 2: Run via Terminal / Colab Cells directly

Open a blank Google Colab notebook and run:

```bash
# 1. Install dependencies
!pip install -q fastapi uvicorn python-multipart opencv-python-headless joblib scikit-learn==1.4.2 torch torchvision Pillow

# 2. Install Cloudflared (100% free tunnel, no signup needed)
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb > /dev/null 2>&1

# 3. Clone or upload your files
!git clone https://github.com/YOUR_GITHUB_REPO.git
%cd YOUR_GITHUB_REPO/app/backend

# 4. Run the Colab runner
!python /content/YOUR_GITHUB_REPO/colab/run_colab.py
```

---

## 💡 Why Google Colab + Cloudflare Tunnel is the best choice:
1. **Full Original Model Accuracy**: Runs the original DenseNet-121 feature extractor with 100% fidelity. No approximations or compromises.
2. **GPU Acceleration**: DenseNet inference runs in ~15 milliseconds on Colab's T4 GPU.
3. **Free & Generous Memory**: Colab provides 12.7 GB to 15 GB of RAM (no 512 MB memory limit crashes).
4. **Free Cloudflare Tunnel**: No account, no credit card, no token required. Gives a secure `https://*.trycloudflare.com` URL with SSL and zero request limits.
