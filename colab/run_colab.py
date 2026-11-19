"""
Smile Therapy - Google Colab Backend Runner
Runs FastAPI with original DenseNet-121 + GBDT and creates a public Cloudflare / ngrok tunnel.
"""

import os
import sys
import time
import subprocess
import re
import urllib.request

def install_cloudflared():
    if os.path.exists("/usr/local/bin/cloudflared") or os.path.exists("/usr/bin/cloudflared"):
        print("[✓] Cloudflared already installed.")
        return
    print("[*] Downloading and installing Cloudflared...")
    os.system("wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb")
    os.system("dpkg -i cloudflared-linux-amd64.deb > /dev/null 2>&1")
    print("[✓] Cloudflared installed successfully.")

def start_backend():
    print("[*] Starting Smile Therapy Backend on port 8000...")
    # Start uvicorn in the background
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Wait for server to become responsive
    for _ in range(30):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1) as resp:
                if resp.status == 200:
                    print("[✓] FastAPI backend is UP and running!")
                    break
        except Exception:
            time.sleep(1)
    else:
        print("[!] Backend is still starting or initializing weights...")

    return backend_proc

def start_tunnel():
    print("[*] Establishing secure Cloudflare tunnel...")
    cf_proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    
    tunnel_url = None
    pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    
    # Read cloudflared output to find the public URL
    start_time = time.time()
    while time.time() - start_time < 30:
        line = cf_proc.stdout.readline()
        if not line:
            continue
        match = pattern.search(line)
        if match:
            tunnel_url = match.group(0)
            break
            
    if tunnel_url:
        print("\n" + "=" * 65)
        print("  🎉 SMILE THERAPY AI BACKEND IS LIVE ON GOOGLE COLAB! 🎉")
        print("=" * 65)
        print(f"\n  👉 PUBLIC API URL:   {tunnel_url}")
        print(f"  👉 HEALTH CHECK URL: {tunnel_url}/health\n")
        print("  📋 Paste this URL into your Vercel Frontend (Server Config)")
        print("=" * 65 + "\n")
    else:
        print("[!] Could not auto-detect Cloudflare URL. Check logs below:")
        
    return cf_proc

if __name__ == "__main__":
    install_cloudflared()
    b_proc = start_backend()
    c_proc = start_tunnel()
    
    try:
        # Keep process alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")
        b_proc.terminate()
        c_proc.terminate()
