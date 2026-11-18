import time, sys, io, json, asyncio, cv2, os

sys.path.insert(0, r'd:\CSE\thesis\web_dental\app\backend')
import main
from fastapi import UploadFile

SAMPLES = [
    ('Mild',     r'd:\CSE\thesis\web_dental\app\backend\public\samples\mild_sample.jpg'),
    ('Moderate', r'd:\CSE\thesis\web_dental\app\backend\public\samples\moderate_sample.jpg'),
    ('Severe',   r'd:\CSE\thesis\web_dental\app\backend\public\samples\severe_sample.jpg'),
]

async def test():
    t = time.time()
    await main.startup_event()
    print(f"[STARTUP] {time.time()-t:.2f}s")

    for label, path in SAMPLES:
        if not os.path.exists(path):
            print(f"[SKIP] {label}: not found at {path}")
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"[SKIP] {label}: imread failed")
            continue
        _, buf = cv2.imencode('.jpg', img)
        t0 = time.time()
        up = UploadFile(filename='t.jpg', file=io.BytesIO(buf.tobytes()))
        res = await main.analyze(up)
        elapsed = time.time() - t0
        data = json.loads(res.body.decode())
        pc = data['predicted_class']
        conf = data['confidence']
        si = data['symmetry_percentage']
        gc = len(data.get('gradcam_image', ''))
        lm = len(data.get('landmark_overlay_base64', ''))
        morph = data.get('morphometrics', {})
        print(f"[{label}] predicted={pc} conf={conf:.1f}% SI={si:.1f}% | {elapsed:.2f}s | gradcam={gc} lm={lm}")
        print(f"         CH_R={morph.get('R_CH',0):.1f} CH_L={morph.get('L_CH',0):.1f} RH_R={morph.get('R_RH',0):.1f} RH_L={morph.get('L_RH',0):.1f}")

asyncio.run(test())
print("SMOKE TEST PASSED")
