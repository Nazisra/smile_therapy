import sys, cv2, asyncio, json, io, time, os

sys.path.insert(0, r'D:\CSE\thesis\web_dental\app\backend')

# Measure import time
t_import = time.time()
import main
import_time = time.time() - t_import
print(f"[IMPORT] main.py imported in {import_time:.2f}s")

from fastapi import UploadFile

SAMPLE_PATHS = [
    ('Mild',     r'd:\CSE\thesis\web_dental\app\backend\public\samples\mild_sample.jpg'),
    ('Moderate', r'd:\CSE\thesis\web_dental\app\backend\public\samples\moderate_sample.jpg'),
    ('Severe',   r'd:\CSE\thesis\web_dental\app\backend\public\samples\severe_sample.jpg'),
]

async def test():
    t_startup = time.time()
    await main.startup_event()
    print(f"[STARTUP] done in {time.time()-t_startup:.2f}s")

    for label, path in SAMPLE_PATHS:
        if not os.path.exists(path):
            print(f"[SKIP] {label}: file not found at {path}")
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"[SKIP] {label}: cv2.imread failed")
            continue
        _, buf = cv2.imencode('.jpg', img)
        t0 = time.time()
        upfile = UploadFile(filename='test.jpg', file=io.BytesIO(buf.tobytes()))
        res = await main.analyze(upfile)
        elapsed = time.time() - t0
        data = json.loads(res.body.decode())
        gc_len = len(data.get('gradcam_image', ''))
        lm_len = len(data.get('landmark_overlay_base64', ''))
        print(f"[{label}] predicted={data['predicted_class']} conf={data['confidence']:.1f}% "
              f"SI={data['symmetry_percentage']:.1f}% | {elapsed:.2f}s | "
              f"gradcam={gc_len} chars | landmarks={lm_len} chars")

asyncio.run(test())
print("SMOKE TEST PASSED")
