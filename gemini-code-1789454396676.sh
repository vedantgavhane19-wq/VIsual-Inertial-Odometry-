cat << 'EOF' > smoke_test.py
import time
import numpy as np
from pynq import Overlay, allocate

# 1. Attach to overlay and IP blocks
ol = Overlay("harris_kv260.bit")
dma = ol.axi_dma_0
harris_ip = ol.corner_harris_accel_0

# 2. Configure 640x480 resolution
WIDTH = 640
HEIGHT = 480
harris_ip.register_map.rows = HEIGHT
harris_ip.register_map.cols = WIDTH
harris_ip.register_map.threshold = 442
harris_ip.register_map.k_scaled = 2621

# 3. Allocate physically contiguous buffers
in_buf = allocate(shape=(HEIGHT, WIDTH), dtype=np.uint8)
out_buf = allocate(shape=(HEIGHT, WIDTH), dtype=np.uint8)

# 4. Generate a synthetic test pattern (white square with sharp corners on black background)
test_pattern = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
test_pattern[100:300, 100:300] = 255
np.copyto(in_buf, test_pattern)

print("[INFO] Initiating DMA transfers...")
start_time = time.time()

# 5. Start HLS core and trigger bidirectional DMA (with debug tracing)
harris_ip.register_map.CTRL.AP_START = 1
print("[DEBUG] CTRL after start:", harris_ip.register_map.CTRL)

dma.sendchannel.transfer(in_buf)
print("[DEBUG] Send transfer issued")
dma.sendchannel.wait()
print("[DEBUG] Send done")

dma.recvchannel.transfer(out_buf)
print("[DEBUG] Recv transfer issued")
dma.recvchannel.wait()
print("[DEBUG] Recv done")

print("[DEBUG] CTRL after transfers:", harris_ip.register_map.CTRL)

elapsed_ms = (time.time() - start_time) * 1000.0
corners_found = np.count_nonzero(out_buf == 255)

print(f"[SUCCESS] Roundtrip completed in {elapsed_ms:.2f} ms")
print(f"[INFO] Corner markers generated: {corners_found}")

in_buf.freebuffer()
out_buf.freebuffer()
EOF
sudo /usr/local/share/pynq-venv/bin/python3 smoke_test.py



import cv2

cap = cv2.VideoCapture(0)
# Force camera to 640x480 resolution instead of high default
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame_count = 0
faces = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Run heavy detection every 2nd or 3rd frame only
    if frame_count % 2 == 0:
        # Downscale 2x for processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # scaleFactor=1.3 and higher minNeighbors reduces CPU iterations
        detected = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(20, 20))
        # Scale bounding box coordinates back to full size
        faces = [(x*2, y*2, w*2, h*2) for (x, y, w, h) in detected]

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face Tracker (Optimized)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
