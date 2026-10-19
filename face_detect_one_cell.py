from pynq import Overlay, allocate
import numpy as np
import cv2
import json
import time

# Load the bitstream (PYNQ parses face_detect.hwh automatically)
ol = Overlay("face_detect.bit")

# Access the accelerator IP
accel = ol.face_detect_0

# Verify register accessibility
print("IP Status: Loaded successfully")
print(accel.register_map)



# 1. Allocate DMA buffers (physically contiguous memory)
# Frame buffer: 320x240 8-bit grayscale
img_buf = allocate(shape=(240, 320), dtype=np.uint8)

# Output buffer: 4096 32-bit integers ((y << 16) | x)
MAX_HITS = 4096
hits_buf = allocate(shape=(MAX_HITS,), dtype=np.uint32)

def run_hardware_detector(frame_gray):
    """Sends one scale level to the PL accelerator and reads back hits."""
    h, w = frame_gray.shape
    if h > 240 or w > 320:
        raise ValueError(f"Resolution {w}x{h} exceeds hardware maximum 320x240.")

    # Copy frame to buffer and push to DDR
    img_buf[:h, :w] = frame_gray
    img_buf.sync_to_device()

    # Pass 64-bit addresses into register map
    frame_addr = img_buf.device_address
    hits_addr = hits_buf.device_address

    accel.register_map.frame_1 = frame_addr & 0xFFFFFFFF
    accel.register_map.frame_2 = (frame_addr >> 32) & 0xFFFFFFFF
    accel.register_map.hits_1 = hits_addr & 0xFFFFFFFF
    accel.register_map.hits_2 = (hits_addr >> 32) & 0xFFFFFFFF

    accel.register_map.width = w
    accel.register_map.height = h

    # Start accelerator (AP_START = 1)
    accel.register_map.CTRL.AP_START = 1

    # Poll for completion (AP_DONE = 1)
    while not accel.register_map.CTRL.AP_DONE:
        pass

    # Invalidate CPU cache to read updated hits
    hits_buf.sync_from_device()
    n_hits = accel.register_map.ap_return

    # Extract detected (x, y) coordinates
    count = min(n_hits, MAX_HITS)
    raw = hits_buf[:count]
    xs = (raw & 0xFFFF).astype(int)
    ys = ((raw >> 16) & 0xFFFF).astype(int)

    return xs, ys



def group_rects(boxes, min_neighbors=3, eps=0.2):
    """Merges overlapping candidate boxes from multiple pyramid scales."""
    n = len(boxes)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def similar(a, b):
        d = eps * (min(a[2], b[2]) + min(a[3], b[3])) * 0.5
        return (abs(a[0] - b[0]) <= d and abs(a[1] - b[1]) <= d
                and abs(a[0] + a[2] - b[0] - b[2]) <= d
                and abs(a[1] + a[3] - b[1] - b[3]) <= d)

    for i in range(n):
        for j in range(i + 1, n):
            if similar(boxes[i], boxes[j]):
                parent[find(i)] = find(j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(boxes[i])

    out = []
    for members in clusters.values():
        if len(members) >= min_neighbors:
            m = np.mean(np.array(members, dtype=np.float64), axis=0)
            out.append(tuple(int(round(v)) for v in m))
    return out



# Test on a static image before testing the camera stream
test_img = cv2.imread("test.jpg")
if test_img is None:
    from skimage import data
    test_img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)

gray_full = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
# Resize input base frame to 320x240
gray_base = cv2.resize(gray_full, (320, 240), interpolation=cv2.INTER_LINEAR)

SCALE = 1.2
W, H = 24, 24
candidate_boxes = []
s = 1.0

# Multi-scale image pyramid
while gray_base.shape[0] / s >= H and gray_base.shape[1] / s >= W:
    nh, nw = int(round(gray_base.shape[0] / s)), int(round(gray_base.shape[1] / s))
    lvl = cv2.resize(gray_base, (nw, nh), interpolation=cv2.INTER_LINEAR)

    xs, ys = run_hardware_detector(lvl)
    for x, y in zip(xs, ys):
        candidate_boxes.append([int(x * s), int(y * s), int(W * s), int(H * s)])
    s *= SCALE

# Merge boxes
faces = group_rects(candidate_boxes, min_neighbors=3)

# Display result inside Jupyter
result = cv2.resize(test_img, (320, 240))
for (x, y, w, h) in faces:
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title(f"Detected Faces: {len(faces)}")
plt.show()




from IPython.display import display, Image, clear_output

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("USB webcam not accessible on /dev/video0.")

print("Starting live face detection. Stop the notebook cell to exit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        base = cv2.resize(gray, (320, 240), interpolation=cv2.INTER_LINEAR)

        candidate_boxes = []
        s = 1.0

        # Scan scales
        while base.shape[0] / s >= H and base.shape[1] / s >= W:
            nh, nw = int(round(base.shape[0] / s)), int(round(base.shape[1] / s))
            lvl = cv2.resize(base, (nw, nh), interpolation=cv2.INTER_LINEAR)

            xs, ys = run_hardware_detector(lvl)
            for x, y in zip(xs, ys):
                # Scale coordinates back up to 640x480 webcam frame resolution
                scale_up = 640.0 / 320.0
                candidate_boxes.append([
                    int(x * s * scale_up),
                    int(y * s * scale_up),
                    int(W * s * scale_up),
                    int(H * s * scale_up)
                ])
            s *= SCALE

        faces = group_rects(candidate_boxes, min_neighbors=3)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Render output frame directly in Jupyter
        _, jpeg = cv2.imencode('.jpg', frame)
        clear_output(wait=True)
        display(Image(data=jpeg.tobytes()))

except KeyboardInterrupt:
    print("Stream stopped.")
finally:
    cap.release()
    img_buf.freebuffer()
    hits_buf.freebuffer()
