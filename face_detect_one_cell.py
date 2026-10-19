import time
import cv2
import numpy as np
from IPython.display import Image, clear_output, display
from pynq import MMIO, Overlay, allocate

# -------------------------------------------------------------------------
# 1. Hardware & Register Definitions
# -------------------------------------------------------------------------
BIT_FILE = "face_detect.bit"  # face_detect.hwh must be in the same folder
CTRL_BASE = 0xA0000000  # s_axi_control base address
MAX_W, MAX_H, MAX_HITS = 320, 240, 4096
WIN = 24  # Base window size (24x24)

# Register offsets (from HLS hardware handoff)
REG_CTRL = 0x00
REG_RET = 0x10
REG_FRAME_LO, REG_FRAME_HI = 0x18, 0x1C
REG_HITS_LO, REG_HITS_HI = 0x24, 0x28
REG_WIDTH, REG_HEIGHT = 0x30, 0x38

# Download overlay and map control registers
ol = Overlay(BIT_FILE)
ip = MMIO(CTRL_BASE, 0x10000)

# Allocate physically contiguous DMA buffers
frame_buf = allocate(shape=(MAX_W * MAX_H,), dtype=np.uint8)
hits_buf = allocate(shape=(MAX_HITS,), dtype=np.uint32)

# -------------------------------------------------------------------------
# 2. Kernel Driver & Box Grouping
# -------------------------------------------------------------------------
def run_kernel(img, timeout_s=3.0):
  """Sends one pyramid level to the FPGA accelerator and returns hits."""
  h, w = img.shape

  # Copy frame to DMA memory and flush CPU cache
  frame_buf[: w * h] = img.ravel()
  frame_buf.sync_to_device()

  # Write physical memory addresses and image dimensions
  fa = int(frame_buf.device_address)
  ha = int(hits_buf.device_address)
  ip.write(REG_FRAME_LO, fa & 0xFFFFFFFF)
  ip.write(REG_FRAME_HI, fa >> 32)
  ip.write(REG_HITS_LO, ha & 0xFFFFFFFF)
  ip.write(REG_HITS_HI, ha >> 32)
  ip.write(REG_WIDTH, w)
  ip.write(REG_HEIGHT, h)

  # Start accelerator (ap_start)
  ip.write(REG_CTRL, 0x1)

  t0 = time.time()
  while not (ip.read(REG_CTRL) & 0x2):
    if time.time() - t0 > timeout_s:
      raise TimeoutError("Accelerator execution timed out.")

  # Invalidate CPU cache to fetch newly written results from DDR
  n = ip.read(REG_RET)
  hits_buf.sync_from_device()

  k = min(n, MAX_HITS)
  v = np.array(hits_buf[:k], dtype=np.uint32)
  return (v & 0xFFFF).astype(int), (v >> 16).astype(int), n


def group_rects(boxes, min_neighbors=3, eps=0.2):
  """Merges multi-scale overlapping boxes."""
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
    return (
        abs(a[0] - b[0]) <= d
        and abs(a[1] - b[1]) <= d
        and abs(a[0] + a[2] - b[0] - b[2]) <= d
        and abs(a[1] + a[3] - b[1] - b[3]) <= d
    )

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
      out.append(tuple(int(round(x)) for x in m))
  return out


# Updated detection parameters for speed and accuracy
def detect_faces_fast(
    gray_320, scale=1.25, min_neighbors=5, min_size=50, max_levels=6
):
  """scale=1.25: Reduces pyramid levels from ~13 down to ~5-6 passes.

  min_neighbors=5: Filters out noise/false positives (like the window/wall).
  min_size=50: Skips excessively small sub-windows.
  max_levels=6: Caps the search depth.
  """
  boxes = []
  s = 1.0
  lvl_count = 0

  while (
      gray_320.shape[0] / s >= WIN
      and gray_320.shape[1] / s >= WIN
      and lvl_count < max_levels
  ):
    nh, nw = int(round(gray_320.shape[0] / s)), int(round(gray_320.shape[1] / s))
    lvl = cv2.resize(gray_320, (nw, nh), interpolation=cv2.INTER_LINEAR)

    xs, ys, _ = run_kernel(lvl)
    for x, y in zip(xs, ys):
      boxes.append([int(x * s), int(y * s), int(WIN * s), int(WIN * s)])

    s *= scale
    lvl_count += 1

  return [b for b in group_rects(boxes, min_neighbors=min_neighbors) if b[2] >= min_size]


# -------------------------------------------------------------------------
# 3. Live Webcam Capture Loop
# -------------------------------------------------------------------------
# Replace Section 3 in your single cell with this optimized loop:
CAM_INDEX = 0
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Optional: set camera backend buffer size to avoid frame queuing delay
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

for _ in range(5):
  cap.read()

if not cap.isOpened():
  raise RuntimeError(f"Cannot open camera at index {CAM_INDEX}")

print("Fast Detection Running. Press 'Interrupt' to stop.")

try:
  t_prev = time.time()
  frame_count = 0
  boxes = []

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    frame_count += 1

    # Run detection only on every 2nd frame to boost effective frame rate
    if frame_count % 2 == 0 or len(boxes) == 0:
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      small = cv2.resize(gray, (MAX_W, MAX_H), interpolation=cv2.INTER_NEAREST)
      # Use the optimized fast detector
      boxes = detect_faces_fast(
          small, scale=1.25, min_neighbors=5, min_size=50, max_levels=6
      )

    # Draw boxes
    sx, sy = frame.shape[1] / MAX_W, frame.shape[0] / MAX_H
    for x, y, w, h in boxes:
      cv2.rectangle(
          frame,
          (int(x * sx), int(y * sy)),
          (int((x + w) * sx), int((y + h) * sy)),
          (0, 255, 0),
          2,
      )

    # Calculate FPS
    t_curr = time.time()
    fps = 1.0 / (t_curr - t_prev) if (t_curr - t_prev) > 0 else 0.0
    t_prev = t_curr

    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | Faces: {len(boxes)}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    # Render only every 2nd display cycle to reduce Jupyter HTTP overhead
    if frame_count % 2 == 0:
      _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
      clear_output(wait=True)
      display(Image(data=jpeg.tobytes()))

except KeyboardInterrupt:
  print("Stream stopped by user.")
finally:
  cap.release()
  frame_buf.freebuffer()
  hits_buf.freebuffer()






import time
import cv2
import numpy as np
from IPython.display import Image, clear_output, display
from pynq import MMIO, Overlay, allocate

# -------------------------------------------------------------------------
# 1. Hardware, Overlay & Memory Allocations
# -------------------------------------------------------------------------
BIT_FILE  = "face_detect.bit"     # Must be in the same folder as face_detect.hwh
CTRL_BASE = 0xA0000000            # Matches Vivado Address Editor

MAX_W, MAX_H, MAX_HITS = 320, 240, 4096
WIN = 24                          # Base 24x24 Haar window size

# Register offsets (from HLS hardware handoff)
REG_CTRL                   = 0x00
REG_RET                    = 0x10
REG_FRAME_LO, REG_FRAME_HI = 0x18, 0x1C
REG_HITS_LO,  REG_HITS_HI  = 0x24, 0x28
REG_WIDTH,    REG_HEIGHT   = 0x30, 0x38

# Load bitstream & memory map
ol = Overlay(BIT_FILE)
ip = MMIO(CTRL_BASE, 0x10000)

# Allocate physically contiguous DMA buffers
frame_buf = allocate(shape=(MAX_W * MAX_H,), dtype=np.uint8)
hits_buf  = allocate(shape=(MAX_HITS,),      dtype=np.uint32)

# -------------------------------------------------------------------------
# 2. Hardware Driver & Clustering Helpers
# -------------------------------------------------------------------------
def run_kernel(img, timeout_s=3.0):
    """Executes the FPGA face_detect IP core on one grayscale image level."""
    h, w = img.shape

    # Transfer image data to DDR and flush CPU cache
    frame_buf[: w * h] = img.ravel()
    frame_buf.sync_to_device()

    # Pass 64-bit physical memory addresses & dimensions
    fa = int(frame_buf.device_address)
    ha = int(hits_buf.device_address)
    ip.write(REG_FRAME_LO, fa & 0xFFFFFFFF)
    ip.write(REG_FRAME_HI, fa >> 32)
    ip.write(REG_HITS_LO,  ha & 0xFFFFFFFF)
    ip.write(REG_HITS_HI,  ha >> 32)
    ip.write(REG_WIDTH,  w)
    ip.write(REG_HEIGHT, h)

    # Start accelerator (ap_start = bit 0)
    ip.write(REG_CTRL, 0x1)

    t0 = time.time()
    while not (ip.read(REG_CTRL) & 0x2):
        if time.time() - t0 > timeout_s:
            raise TimeoutError("Accelerator execution timed out.")

    # Read back hit count & sync CPU cache
    n = ip.read(REG_RET)
    hits_buf.sync_from_device()

    k = min(n, MAX_HITS)
    v = np.array(hits_buf[:k], dtype=np.uint32)
    return (v & 0xFFFF).astype(int), (v >> 16).astype(int), n


def group_rects(boxes, min_neighbors=4, eps=0.2):
    """Merges candidate bounding boxes across pyramid scales."""
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
            out.append(tuple(int(round(x)) for x in m))
    return out


def detect_faces_tuned(frame_bgr, min_neighbors=4, min_size=40):
    """Fast multi-scale scan using 4 targeted resolution steps."""
    BASE_W, BASE_H = 240, 180
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    base = cv2.resize(gray, (BASE_W, BASE_H), interpolation=cv2.INTER_LINEAR)

    SCALES = [1.0, 1.35, 1.8, 2.4]
    boxes = []

    for s in SCALES:
        nw = int(round(BASE_W / s))
        nh = int(round(BASE_H / s))

        if nw < WIN or nh < WIN:
            break

        lvl = cv2.resize(base, (nw, nh), interpolation=cv2.INTER_NEAREST)
        xs, ys, n = run_kernel(lvl)

        norm_x = (640.0 / BASE_W) * s
        norm_y = (480.0 / BASE_H) * s

        for x, y in zip(xs, ys):
            boxes.append([
                int(x * norm_x),
                int(y * norm_y),
                int(WIN * norm_x),
                int(WIN * norm_y)
            ])

    return [b for b in group_rects(boxes, min_neighbors=min_neighbors) if b[2] >= min_size]

# -------------------------------------------------------------------------
# 3. Live Webcam Stream Loop
# -------------------------------------------------------------------------
CAM_INDEX = 0
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

for _ in range(5):
    cap.read()

if not cap.isOpened():
    raise RuntimeError(f"Cannot access camera on /dev/video{CAM_INDEX}")

print("Detector running. Press Stop/Interrupt to exit.")

try:
    t_prev = time.time()
    frame_idx = 0
    active_faces = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Detect every 2nd frame; keep previous tracking on odd frames
        if frame_idx % 2 == 0 or len(active_faces) == 0:
            active_faces = detect_faces_tuned(frame, min_neighbors=4, min_size=40)

        for (x, y, w, h) in active_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        t_curr = time.time()
        fps = 1.0 / (t_curr - t_prev) if (t_curr - t_prev) > 0 else 0.0
        t_prev = t_curr

        cv2.putText(frame, f"FPS: {fps:.1f} | Faces: {len(active_faces)}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        clear_output(wait=True)
        display(Image(data=jpeg.tobytes()))

except KeyboardInterrupt:
    print("Stream stopped.")
finally:
    cap.release()
    frame_buf.freebuffer()
    hits_buf.freebuffer()






import cv2
import numpy as np

# Capture a single test frame
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Test 1: Without equalizeHist (matches OpenCV cascade training)
small_raw = cv2.resize(gray, (320, 240), interpolation=cv2.INTER_LINEAR)
xs, ys, count = run_kernel(small_raw)
print(f"Raw hits (Standard Grayscale): {count}")

# Test 2: With equalizeHist
small_eq = cv2.resize(
    cv2.equalizeHist(gray), (320, 240), interpolation=cv2.INTER_LINEAR
)
xs, ys, count_eq = run_kernel(small_eq)
print(f"Raw hits (Histogram Equalized): {count_eq}")





import time
import cv2
from IPython.display import Image, clear_output, display
import numpy as np
from pynq import MMIO, Overlay, allocate

# -------------------------------------------------------------------------
# 1. Hardware, Overlay & Memory Allocations
# -------------------------------------------------------------------------
BIT_FILE = "face_detect.bit"  # Must be in the same folder as face_detect.hwh[cite: 7]
CTRL_BASE = 0xA0000000  # Matches Vivado Address Editor[cite: 5, 7]

MAX_W, MAX_H, MAX_HITS = 320, 240, 4096  #[cite: 5]
WIN = 24  # Base 24x24 Haar window size[cite: 3]

# Register offsets (from HLS hardware handoff)[cite: 6]
REG_CTRL = 0x00
REG_RET = 0x10
REG_FRAME_LO, REG_FRAME_HI = 0x18, 0x1C
REG_HITS_LO, REG_HITS_HI = 0x24, 0x28
REG_WIDTH, REG_HEIGHT = 0x30, 0x38

# Load bitstream & memory map
ol = Overlay(BIT_FILE)
ip = MMIO(CTRL_BASE, 0x10000)

# Allocate physically contiguous DMA buffers
frame_buf = allocate(shape=(MAX_W * MAX_H,), dtype=np.uint8)
hits_buf = allocate(shape=(MAX_HITS,), dtype=np.uint32)


# -------------------------------------------------------------------------
# 2. Hardware Driver & Clustering Helpers
# -------------------------------------------------------------------------
def run_kernel(img, timeout_s=3.0):
  """Executes the FPGA face_detect IP core on one grayscale image level."""
  h, w = img.shape

  # Transfer image data to DDR and flush CPU cache
  frame_buf[: w * h] = img.ravel()
  frame_buf.sync_to_device()

  # Pass 64-bit physical memory addresses & dimensions[cite: 6, 7]
  fa = int(frame_buf.device_address)
  ha = int(hits_buf.device_address)
  ip.write(REG_FRAME_LO, fa & 0xFFFFFFFF)
  ip.write(REG_FRAME_HI, fa >> 32)
  ip.write(REG_HITS_LO, ha & 0xFFFFFFFF)
  ip.write(REG_HITS_HI, ha >> 32)
  ip.write(REG_WIDTH, w)
  ip.write(REG_HEIGHT, h)

  # Start accelerator (ap_start = bit 0)[cite: 6]
  ip.write(REG_CTRL, 0x1)

  t0 = time.time()
  while not (ip.read(REG_CTRL) & 0x2):  #[cite: 6]
    if time.time() - t0 > timeout_s:
      raise TimeoutError("Accelerator execution timed out.")

  # Read back hit count & sync CPU cache[cite: 6]
  n = ip.read(REG_RET)
  hits_buf.sync_from_device()

  k = min(n, MAX_HITS)  #[cite: 5]
  v = np.array(hits_buf[:k], dtype=np.uint32)
  return (v & 0xFFFF).astype(int), (v >> 16).astype(int), n  #[cite: 5, 7]


def group_rects(boxes, min_neighbors=4, eps=0.2):
  """Merges candidate bounding boxes across pyramid scales."""
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
    return (
        abs(a[0] - b[0]) <= d
        and abs(a[1] - b[1]) <= d
        and abs(a[0] + a[2] - b[0] - b[2]) <= d
        and abs(a[1] + a[3] - b[1] - b[3]) <= d
    )

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
      out.append(tuple(int(round(x)) for x in m))
  return out


def detect_faces_tuned(frame_bgr, min_neighbors=4, min_size=40):
  """Fast multi-scale scan using 4 targeted resolution steps."""
  BASE_W, BASE_H = 240, 180
  gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
  base = cv2.resize(gray, (BASE_W, BASE_H), interpolation=cv2.INTER_LINEAR)

  SCALES = [1.0, 1.35, 1.8, 2.4]
  boxes = []

  for s in SCALES:
    nw = int(round(BASE_W / s))
    nh = int(round(BASE_H / s))

    if nw < WIN or nh < WIN:
      break

    lvl = cv2.resize(base, (nw, nh), interpolation=cv2.INTER_NEAREST)
    xs, ys, n = run_kernel(lvl)

    norm_x = (640.0 / BASE_W) * s
    norm_y = (480.0 / BASE_H) * s

    for x, y in zip(xs, ys):
      boxes.append([
          int(x * norm_x),
          int(y * norm_y),
          int(WIN * norm_x),
          int(WIN * norm_y),
      ])

  return [
      b
      for b in group_rects(boxes, min_neighbors=min_neighbors)
      if b[2] >= min_size
  ]


# -------------------------------------------------------------------------
# 3. Live Webcam Stream Loop (Paced at 15 FPS)
# -------------------------------------------------------------------------
CAM_INDEX = 0
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

for _ in range(5):
  cap.read()

if not cap.isOpened():
  raise RuntimeError(f"Cannot access camera on /dev/video{CAM_INDEX}")

TARGET_FPS = 15.0
FRAME_BUDGET = 1.0 / TARGET_FPS  # 0.0667 seconds per frame

print(f"Detector running locked at {TARGET_FPS} FPS. Press Stop to exit.")

try:
  t_prev = time.time()
  fps_smooth = TARGET_FPS
  alpha = 0.1
  frame_idx = 0
  active_faces = []

  while True:
    t_frame_start = time.time()

    ret, frame = cap.read()
    if not ret:
      break

    frame_idx += 1

    # Detect faces on even frames; preserve previous positions on odd frames
    if frame_idx % 2 == 0 or len(active_faces) == 0:
      active_faces = detect_faces_tuned(frame, min_neighbors=3, min_size=40)

    # Render bounding boxes
    for x, y, w, h in active_faces:
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw smoothed FPS counter on frame
    cv2.putText(
        frame,
        f"FPS: {fps_smooth:.1f} | Faces: {len(active_faces)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    # Encode and stream to Jupyter
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
    clear_output(wait=True)
    display(Image(data=jpeg.tobytes()))

    # --- Frame Rate Pacer ---
    t_process_done = time.time()
    elapsed = t_process_done - t_frame_start
    sleep_needed = FRAME_BUDGET - elapsed

    if sleep_needed > 0:
      time.sleep(sleep_needed)

    # Actual loop timing calculation
    t_curr = time.time()
    dt = t_curr - t_prev
    t_prev = t_curr

    if dt > 0:
      inst_fps = 1.0 / dt
      fps_smooth = alpha * inst_fps + (1.0 - alpha) * fps_smooth

except KeyboardInterrupt:
  print("Stream stopped.")
finally:
  cap.release()
  frame_buf.freebuffer()
  hits_buf.freebuffer()
