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


def detect_faces(gray_320, scale=1.2, min_neighbors=3, min_size=30):
  """Runs the multi-scale pyramid scan across hardware levels."""
  boxes = []
  s = 1.0
  while gray_320.shape[0] / s >= WIN and gray_320.shape[1] / s >= WIN:
    nh, nw = int(round(gray_320.shape[0] / s)), int(
        round(gray_320.shape[1] / s)
    )
    lvl = cv2.resize(gray_320, (nw, nh), interpolation=cv2.INTER_LINEAR)

    xs, ys, _ = run_kernel(lvl)
    for x, y in zip(xs, ys):
      boxes.append([int(x * s), int(y * s), int(WIN * s), int(WIN * s)])
    s *= scale

  return [b for b in group_rects(boxes, min_neighbors) if b[2] >= min_size]


# -------------------------------------------------------------------------
# 3. Live Webcam Capture Loop
# -------------------------------------------------------------------------
CAM_INDEX = 0  # Change to 1 or 2 if /dev/video0 is not your USB webcam
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

for _ in range(5):  # Let auto-exposure settle
  cap.read()

if not cap.isOpened():
  raise RuntimeError(
      f"Cannot open camera at index {CAM_INDEX}. Check `ls -l /dev/video*`."
  )

print("Accelerator running. Press 'Interrupt' (stop button) to exit.")

try:
  t_prev = time.time()
  while True:
    ret, frame = cap.read()
    if not ret:
      break

    # Downsample to maximum hardware level: 320x240
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (MAX_W, MAX_H), interpolation=cv2.INTER_AREA)

    # Detect faces via PL accelerator
    boxes = detect_faces(small)

    # Scale bounding boxes up from 320x240 to 640x480 frame coordinates
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

    # Display inside notebook
    _, jpeg = cv2.imencode(".jpg", frame)
    clear_output(wait=True)
    display(Image(data=jpeg.tobytes()))

except KeyboardInterrupt:
  print("Stream stopped by user.")
finally:
  cap.release()
  frame_buf.freebuffer()
  hits_buf.freebuffer()
