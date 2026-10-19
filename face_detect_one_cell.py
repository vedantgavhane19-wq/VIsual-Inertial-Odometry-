# ============================================================================
# KV260 face detection - everything in ONE cell
# Needs next to this notebook: face_detect.bit, face_detect.hwh, tb_data/
# ============================================================================
RUN_SELFTEST = True      # compare hardware against the golden-model test data
WEBCAM_MODE  = "still"   # "still" = one frame, "live" = loop, None = no webcam
CAM_INDEX    = 0         # check with:  ls /dev/video*
MAX_FRAMES   = 300       # live mode length (Kernel > Interrupt also stops it)

# ---- 1. Load the bitstream and set up the accelerator ----------------------
import time
import numpy as np
import cv2
from pynq import Overlay, allocate, MMIO

BIT_FILE  = "face_detect.bit"     # face_detect.hwh must be next to it
CTRL_BASE = 0xA0000000            # s_axi_control base address (Address Editor / .hwh)

# Must match hls/face_detect.h
MAX_W, MAX_H, MAX_HITS = 320, 240, 4096
WIN = 24                          # Haar window size

# Register offsets, taken from design_1.hwh
REG_CTRL                   = 0x00   # bit0 ap_start, bit1 ap_done, bit2 ap_idle
REG_RET                    = 0x10   # number of hits
REG_FRAME_LO, REG_FRAME_HI = 0x18, 0x1C
REG_HITS_LO,  REG_HITS_HI  = 0x24, 0x28
REG_WIDTH,    REG_HEIGHT   = 0x30, 0x38

ol = Overlay(BIT_FILE)            # downloads the bitstream to the PL
ip = MMIO(CTRL_BASE, 0x10000)

frame_buf = allocate(shape=(MAX_W * MAX_H,), dtype=np.uint8)    # input frame in DDR
hits_buf  = allocate(shape=(MAX_HITS,),      dtype=np.uint32)   # output hits in DDR

print("bitstream loaded; accelerator idle:", bool(ip.read(REG_CTRL) & 0x4))

# ---- 2. Helpers: run the kernel, image pyramid, box grouping --------------
def run_kernel(img, timeout_s=5.0):
    """Run the accelerator on one 8-bit grayscale image (<= 320x240).
    Returns (xs, ys, n_total) - top-left corners of every 24x24 window that passed."""
    h, w = img.shape
    assert img.dtype == np.uint8 and w <= MAX_W and h <= MAX_H, "need uint8 image <= 320x240"

    frame_buf[: w * h] = img.ravel()
    frame_buf.flush()

    fa, ha = int(frame_buf.device_address), int(hits_buf.device_address)
    ip.write(REG_FRAME_LO, fa & 0xFFFFFFFF); ip.write(REG_FRAME_HI, fa >> 32)
    ip.write(REG_HITS_LO,  ha & 0xFFFFFFFF); ip.write(REG_HITS_HI,  ha >> 32)
    ip.write(REG_WIDTH,  w)
    ip.write(REG_HEIGHT, h)

    ip.write(REG_CTRL, 0x1)                      # ap_start
    t0 = time.time()
    while True:
        if ip.read(REG_CTRL) & 0x2:              # ap_done (clears when read)
            break
        if time.time() - t0 > timeout_s:
            raise TimeoutError("accelerator did not finish")

    n = ip.read(REG_RET)
    hits_buf.invalidate()
    k = min(n, MAX_HITS)
    v = np.array(hits_buf[:k], dtype=np.uint32)
    return (v & 0xFFFF).astype(int), (v >> 16).astype(int), n


def group_rects(boxes, min_neighbors=5, eps=0.2):
    """Merge overlapping detections; keep clusters with more than min_neighbors boxes."""
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
        if len(members) > min_neighbors:
            m = np.mean(np.array(members, dtype=np.float64), axis=0)
            out.append(tuple(int(round(x)) for x in m))
    return out


def detect_faces(gray, scale=1.2, min_neighbors=5, min_size=40, verbose=False):
    """gray: uint8 image <= 320x240. Returns boxes (x, y, w, h) in that image's pixels."""
    boxes, s, t_hw = [], 1.0, 0.0
    while gray.shape[0] / s >= WIN and gray.shape[1] / s >= WIN:
        nh, nw = int(round(gray.shape[0] / s)), int(round(gray.shape[1] / s))
        lvl = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
        t0 = time.time()
        xs, ys, n = run_kernel(lvl)
        t_hw += time.time() - t0
        if n > MAX_HITS:
            print(f"warning: {n} hits on a {nw}x{nh} level, only {MAX_HITS} kept")
        for x, y in zip(xs, ys):
            boxes.append([int(x * s), int(y * s), int(WIN * s), int(WIN * s)])
        s *= scale
    if verbose:
        print(f"accelerator time for all levels: {t_hw * 1000:.1f} ms")
    return [b for b in group_rects(boxes, min_neighbors) if b[2] >= min_size]

# ---- 4. One webcam frame -> detect -> show ----------------------------------
# Find your camera first in a terminal:  ls /dev/video*
from IPython.display import display, Image, clear_output


def grab_gray(cap):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("could not read from the camera (check CAM_INDEX)")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (MAX_W, MAX_H), interpolation=cv2.INTER_AREA)
    return frame, np.ascontiguousarray(small)

def draw(frame, boxes):
    sx, sy = frame.shape[1] / MAX_W, frame.shape[0] / MAX_H
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (int(x * sx), int(y * sy)),
                      (int((x + w) * sx), int((y + h) * sy)), (0, 255, 0), 2)
    return frame

def show(frame):
    ok, jpg = cv2.imencode(".jpg", frame)
    display(Image(data=jpg.tobytes()))



# ---- run ---------------------------------------------------------------------
if RUN_SELFTEST:
    # compare the hardware against the golden-model test data (every pyramid level)
    import os
    DATA_DIR = "tb_data"

    bad = 0
    with open(os.path.join(DATA_DIR, "manifest.txt")) as f:
        levels = [line.split() for line in f if line.strip()]

    for k, (img_name, exp_name) in enumerate(levels):
        img = cv2.imread(os.path.join(DATA_DIR, img_name), cv2.IMREAD_GRAYSCALE)
        expected = set()
        with open(os.path.join(DATA_DIR, exp_name)) as f:
            for line in f:
                x, y = line.split()
                expected.add((int(x), int(y)))

        t0 = time.time()
        xs, ys, n = run_kernel(img)
        dt = (time.time() - t0) * 1000
        got = set(zip(xs.tolist(), ys.tolist()))

        missing, extra = len(expected - got), len(got - expected)
        ok = (missing + extra) <= 2          # same tolerance as the C testbench
        bad += (not ok)
        print(f"level {k:02d} {img.shape[1]:3d}x{img.shape[0]:3d}  expected={len(expected):4d} "
              f"hw={len(got):4d} missing={missing} extra={extra}  {dt:6.1f} ms  {'ok' if ok else 'MISMATCH'}")

    print("PASS: hardware matches the golden model" if bad == 0 else f"FAIL: {bad} level(s) differ")

if WEBCAM_MODE == "still":
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(5):                      # let auto-exposure settle
        cap.read()
    try:
        frame, small = grab_gray(cap)
        boxes = detect_faces(small, verbose=True)
        print("faces:", boxes)
        show(draw(frame, boxes))
    finally:
        cap.release()

elif WEBCAM_MODE == "live":
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        t_start, n_done = time.time(), 0
        while n_done < MAX_FRAMES:
            frame, small = grab_gray(cap)
            boxes = detect_faces(small)
            n_done += 1
            fps = n_done / (time.time() - t_start)
            clear_output(wait=True)
            show(draw(frame, boxes))
            print(f"frame {n_done}   faces: {len(boxes)}   {fps:.1f} fps")
    except KeyboardInterrupt:
        print("stopped")
    finally:
        cap.release()
