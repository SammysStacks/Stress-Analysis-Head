# fast_pdf_to_video_preserve_res_fallback.py
import os, gc, re, cv2, fitz, numpy as np
from math import ceil
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from natsort import natsorted

# ───── user settings ──────────────────────────────────────────────────
DPI          = 300
FPS          = 15
CLIP_SEC     = 16
RUN_PREFIX   = "2025-04-03"
DATASETS     = {"wesad","amigos","case","empatch","dapper","emognition"}
ROOT_DIR     = Path(__file__).resolve().parent / "modelInstances"
PREFERRED_CODECS = [             # (fourcc, extension)
    ("MJPG", ".avi"),  # always works, any resolution
    ("FFV1", ".mkv"),  # loss‑less, if ffv1 is compiled in
    ("avc1", ".mp4"),            # H.264/AVC
    ("H264", ".mp4"),            # some FFmpeg builds use this tag
    ("HEVC", ".mp4"),            # H.265/HEVC (if compiled in)
    ("FFV1", ".mkv"),            # lossless, unlimited size
]
GC_EVERY     = 25
PDF_RE       = re.compile(r"^(.*?)[ _-]?epochs(\d+)\.pdf$", re.I)
# ──────────────────────────────────────────────────────────────────────


# ---------- helpers ----------
def even(v: int) -> int:
    """Return the next even integer (FFmpeg requires even dims)."""
    return v + (v & 1)

def page_px_size(pdf: Path, dpi=DPI):
    with fitz.open(pdf) as d:
        r = d[0].rect
        scale = dpi / 72.0
        return int(r.width*scale+0.5), int(r.height*scale+0.5)

def render_page(pdf: Path, frame_size, dpi=DPI):
    with fitz.open(pdf) as d:
        pix = d[0].get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    if (pix.width, pix.height) != frame_size:
        canvas = np.zeros((frame_size[1], frame_size[0], 3), np.uint8)
        x0 = (frame_size[0] - pix.width)//2
        y0 = (frame_size[1] - pix.height)//2
        canvas[y0:y0+pix.height, x0:x0+pix.width] = img
        img = canvas
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def open_writer(path_base: Path, frame_size):
    for fourcc_str, ext in PREFERRED_CODECS:
        out_path = path_base.with_suffix(ext)
        fourcc   = cv2.VideoWriter_fourcc(*fourcc_str)
        vw = cv2.VideoWriter(str(out_path), fourcc, FPS, frame_size)
        if vw.isOpened():
            return vw, out_path
        vw.release()                # <‑‑ add this
    raise RuntimeError("No available codec …")


# ---------- worker ----------
def build_video(job):
    pdf_folder, prefix, pdf_list = job
    # 1. determine constant frame size (max w/h, then even)
    max_w = max_h = 0
    for pdf in pdf_list:
        w, h = page_px_size(pdf)
        max_w, max_h = max(max_w, w), max(max_h, h)
    frame_size = (even(max_w), even(max_h))

    # 2. open a writer with the first available large‑frame codec
    out_base = pdf_folder / prefix
    vw, out_path = open_writer(out_base, frame_size)

    dup = max(1, ceil((FPS*CLIP_SEC)/len(pdf_list)))
    for i, pdf in enumerate(pdf_list, 1):
        frame = render_page(pdf, frame_size)
        for _ in range(dup):
            vw.write(frame)
        if i % GC_EVERY == 0:
            gc.collect()
    vw.release()
    return prefix, len(pdf_list), out_path.name

# ---------- job gathering ----------
def gather_jobs():
    for model_dir in ROOT_DIR.iterdir():
        if not (model_dir.is_dir() and model_dir.name.startswith(RUN_PREFIX)):
            continue
        for ds_dir in model_dir.iterdir():
            if ds_dir.name.lower() not in DATASETS or not ds_dir.is_dir():
                continue
            for sub in ("signalEncoding", "signalReconstruction"):
                folder = ds_dir / sub
                if not folder.is_dir(): continue
                groups = defaultdict(list)
                for pdf in folder.glob("*.pdf"):
                    m = PDF_RE.match(pdf.name)
                    if m:
                        prefix, epoch = m.groups()
                        groups[prefix].append((int(epoch), pdf))
                for p, lst in groups.items():
                    lst = natsorted(lst, key=lambda t: t[0])
                    yield (folder, p, [pdf for _, pdf in lst])

# ---------- main ----------
def main():
    jobs = list(gather_jobs())
    if not jobs:
        print("Nothing to convert."); return

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    print(f"Processing {len(jobs)} video groups with "
          f"{os.cpu_count()} CPU cores …\n")

    with ProcessPoolExecutor() as pool:
        futs = {pool.submit(build_video, j): j[1] for j in jobs}
        for f in as_completed(futs):
            prefix, n, fname = f.result()
            print(f"{prefix:<40} {n:>4} pages  → {fname}")

if __name__ == "__main__":
    main()
