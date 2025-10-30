import cv2
from pathlib import Path

def extract_sfm_frames(
    video_path: str,
    out_dir: str,
    step: int = 5,
    long_side: int = 1600,
    fmt: str = "png",
    jpg_quality: int = 95,
    prefix: str = "frame",
) -> int:
    """
    Extracts frames from a video optimized for SfM and saves them.

    Args:
        video_path: Path to the video file (.mp4, .mov, etc.)
        out_dir: Directory to save the extracted frames
        step: Save every N-th frame (default=5)
        long_side: Resize the longer side to this length in pixels (default=1600, <=0: keep original size)
        fmt: "png" (recommended) or "jpg"
        jpg_quality: JPEG quality (1~100, default=95)
        prefix: Filename prefix

    Returns:
        Number of images saved (int)
    """
    step = max(1, int(step))
    fmt = fmt.lower()
    assert fmt in ("png", "jpg", "jpeg"), "fmt must be 'png' or 'jpg' only."

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    def resize_long_side(img, target):
        # Resize the image so that the longer side matches 'target' pixels
        if target is None or target <= 0:
            return img
        h, w = img.shape[:2]
        m = max(h, w)
        if m <= target:
            return img
        scale = target / float(m)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if fmt in ("jpg", "jpeg"):
        imwrite_params = [cv2.IMWRITE_JPEG_QUALITY, int(max(1, min(100, jpg_quality)))]
        ext = ".jpg"
    else:  
        imwrite_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        ext = ".png"

    saved = 0
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Save every N-th frame according to the step interval
            if idx % step == 0:
                frame = resize_long_side(frame, long_side)
                filename = out / f"{prefix}_{saved:06d}{ext}"
                ok = cv2.imwrite(str(filename), frame, imwrite_params)
                if ok:
                    saved += 1
            idx += 1
    finally:
        cap.release()

    return saved

if __name__ == "__main__":
    n = extract_sfm_frames("./data/input/video/checkerboard.MOV", "./data/input/photo/checkerboard", step=7, long_side=-2, fmt="png")
    print(f"Number of images saved: {n}")
