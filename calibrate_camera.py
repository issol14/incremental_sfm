import argparse
import glob
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import cv2

@dataclass
class Intrinsics:
    K: np.ndarray  
    dist: np.ndarray | None = None  

@dataclass
class Extrinsics:
    R: np.ndarray  
    t: np.ndarray  

def detect_corners(image_paths: List[str], pattern_size: Tuple[int, int]) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    cols, rows = pattern_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    image_points = []
    image_shape = None  

    for p in sorted(image_paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Failed to read {p}, skipping.")
            continue
        if image_shape is None:
            image_shape = img.shape[:2]
        if img.shape[:2] != image_shape:
            print(f"[WARN] Image {p} has different shape {img.shape[:2]} from {image_shape}, skipping.")
            continue

        # Prefer the newer SB detector if available
        ret = False
        try:
            ret, corners = cv2.findChessboardCornersSB(img, (cols, rows))
        except Exception:
            ret, corners = cv2.findChessboardCorners(img, (cols, rows))

        if not ret:
            print(f"[WARN] No corners in {p}, skipping.")
            continue

        # Subpixel refinement
        corners = corners.astype(np.float32)
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), criteria)
        image_points.append(corners.reshape(-1, 2))

    if not image_points:
        raise RuntimeError("No valid checkerboard detections.")

    return image_points, image_shape  # (h, w)

def make_object_points(pattern_size: Tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), dtype=np.float64)
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    objp[:, 0] = xs.flatten() * square_size
    objp[:, 1] = ys.flatten() * square_size
    return objp  

def project_points(K: np.ndarray, R: np.ndarray, t: np.ndarray, Pw: np.ndarray) -> np.ndarray:
    Pw_h = np.hstack([Pw, np.ones((Pw.shape[0], 1))])  
    Rt = np.hstack([R, t])  
    cam = Rt @ Pw_h.T  
    x = (K @ cam).T  
    u = x[:, 0] / x[:, 2]
    v = x[:, 1] / x[:, 2]
    return np.stack([u, v], axis=1)

def mean_reprojection_error(K: np.ndarray, extrs: List[Extrinsics], objp: np.ndarray, img_points: List[np.ndarray]) -> float:
    errs = []
    for (R, t), imgp in zip([(e.R, e.t) for e in extrs], img_points):
        proj = project_points(K, R, t, objp)
        errs.append(np.linalg.norm(proj - imgp, axis=1).mean())
    return float(np.mean(errs))

def opencv_calibrate(objp: np.ndarray, img_points: List[np.ndarray], image_shape: Tuple[int, int]):
    obj_points = [objp.astype(np.float32)] * len(img_points)
    img_points32 = [ip.astype(np.float32) for ip in img_points]
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=obj_points,
        imagePoints=img_points32,
        imageSize=(image_shape[1], image_shape[0]),
        cameraMatrix=None,
        distCoeffs=None,
        flags=0,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8),
    )
    extrs = []
    for r, t in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(r)
        extrs.append(Extrinsics(R=R, t=t))
    return ret, Intrinsics(K=K, dist=dist), extrs


def main(image_paths: list[str], refine: bool = False, pattern: tuple[int, int] = (13, 9), square_size: float = 0.02):
    img_points, image_shape = detect_corners(image_paths, pattern) 
    print(f"Collected views : {len(img_points)}")
    print(f"Image size : {image_shape[1]}x{image_shape[0]}") 

    objp = make_object_points(pattern, square_size)

    ret, intr_cv, extrs_cv = opencv_calibrate(objp, img_points, image_shape)
    print("\n[OpenCV] K:\n", intr_cv.K)
    print("[OpenCV] Distortion (k1,k2,p1,p2,k3):", intr_cv.dist.ravel())
    print(f"[OpenCV] RMS reported by OpenCV: {ret:.6f}")

    err_cv = mean_reprojection_error(intr_cv.K, extrs_cv, objp, img_points)
    print(f"[OpenCV] Mean reprojection error (ours, undistorted model): {err_cv:.4f} px")

if __name__ == "__main__":
    data_path = "/home/mori/lab/issol-ku/introduction_to_cv/StructurefromMotion/data/input/photo/checkerboard"
    paths = [os.path.join(data_path, p) for p in os.listdir(data_path)]
    main(paths, refine=True)