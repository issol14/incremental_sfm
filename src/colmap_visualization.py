
from pathlib import Path
import argparse
import sys
import numpy as np
import pycolmap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [  2*(x*y + z*w),   1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [  2*(x*z - y*w),     2*(y*z + x*w),   1 - 2*(x*x + y*y)]
    ], dtype=float)

def compute_Twc_from_img(img):
    if hasattr(img, "qvec") and hasattr(img, "tvec"):
        q = np.asarray(img.qvec, dtype=float)
        Rcw = qvec2rotmat(q)
        tcw = np.asarray(img.tvec, dtype=float).reshape(3, 1)
        Twc = np.eye(4)
        Twc[:3, :3] = Rcw.T
        Twc[:3, 3] = (-Rcw.T @ tcw)[:, 0]
        return Twc

    def _rigid_to_Twc(Tobj, is_Tcw: bool) -> np.ndarray | None:
        if Tobj is None:
            return None
        if callable(Tobj):
            Tobj = Tobj()
        mat_attr = getattr(Tobj, "matrix", None)
        if callable(mat_attr):
            M = np.asarray(Tobj.matrix(), dtype=float).squeeze()
            if M.ndim == 1:
                if M.size == 16: M = M.reshape(4, 4)
                elif M.size == 12: M = M.reshape(3, 4)
            if M.shape == (3, 4):
                M = np.vstack([M, np.array([[0.0, 0.0, 0.0, 1.0]])])
            if M.shape == (3, 3):
                t_cand = None
                for t_name in ("translation", "t", "tvec"):
                    if hasattr(Tobj, t_name):
                        t_cand = np.asarray(getattr(Tobj, t_name), dtype=float).reshape(3, 1)
                        break
                if t_cand is None: return None
                M4 = np.eye(4); M4[:3, :3] = M; M4[:3, 3] = t_cand[:, 0]; M = M4
            if M.shape != (4, 4): return None
            return np.linalg.inv(M) if is_Tcw else M

        R = None
        if hasattr(Tobj, "rotation"):
            rot = Tobj.rotation
            rot_m = getattr(rot, "matrix", None)
            R = np.asarray(rot.matrix(), float) if callable(rot_m) else np.asarray(rot, float)
        if R is None:
            if hasattr(Tobj, "R"): R = np.asarray(getattr(Tobj, "R"), float)
            elif hasattr(Tobj, "qvec"): R = qvec2rotmat(np.asarray(getattr(Tobj, "qvec"), float))
            elif hasattr(Tobj, "q"): R = qvec2rotmat(np.asarray(getattr(Tobj, "q"), float))
        t = None
        if hasattr(Tobj, "translation"): t = np.asarray(Tobj.translation, float).reshape(3, 1)
        elif hasattr(Tobj, "t"):         t = np.asarray(getattr(Tobj, "t"), float).reshape(3, 1)
        elif hasattr(Tobj, "tvec"):      t = np.asarray(getattr(Tobj, "tvec"), float).reshape(3, 1)
        if R is None or t is None: return None
        Twc = np.eye(4)
        if is_Tcw: Twc[:3, :3] = R.T; Twc[:3, 3] = (-R.T @ t)[:, 0]
        else:      Twc[:3, :3] = R;   Twc[:3, 3] = t[:, 0]
        return Twc

    for attr in ("cam_from_world", "camera_from_world", "world_to_cam", "Tcw", "T_cw"):
        if hasattr(img, attr):
            Twc = _rigid_to_Twc(getattr(img, attr), is_Tcw=True)
            if Twc is not None: return Twc
    for attr in ("world_from_cam", "cam_to_world", "camera_to_world", "Twc", "T_wc"):
        if hasattr(img, attr):
            Twc = _rigid_to_Twc(getattr(img, attr), is_Tcw=False)
            if Twc is not None: return Twc
    raise AttributeError("Unsupported pycolmap.Image pose interface")

def visualize_and_save(model_dir: Path,
                       out_png: Path,
                       width: int = 1600,
                       height: int = 1200,
                       point_size: float = 1.5,
                       bg_color=(1, 1, 1),
                       axis_len: float = 0.1,
                       viz_scale: float = 1.0,
                       zoom: float = 1.0,
                       elev: float = 30,
                       azim: float = 0):
    recon = pycolmap.Reconstruction(str(model_dir))
    pts = []
    for pt in recon.points3D.values():
        pts.append(np.asarray(pt.xyz, float))
    P = np.vstack(pts) if pts else np.empty((0, 3), float)
    # C 제거 (색상 정보 사용하지 않음)
    images = list(recon.images.values())
    Twc_list = [compute_Twc_from_img(img) for img in images]
    if viz_scale != 1.0 and P.size:
        P = P * float(viz_scale)
    if viz_scale != 1.0 and Twc_list:
        for i in range(len(Twc_list)):
            Twc_list[i] = Twc_list[i].copy()
            Twc_list[i][:3, 3] *= float(viz_scale)
    fig = plt.figure(figsize=(width/100.0, height/100.0), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    if P.size:
        s = max(0.1, point_size*0.3)
        # 점 색상을 회색 단색으로 지정
        gray_color = np.array([[0.5, 0.5, 0.5]])
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=s, c=gray_color, depthshade=False)
    for Twc in Twc_list:
        Cc = Twc[:3, 3]
        R = Twc[:3, :3]
        axes = R * float(axis_len)
        ax.plot([Cc[0], Cc[0]+axes[0,0]],[Cc[1], Cc[1]+axes[1,0]],[Cc[2], Cc[2]+axes[2,0]], color='r', linewidth=5)
        ax.plot([Cc[0], Cc[0]+axes[0,1]],[Cc[1], Cc[1]+axes[1,1]],[Cc[2], Cc[2]+axes[2,1]], color='g', linewidth=5)
        ax.plot([Cc[0], Cc[0]+axes[0,2]],[Cc[1], Cc[1]+axes[1,2]],[Cc[2], Cc[2]+axes[2,2]], color='b', linewidth=5)
    try:
        xyz = P if P.size else np.array([[0,0,0]])
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)
        centers = (mins + maxs) / 2.0
        span = float(np.max(maxs - mins) or 1.0)
        span = span / max(float(zoom), 1e-6)
        ax.set_xlim(centers[0]-span/2, centers[0]+span/2)
        ax.set_ylim(centers[1]-span/2, centers[1]+span/2)
        ax.set_zlim(centers[2]-span/2, centers[2]+span/2)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
    plt.tight_layout()
    fig.savefig(str(out_png), dpi=100)
    plt.close(fig)
    print(f"[+] Saved (matplotlib): {out_png}")

def find_sparse_model_dir(out_dir: Path) -> Path:
    if (out_dir / "cameras.bin").exists() and (out_dir / "images.bin").exists():
        return out_dir
    cand = out_dir / "0"
    if (cand / "cameras.bin").exists() and (cand / "images.bin").exists():
        return cand
    for p in sorted(out_dir.glob("*")):
        if p.is_dir() and (p / "cameras.bin").exists() and (p / "images.bin").exists():
            return p
    raise FileNotFoundError("sparse 모델 폴더(cameras.bin/images.bin)가 보이지 않습니다.")

def _normalize_camera_model(cam) -> str:
    m = getattr(cam, "model", None)
    model_name = getattr(cam, "model_name", None)
    if isinstance(model_name, str) and model_name:
        s = model_name
    else:
        s = m.name if hasattr(m, "name") else (str(m) if m is not None else "")
    s = s.upper()
    if s.startswith("CAMERAMODELID."):
        s = s.split(".", 1)[1]
    return s

def K_from_camera(cam):
    model = _normalize_camera_model(cam)
    p = np.array(cam.params, dtype=float)
    if model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
    elif model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        f, cx, cy = p[:3]; fx = fy = f
    elif model in {"OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE", "FOV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = p[:4]
    else:
        raise NotImplementedError(f"Unsupported camera model: {model}")
    return np.array([[fx, 0,  cx],
                     [0,  fy, cy],
                     [0,  0,  1]], float)

def read_camera_intrinsics(model_dir: str | Path) -> dict[int, dict]:
    recon = pycolmap.Reconstruction(str(model_dir))
    out: dict[int, dict] = {}
    for cid, cam in recon.cameras.items():
        model = _normalize_camera_model(cam)
        p = np.asarray(cam.params, float)
        K = K_from_camera(cam)
        if model == "PINHOLE":
            used = 4
        elif model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            used = 3
        else:
            used = 4
        dist = p[used:].tolist()
        out[int(cid)] = {
            "model": model,
            "width": int(cam.width),
            "height": int(cam.height),
            "params": p.tolist(),
            "K": K,
            "dist": dist,
        }
    return out

def read_intrinsics_per_image(model_dir: str | Path) -> dict[str, dict]:
    recon = pycolmap.Reconstruction(str(model_dir))
    by_cam = read_camera_intrinsics(model_dir)
    out: dict[str, dict] = {}
    for img in recon.images.values():
        cid = int(img.camera_id)
        cam_info = by_cam[cid]
        out[str(img.name)] = {
            "camera_id": cid,
            "K": cam_info["K"],
            "dist": cam_info["dist"],
            "width": cam_info["width"],
            "height": cam_info["height"],
            "model": cam_info["model"],
        }
    return out

def main():
    ap = argparse.ArgumentParser(description="Visualize existing COLMAP output (Matplotlib only)")
    ap.add_argument("--out", type=Path, required=True, help="COLMAP 결과 출력 폴더 (out)")
    ap.add_argument("--viz_png", type=Path, default=None, help="시각화 저장 경로(기본: out/visualize.png)")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--point_size", type=float, default=1.5)
    ap.add_argument("--axis_len", type=float, default=0.1, help="카메라 좌표축 길이")
    ap.add_argument("--viz_scale", type=float, default=1.0, help="시각화용 좌표 스케일(데이터 파일은 그대로)")
    ap.add_argument("--zoom", type=float, default=1.0, help="화면 확대 배율(>1 확대, 1=기본)")
    ap.add_argument("--elev", type=float, default=30, help="카메라 고도(상하)")
    ap.add_argument("--azim", type=float, default=0, help="카메라 방향(좌우)")
    args = ap.parse_args()

    sparse_model_dir = find_sparse_model_dir(args.out)
    viz_png = args.viz_png or (sparse_model_dir / "visualize.png")

    visualize_and_save(
        model_dir=sparse_model_dir,
        out_png=viz_png,
        width=int(args.width),
        height=int(args.height),
        point_size=float(args.point_size),
        bg_color=(1, 1, 1),
        axis_len=float(args.axis_len),
        viz_scale=float(args.viz_scale),
        zoom=float(args.zoom),
    )
    sparse_dir = str(args.out)
    cams = read_camera_intrinsics(sparse_dir)
    print("Cameras:", {k: {kk: (vv if kk!='K' else np.round(vv,2).tolist())
                        for kk, vv in v.items() if kk in ('model','width','height','K','dist')}
                    for k, v in cams.items()})
    per_img = read_intrinsics_per_image(sparse_dir)
    first = next(iter(per_img.items()))
    print("First image intrinsics:", first[0], np.round(first[1]["K"], 2), first[1]["dist"])

    print("\nDone.")
    print(f"- Sparse model dir: {sparse_model_dir}")
    print(f"- Visualization: {viz_png}")

if __name__ == "__main__":
    try:
        main()
        
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[!] ERROR: {e}")
        sys.exit(1)
