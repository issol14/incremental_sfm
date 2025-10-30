from pathlib import Path
import argparse
import shutil
import json
import sys
import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

import pycolmap

def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z + y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ], dtype=float)

def _normalize_camera_model(cam) -> str:
    m = getattr(cam, "model", None)
    model_name = getattr(cam, "model_name", None)
    if isinstance(model_name, str) and model_name:
        s = model_name
    else:
        if hasattr(m, "name"):
            s = m.name
        else:
            s = str(m) if m is not None else ""
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
        f, cx, cy = p[:3]
        fx = fy = f
    elif model in {"OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE", "FOV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = p[:4]
    else:
        raise NotImplementedError(f"Unsupported camera model: {model}")
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], float)

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
                if M.size == 16:
                    M = M.reshape(4, 4)
                elif M.size == 12:
                    M = M.reshape(3, 4)
            if M.shape == (3, 4):
                M = np.vstack([M, np.array([[0.0, 0.0, 0.0, 1.0]])])
            if M.shape == (3, 3):
                t_cand = None
                for t_name in ("translation", "t", "tvec"):
                    if hasattr(Tobj, t_name):
                        t_cand = np.asarray(getattr(Tobj, t_name), dtype=float).reshape(3, 1)
                        break
                if t_cand is None:
                    return None
                M4 = np.eye(4)
                M4[:3, :3] = M
                M4[:3, 3] = t_cand[:, 0]
                M = M4
            if M.shape != (4, 4):
                return None
            return np.linalg.inv(M) if is_Tcw else M

        R = None
        if hasattr(Tobj, "rotation"):
            rot = Tobj.rotation
            rot_m = getattr(rot, "matrix", None)
            if callable(rot_m):
                R = np.asarray(rot.matrix(), dtype=float)
            else:
                R = np.asarray(rot, dtype=float)
        if R is None:
            if hasattr(Tobj, "R"):
                R = np.asarray(getattr(Tobj, "R"), dtype=float)
            elif hasattr(Tobj, "qvec"):
                q = np.asarray(getattr(Tobj, "qvec"), dtype=float)
                R = qvec2rotmat(q)
            elif hasattr(Tobj, "q"):
                q = np.asarray(getattr(Tobj, "q"), dtype=float)
                R = qvec2rotmat(q)
        t = None
        if hasattr(Tobj, "translation"):
            t = np.asarray(Tobj.translation, dtype=float).reshape(3, 1)
        elif hasattr(Tobj, "t"):
            t = np.asarray(getattr(Tobj, "t"), dtype=float).reshape(3, 1)
        elif hasattr(Tobj, "tvec"):
            t = np.asarray(getattr(Tobj, "tvec"), dtype=float).reshape(3, 1)
        if R is None or t is None:
            return None
        Twc = np.eye(4)
        if is_Tcw:
            Twc[:3, :3] = R.T
            Twc[:3, 3] = (-R.T @ t)[:, 0]
        else:
            Twc[:3, :3] = R
            Twc[:3, 3] = t[:, 0]
        return Twc

    for attr in ("cam_from_world", "camera_from_world", "world_to_cam", "Tcw", "T_cw"):
        if hasattr(img, attr):
            Twc = _rigid_to_Twc(getattr(img, attr), is_Tcw=True)
            if Twc is not None:
                return Twc
    for attr in ("world_from_cam", "cam_to_world", "camera_to_world", "Twc", "T_wc"):
        if hasattr(img, attr):
            Twc = _rigid_to_Twc(getattr(img, attr), is_Tcw=False)
            if Twc is not None:
                return Twc
    raise AttributeError("Unsupported pycolmap.Image pose interface")

def extract_and_match(database_path: Path,
                      image_dir: Path,
                      matching: str,
                      max_features: int | None,
                      camera_mode_str: str | None,
                      single_camera_flag: bool,
                      camera_model: str | None,
                      vocab_tree_path: Path | None):
    sift_options = pycolmap.SiftExtractionOptions()
    if max_features and max_features > 0:
        sift_options.max_num_features = int(max_features)

    reader_options = pycolmap.ImageReaderOptions()

    cam_mode = pycolmap.CameraMode.AUTO
    if camera_mode_str:
        cm = camera_mode_str.upper()
        if cm not in {"AUTO", "SINGLE", "PER_FOLDER", "PER_IMAGE"}:
            raise ValueError(f"Unsupported --camera_mode: {camera_mode_str}")
        cam_mode = getattr(pycolmap.CameraMode, cm)
    elif single_camera_flag:
        cam_mode = pycolmap.CameraMode.SINGLE

    if camera_model is None:
        camera_model = "SIMPLE_RADIAL"

    print("[1/5] Extracting features …")
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(image_dir),
        camera_mode=cam_mode,
        camera_model=str(camera_model),
        reader_options=reader_options,
        sift_options=sift_options,
    )

    print(f"[2/5] Matching ({matching}) …")
    matching = matching.lower()
    if matching == "exhaustive":
        pycolmap.match_exhaustive(str(database_path))
    elif matching == "sequential":
        pycolmap.match_sequential(str(database_path))
    elif matching == "spatial":
        pycolmap.match_spatial(str(database_path))
    elif matching == "vocabtree":
        if not vocab_tree_path:
            raise ValueError("--matching vocabtree requires --vocab_tree path")
        pycolmap.match_vocabtree(str(database_path), vocab_tree_path=str(vocab_tree_path))
    else:
        raise ValueError("matching should be one of exhaustive|sequential|spatial|vocabtree")

def run_incremental_mapping(database_path: Path, image_dir: Path, out_dir: Path):
    print("[3/5] Incremental mapping …")
    maps = pycolmap.incremental_mapping(str(database_path), str(image_dir), str(out_dir))
    if not maps:
        raise RuntimeError("Reconstruction failed: no map was generated")
    maps[0].write(str(out_dir))
    return find_sparse_model_dir(out_dir)

def run_dense(mvs_dir: Path, sparse_model_dir: Path, image_dir: Path):
    print("[4/5] Undistort images for MVS …")
    pycolmap.undistort_images(str(mvs_dir), str(sparse_model_dir), str(image_dir))
    print("[5/5] PatchMatch Stereo + Stereo Fusion …")
    pycolmap.patch_match_stereo(str(mvs_dir))
    pycolmap.stereo_fusion(str(mvs_dir / "dense.ply"), str(mvs_dir))

def collect_reconstruction(model_dir: Path):
    recon = pycolmap.Reconstruction(str(model_dir))

    pts, cols = [], []
    for pt in recon.points3D.values():
        pts.append(np.asarray(pt.xyz, float))
        cols.append(np.asarray(pt.color, float) / 255.0)
    P = np.vstack(pts) if pts else np.empty((0, 3), float)
    C = np.vstack(cols) if cols else np.empty((0, 3), float)

    image_ids, image_names, camera_ids, Twc_list = [], [], [], []
    for img in recon.images.values():
        Twc = compute_Twc_from_img(img)
        Twc_list.append(Twc)
        image_ids.append(int(getattr(img, "image_id", getattr(img, "id", -1))))
        image_names.append(str(img.name))
        camera_ids.append(int(img.camera_id))
    Twc_arr = np.stack(Twc_list, axis=0) if Twc_list else np.empty((0, 4, 4), float)

    cam_dict = {}
    for cam_id, cam in recon.cameras.items():
        K = K_from_camera(cam)
        cam_dict[int(cam_id)] = {
            "model": _normalize_camera_model(cam),
            "width": int(cam.width),
            "height": int(cam.height),
            "params": [float(x) for x in np.array(cam.params, float).tolist()],
            "K": K.tolist(),
        }

    return {
        "points_xyz": P,
        "points_rgb": C,
        "Twc": Twc_arr,
        "image_ids": np.array(image_ids, int),
        "image_names": image_names,
        "camera_ids": np.array(camera_ids, int),
        "cameras": cam_dict,
    }

def save_artifacts_npz_json(artifacts: dict, out_dir: Path, npz_path: Path | None, json_path: Path | None):
    if npz_path is None:
        npz_path = out_dir / "artifacts.npz"
    np.savez_compressed(
        npz_path,
        points_xyz=artifacts["points_xyz"],
        points_rgb=artifacts["points_rgb"],
        Twc=artifacts["Twc"],
        image_ids=artifacts["image_ids"],
        image_names=np.array(artifacts["image_names"], dtype=object),
        camera_ids=artifacts["camera_ids"],
    )
    print(f"[+] Saved artifacts (npz): {npz_path}")

    if json_path is None:
        json_path = out_dir / "artifacts.json"
    json_obj = {
        "image_names": artifacts["image_names"],
        "image_ids": artifacts["image_ids"].tolist(),
        "camera_ids": artifacts["camera_ids"].tolist(),
        "cameras": artifacts["cameras"],
        "num_points": int(artifacts["points_xyz"].shape[0]),
        "num_images": int(artifacts["Twc"].shape[0]),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved artifacts (json): {json_path}")

def visualize_and_save(model_dir: Path,
                       out_png: Path,
                       show_dense: bool = False,
                       width: int = 1600,
                       height: int = 1200,
                       point_size: float = 1.5,
                       bg_color=(1, 1, 1)):
    recon = pycolmap.Reconstruction(str(model_dir))

    pts, colors = [], []
    for pt in recon.points3D.values():
        pts.append(np.asarray(pt.xyz, float))
        colors.append(np.asarray(pt.color, float) / 255.0)
    P = np.vstack(pts) if pts else np.empty((0, 3), float)
    C = np.vstack(colors) if colors else np.empty((0, 3), float)

    images = list(recon.images.values())
    Twc_list = [compute_Twc_from_img(img) for img in images]

    if o3d is not None:
        try:
            pcd = o3d.geometry.PointCloud()
            if P.size:
                pcd.points = o3d.utility.Vector3dVector(P)
                pcd.colors = o3d.utility.Vector3dVector(C)
            geoms = [pcd]

            for Twc in Twc_list:
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                axis.transform(Twc)
                geoms.append(axis)

            if show_dense:
                dense_path = (model_dir.parent / "mvs" / "dense.ply").resolve()
                if dense_path.exists():
                    dense = o3d.io.read_point_cloud(str(dense_path))
                    geoms.append(dense)

            vis = o3d.visualization.Visualizer()
            if not vis.create_window(window_name='Open3D', width=int(width), height=int(height), visible=False):
                raise RuntimeError("Open3D create_window failed (headless)")
            for g in geoms:
                vis.add_geometry(g)
            ro = vis.get_render_option()
            if ro is None:
                raise RuntimeError("Open3D RenderOption is None (headless)")
            ro.point_size = float(point_size)
            ro.background_color = np.asarray(bg_color, float)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(str(out_png), do_render=True)
            vis.destroy_window()
            print(f"[+] Saved visualization to: {out_png}")
            return
        except Exception as e:
            print(f"[!] Open3D headless render failed: {e}. Falling back to Matplotlib…")

    if plt is None:
        raise RuntimeError("Headless rendering failed and matplotlib is unavailable")

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    if P.size:
        s = max(0.1, point_size * 0.3)
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=s, c=C if C.size else 'k', depthshade=False)

    axis_len = 0.1
    for Twc in Twc_list:
        Cc = Twc[:3, 3]
        R = Twc[:3, :3]
        axes = R * axis_len
        ax.plot([Cc[0], Cc[0] + axes[0, 0]], [Cc[1], Cc[1] + axes[1, 0]], [Cc[2], Cc[2] + axes[2, 0]], color='r', linewidth=1)
        ax.plot([Cc[0], Cc[0] + axes[0, 1]], [Cc[1], Cc[1] + axes[1, 1]], [Cc[2], Cc[2] + axes[2, 1]], color='g', linewidth=1)
        ax.plot([Cc[0], Cc[0] + axes[0, 2]], [Cc[1], Cc[1] + axes[1, 2]], [Cc[2], Cc[2] + axes[2, 2]], color='b', linewidth=1)

    try:
        xyz = P if P.size else np.array([[0, 0, 0]])
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)
        centers = (mins + maxs) / 2.0
        span = float(np.max(maxs - mins) or 1.0)
        ax.set_xlim(centers[0] - span / 2, centers[0] + span / 2)
        ax.set_ylim(centers[1] - span / 2, centers[1] + span / 2)
        ax.set_zlim(centers[2] - span / 2, centers[2] + span / 2)
    except Exception:
        pass

    plt.tight_layout()
    fig.savefig(str(out_png), dpi=100)
    plt.close(fig)
    print(f"[+] Saved (matplotlib fallback) to: {out_png}")

def find_sparse_model_dir(out_dir: Path) -> Path:
    if (out_dir / "cameras.bin").exists() and (out_dir / "images.bin").exists():
        return out_dir
    cand = out_dir / "0"
    if (cand / "cameras.bin").exists() and (cand / "images.bin").exists():
        return cand
    for p in sorted(out_dir.glob("*")):
        if p.is_dir() and (p / "cameras.bin").exists() and (p / "images.bin").exists():
            return p
    raise FileNotFoundError("Sparse model folder (cameras.bin/images.bin) not found.")

def ensure_images(image_dir: Path):
    if not image_dir.exists():
        raise FileNotFoundError(f"--images path does not exist: {image_dir}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [p for p in image_dir.iterdir() if p.suffix.lower() in exts]
    if len(files) < 2:
        raise RuntimeError(f"At least 2 image files are required. ({image_dir})")
    return files

def run_pipeline(
    image_dir: Path,
    out_dir: Path,
    matching: str = "exhaustive",
    max_features: int = 4096,
    single_camera: bool = False,
    camera_mode: str | None = None,
    camera_model: str | None = None,
    use_dense: bool = False,
    vocab_tree_path: Path | None = None,
    viz_png: Path | None = None,
    save_npz: Path | None = None,
    save_json: Path | None = None,
    overwrite: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    database_path = out_dir / "database.db"
    mvs_dir = out_dir / "mvs"
    if overwrite:
        if database_path.exists():
            database_path.unlink()
        if mvs_dir.exists():
            shutil.rmtree(mvs_dir)

    ensure_images(image_dir)

    extract_and_match(
        database_path=database_path,
        image_dir=image_dir,
        matching=matching,
        max_features=max_features,
        camera_mode_str=camera_mode,
        single_camera_flag=single_camera,
        camera_model=camera_model,
        vocab_tree_path=vocab_tree_path,
    )

    sparse_model_dir = run_incremental_mapping(
        database_path=database_path, image_dir=image_dir, out_dir=out_dir
    )

    if use_dense:
        try:
            run_dense(mvs_dir=mvs_dir, sparse_model_dir=sparse_model_dir, image_dir=image_dir)
        except Exception as e:
            print(f"[!] Error during dense reconstruction (proceeding anyway): {e}")

    artifacts = collect_reconstruction(sparse_model_dir)
    save_artifacts_npz_json(artifacts, out_dir, save_npz, save_json)

    if viz_png is None:
        viz_png = out_dir / "visualize.png"
    visualize_and_save(
        model_dir=sparse_model_dir,
        out_png=viz_png,
        show_dense=use_dense,
        width=1600,
        height=1000,
        point_size=1.5,
        bg_color=(1, 1, 1),
    )

    recon = pycolmap.Reconstruction(str(sparse_model_dir))
    if recon.images:
        first_img = list(recon.images.values())[0]
        cam = recon.cameras[first_img.camera_id]
        K = K_from_camera(cam)
        print("Camera model:", _normalize_camera_model(cam))
        print("Size (w,h):", cam.width, cam.height)
        print("Params:", np.array(cam.params))
        print("K:\n", K)

    print("\nDone.")
    print(f"- Sparse model dir: {sparse_model_dir}")
    if use_dense:
        print(f"- Dense point cloud: {mvs_dir / 'dense.ply'}")
    print(f"- Visualization: {viz_png}")

def main():
    ap = argparse.ArgumentParser(description="SfM pipeline with PyCOLMAP + Visualization + Artifacts (fixed)")
    ap.add_argument("--images", type=Path, required=True, help="Input image directory (contains extracted frames)")
    ap.add_argument("--out", type=Path, required=True, help="Output folder")
    ap.add_argument("--matching", type=str, default="exhaustive",
                    choices=["exhaustive", "sequential", "spatial", "vocabtree"], help="Feature matching method")
    ap.add_argument("--vocab_tree", type=Path, default=None, help="Path to vocab tree file for vocabtree matching")
    ap.add_argument("--max_features", type=int, default=4096, help="SIFT max features")
    ap.add_argument("--single_camera", action="store_true", help="Treat all images as single camera (identical intrinsics)")
    ap.add_argument("--camera_mode", type=str, default=None,
                    choices=["AUTO", "SINGLE", "PER_FOLDER", "PER_IMAGE"],
                    help="Camera mode (overrides --single_camera if set)")
    ap.add_argument("--camera_model", type=str, default=None,
                    help="Force camera model (e.g. SIMPLE_RADIAL, PINHOLE etc.)")
    ap.add_argument("--dense", action="store_true", help="Run dense (PatchMatch + Fusion)")
    ap.add_argument("--viz_png", type=Path, default=None, help="Visualization output path (default: out/visualize.png)")
    ap.add_argument("--save_npz", type=Path, default=None, help="Artifacts (npz) output path (default: out/artifacts.npz)")
    ap.add_argument("--save_json", type=Path, default=None, help="Artifacts (json) output path (default: out/artifacts.json)")
    ap.add_argument("--overwrite", action="store_true", help="Remove existing DB/mvs folder before execution")
    args = ap.parse_args()

    run_pipeline(
        image_dir=args.images,
        out_dir=args.out,
        matching=args.matching,
        max_features=args.max_features,
        single_camera=args.single_camera,
        camera_mode=args.camera_mode,
        camera_model=args.camera_model,
        use_dense=args.dense,
        vocab_tree_path=args.vocab_tree,
        viz_png=args.viz_png,
        save_npz=args.save_npz,
        save_json=args.save_json,
        overwrite=args.overwrite,
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[!] ERROR: {e}")
        sys.exit(1)
