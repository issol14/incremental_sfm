import cv2
import numpy as np
import glob
import math
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

K = np.array([[1641.5, 0.0,   960.0],
              [0.0,   1651.8, 540.0],
              [0.0,      0.0,   1.0]], dtype=np.float64)  # from calibrate_camera.py

DIST_COEFFS = None  
PIXEL_DISP_MIN = 5.0 
RATIO_TEST = 0.75
MIN_RAW_MATCHES_PER_PAIR = 50
REPROJ_THRESH_PX = 3.0
MIN_PNP_2D3D = 6
ESSENTIAL_RANSAC_THRESH = 1.5   
MIN_INITIAL_INLIERS     = 60   
MIN_PARALLAX_DEG        = 1.5
save_path = f"/home/mori/lab/issol-ku/introduction_to_cv/Structure_from_motion/data/my_output/visualize.png"

class ReconstructionMap:
    def __init__(self):
        self.points_3d = {}              
        self.poses = {}               
        self.point_observers = defaultdict(list) 
        self.kp_to_point_id = defaultdict(dict) 
        self.registered_images = set()
        self.next_point_id = 0
        self.image_id_to_path = {}          
        self.image_sizes = {}                 
        self.K = K
        self.dist_coeffs = DIST_COEFFS

def ensure_float32(arr):
    return np.asarray(arr, dtype=np.float32)

def rodrigues_to_R(rvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
    return R

def R_to_rodrigues(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3,)

def camera_matrix(R, t, K):
    return K @ np.hstack([R, t])

def project_points(X, R, t, K):
    Xh = (R @ X.T + t).T        # (N,3) in camera
    x = Xh[:, :2] / Xh[:, 2:3]  # normalize
    uv = (K[:2, :2] @ x.T + K[:2, 2:3]).T
    return uv

def compute_parallax_deg(R, pts1, pts2, K):
    if len(pts1) == 0:
        return 0.0
    Kinv = np.linalg.inv(K)
    v1 = (Kinv @ np.hstack([pts1, np.ones((len(pts1),1))]).T).T
    v1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2_cam2 = (Kinv @ np.hstack([pts2, np.ones((len(pts2),1))]).T).T
    v2_cam2 = v2_cam2 / np.linalg.norm(v2_cam2, axis=1, keepdims=True)
    v2_in_cam1 = (R.T @ v2_cam2.T).T
    dots = np.clip(np.sum(v1 * v2_in_cam1, axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    if len(ang) == 0:
        return 0.0
    return float(np.median(ang))

def cheirality_mask(R1, t1, R2, t2, Xs):
    C1 = (-R1.T @ t1).reshape(3,)
    C2 = (-R2.T @ t2).reshape(3,)
    z1 = R1[2, :]
    z2 = R2[2, :]
    mask = np.zeros(len(Xs), dtype=bool)
    for i, X in enumerate(Xs):
        d1 = float(z1 @ (X - C1))
        d2 = float(z2 @ (X - C2))
        if d1 > 0 and d2 > 0:
            mask[i] = True
    return mask

def reprojection_errors(P1, P2, pts1, pts2, Xs):
    def proj(P, X):
        Xh = np.hstack([X, np.ones((X.shape[0],1))])
        xh = (P @ Xh.T).T
        uv = xh[:, :2] / xh[:, 2:3]
        return uv
    uv1 = proj(P1, Xs)
    uv2 = proj(P2, Xs)
    e1 = np.linalg.norm(uv1 - pts1, axis=1)
    e2 = np.linalg.norm(uv2 - pts2, axis=1)
    return 0.5*(e1 + e2)

def triangulate_points_dlt(P1, P2, pts1, pts2):
    N = len(pts1)
    Xs = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        u1, v1 = pts1[i]
        u2, v2 = pts2[i]
        A = np.zeros((4,4), dtype=np.float64)
        A[0] = u1 * P1[2] - P1[0]
        A[1] = v1 * P1[2] - P1[1]
        A[2] = u2 * P2[2] - P2[0]
        A[3] = v2 * P2[2] - P2[1]
        _, _, Vh = np.linalg.svd(A)
        Xh = Vh[-1]
        Xs[i] = Xh[:3] / Xh[3]
    return Xs

def filter_by_reprojection(P1, P2, pts1, pts2, Xs, thresh=REPROJ_THRESH_PX):
    errs = reprojection_errors(P1, P2, pts1, pts2, Xs)
    return errs < thresh, errs


def extract_features(image_paths):
    print("Extracting features...")
    sift = cv2.SIFT_create()
    features = {}
    for i, path in enumerate(image_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Failed to read: {path} (skip)")
            continue
        kp, des = sift.detectAndCompute(img, None)
        features[i] = {'kp': kp, 'des': des}
    print(f"Done. Extracted features from {len(features)} images.")
    return features

def match_features(features):
    print("Matching features (BF + Lowe ratio)...")
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = {}
    ids = sorted(list(features.keys()))
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            id1, id2 = ids[i], ids[j]
            des1, des2 = features[id1]['des'], features[id2]['des']
            if des1 is None or des2 is None or len(des1)==0 or len(des2)==0:
                continue
            raw = bf.knnMatch(des1, des2, k=2)
            good = []
            for pair in raw:
                if len(pair) < 2: 
                    continue
                m, n = pair
                if m.distance < RATIO_TEST * n.distance:
                    good.append(m)
            good = dedupe_matches(good)
            if len(good) >= MIN_RAW_MATCHES_PER_PAIR:
                matches[(id1, id2)] = good
    print(f"Done. Matched {len(matches)} pairs (≥{MIN_RAW_MATCHES_PER_PAIR} good matches).")
    return matches

def dedupe_matches(good):
    best_q = {}
    best_t = {}
    for m in good:
        if (m.queryIdx not in best_q) or (m.distance < best_q[m.queryIdx].distance):
            best_q[m.queryIdx] = m
        if (m.trainIdx not in best_t) or (m.distance < best_t[m.trainIdx].distance):
            best_t[m.trainIdx] = m
    uniq = []
    for q, m in best_q.items():
        if best_t.get(m.trainIdx, None) is m:
            uniq.append(m)
    return uniq

def score_initial_pair(kp1, kp2, ms, K):
    pts1 = np.float64([kp1[m.queryIdx].pt for m in ms])
    pts2 = np.float64([kp2[m.trainIdx].pt for m in ms])
    best = (-1.0, None)  # (score, payload)

    for thr in (ESSENTIAL_RANSAC_THRESH, 2.0):  # ladder
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=thr
        )
        if E is None or mask is None or int(mask.sum()) < MIN_INITIAL_INLIERS:
            continue
        in1 = pts1[mask.ravel() == 1]
        in2 = pts2[mask.ravel() == 1]

        retval, R, t, mask_pose = cv2.recoverPose(E, in1, in2, K)
        if retval < MIN_INITIAL_INLIERS:
            continue

        par_deg = compute_parallax_deg(R, in1, in2, K)
        if par_deg < MIN_PARALLAX_DEG:
            continue

        score = float(retval) * (1.0 + par_deg / 5.0)
        if score > best[0]:
            best = (score, (R, t, in1, in2, mask, mask_pose, par_deg, thr))

    return best  # (score, payload)

def find_best_initial_pair(features, matches, K):
    print("Finding best initial pair (cheirality + parallax aware)...")
    best_pair = (-1, -1)
    best_score = -1.0
    for (i, j), ms in matches.items():
        if len(ms) < MIN_RAW_MATCHES_PER_PAIR:
            continue
        kp1, kp2 = features[i]['kp'], features[j]['kp']
        score, payload = score_initial_pair(kp1, kp2, ms, K)
        if score is None or score <= 0:
            continue
        if score > best_score:
            best_score = score
            best_pair = (i, j)
    print(f"Best pair: {best_pair} (score={best_score:.1f})")
    return best_pair

def initialize_map(rec_map, features, matches, initial_pair):
    print("Initializing map...")
    i, j = initial_pair
    pair = (i, j) if (i, j) in matches else (j, i)
    inv  = (pair != (i, j))
    ms   = matches[pair]
    kp1  = features[pair[0]]['kp']
    kp2  = features[pair[1]]['kp']

    if not inv:
        pts1 = np.float64([kp1[m.queryIdx].pt for m in ms])
        pts2 = np.float64([kp2[m.trainIdx].pt for m in ms])
    else:
        pts1 = np.float64([kp1[m.trainIdx].pt for m in ms])
        pts2 = np.float64([kp2[m.queryIdx].pt for m in ms])

    chosen = None
    for thr in (ESSENTIAL_RANSAC_THRESH, 2.0):
        E, mask = cv2.findEssentialMat(pts1, pts2, rec_map.K, method=cv2.RANSAC, prob=0.999, threshold=thr)
        if E is None or mask is None or int(mask.sum()) < MIN_INITIAL_INLIERS:
            continue
        in1 = pts1[mask.ravel()==1]
        in2 = pts2[mask.ravel()==1]
        retval, R, t, mask_pose = cv2.recoverPose(E, in1, in2, rec_map.K)
        if retval >= MIN_INITIAL_INLIERS:
            par_deg = compute_parallax_deg(R, in1, in2, rec_map.K)
            if par_deg >= MIN_PARALLAX_DEG:
                chosen = (R, t, in1, in2, mask, mask_pose)
                break

    if chosen is None:
        raise RuntimeError("Failed to initialize: E/recoverPose not stable even with threshold ladder.")

    R, t, in1, in2, mask, mask_pose = chosen

    R1, t1 = np.eye(3), np.zeros((3,1))
    R2, t2 = R, t
    rec_map.poses[i] = (R1, t1)
    rec_map.poses[j] = (R2, t2)
    rec_map.registered_images.update([i, j])

    inlier_matches = []
    for m, ok in zip(ms, mask.ravel()):
        if not ok: 
            continue
        if not inv:
            inlier_matches.append((m.queryIdx, m.trainIdx))
        else:
            inlier_matches.append((m.trainIdx, m.queryIdx))

    P1 = camera_matrix(R1, t1, rec_map.K)
    P2 = camera_matrix(R2, t2, rec_map.K)

    Xs = triangulate_points_cv(P1, P2, in1, in2)

    ok_ch = cheirality_mask(R1, t1, R2, t2, Xs)
    ok_rep, errs = filter_by_reprojection(P1, P2, in1, in2, Xs, REPROJ_THRESH_PX)
    ok = ok_ch & ok_rep

    added = 0
    for idx, good in enumerate(ok):
        if not good:
            continue
        kp_idx_i, kp_idx_j = inlier_matches[idx]
        pid = rec_map.next_point_id; rec_map.next_point_id += 1
        rec_map.points_3d[pid] = Xs[idx]
        rec_map.point_observers[pid] = [(i, kp_idx_i), (j, kp_idx_j)]
        rec_map.kp_to_point_id[i][kp_idx_i] = pid
        rec_map.kp_to_point_id[j][kp_idx_j] = pid
        added += 1

    print(f"Initial map points: {added} (thr ladder used)")



def find_next_view_to_register(rec_map, features, matches):
    best_id, best_count = -1, -1
    reg = rec_map.registered_images
    for img_id in features.keys():
        if img_id in reg:
            continue
        count = 0
        for reg_id in reg:
            pair = tuple(sorted((img_id, reg_id)))
            if pair not in matches:
                continue
            ms = matches[pair]
            for m in ms:
                if pair[0] == img_id:
                    kp_idx_reg = m.trainIdx
                    ref_id = pair[1]
                else:
                    kp_idx_reg = m.queryIdx
                    ref_id = pair[0]
                if kp_idx_reg in rec_map.kp_to_point_id.get(ref_id, {}):
                    count += 1
        if count > best_count:
            best_id, best_count = img_id, count
    return best_id, best_count

def register_view_with_pnp(rec_map, features, matches, next_view_id):
    print(f"Registering view {next_view_id} via PnP...")
    object_points = []
    image_points = []
    index_map = []  # [(kp_idx_new, point_id)]

    for reg_id in rec_map.registered_images:
        pair = tuple(sorted((next_view_id, reg_id)))
        if pair not in matches:
            continue
        ms = matches[pair]
        for m in ms:
            if pair[0] == next_view_id:
                kp_idx_new = m.queryIdx
                kp_idx_reg = m.trainIdx
                reg_img_id = pair[1]
            else:
                kp_idx_new = m.trainIdx
                kp_idx_reg = m.queryIdx
                reg_img_id = pair[0]

            pid = rec_map.kp_to_point_id.get(reg_img_id, {}).get(kp_idx_reg, None)
            if pid is None:
                continue
            X = rec_map.points_3d[pid]
            uv = features[next_view_id]['kp'][kp_idx_new].pt

            object_points.append(X)
            image_points.append(uv)
            index_map.append((kp_idx_new, pid))

    if len(object_points) < MIN_PNP_2D3D:
        print(f"[FAIL] Not enough 2D-3D correspondences: {len(object_points)} < {MIN_PNP_2D3D}")
        return False

    object_points = ensure_float32(np.vstack(object_points))
    image_points = ensure_float32(np.vstack(image_points))

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points,
        rec_map.K.astype(np.float64),
        distCoeffs=rec_map.dist_coeffs,
        iterationsCount=1500,
        reprojectionError=REPROJ_THRESH_PX,
        flags=cv2.SOLVEPNP_EPNP
    )
    if not ok or inliers is None or len(inliers) < 6:
        print(f"[FAIL] PnP RANSAC failed. Inliers={0 if inliers is None else len(inliers)}")
        return False

    # refine (ITERATIVE) on inliers
    inl = inliers.ravel()
    ok2, rvec, tvec = cv2.solvePnP(
        object_points[inl], image_points[inl],
        rec_map.K.astype(np.float64), rec_map.dist_coeffs,
        rvec, tvec, useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok2:
        print("[WARN] PnP refine failed; using RANSAC pose as-is.")

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3,1)

    
    # 포즈 등록
    rec_map.poses[next_view_id] = (R, t)
    rec_map.registered_images.add(next_view_id)
    
    # 인라이어에 한해 2D-3D 연결 갱신
    if next_view_id not in rec_map.kp_to_point_id:
        rec_map.kp_to_point_id[next_view_id] = {}
        
    for ii in inliers.ravel().tolist():
        kp_new, pid = index_map[ii]
        rec_map.kp_to_point_id[next_view_id][kp_new] = pid
        rec_map.point_observers[pid].append((next_view_id, kp_new))

    print(f"Registered view {next_view_id}: PnP inliers={len(inliers)}")
    return True

def triangulate_points_cv(P1, P2, pts1, pts2):
    Xh = cv2.triangulatePoints(
        P1.astype(np.float64), P2.astype(np.float64),
        pts1.T.astype(np.float64), pts2.T.astype(np.float64)
    )
    X = (Xh[:3] / Xh[3]).T
    return X
def median_pixel_disp(pts1, pts2):
    if len(pts1) == 0:
        return 0.0
    d = np.linalg.norm(pts1 - pts2, axis=1)
    return float(np.median(d))



def triangulate_new_points_for_view(rec_map, features, matches, new_view_id):
    print(f"Triangulating new points for view {new_view_id}...")
    Rn, tn = rec_map.poses[new_view_id]
    Pn = camera_matrix(Rn, tn, rec_map.K)
    kp_new = features[new_view_id]['kp']

    new_points_added = 0

    for reg_id in list(rec_map.registered_images):
        if reg_id == new_view_id:
            continue
        pair = tuple(sorted((new_view_id, reg_id)))
        if pair not in matches:
            continue
            
        ms = matches[pair]
        Rr, tr = rec_map.poses[reg_id]
        Pr = camera_matrix(Rr, tr, rec_map.K)
        kp_reg = features[reg_id]['kp']
        
        pts_new, pts_reg = [], []
        idx_triplets = []  # (kp_idx_new, kp_idx_reg, match_obj)
        
        for m in ms:
            if pair[0] == new_view_id:
                kp_idx_new = m.queryIdx
                kp_idx_reg = m.trainIdx
            else:
                kp_idx_new = m.trainIdx
                kp_idx_reg = m.queryIdx
            
            unmapped_new = kp_idx_new not in rec_map.kp_to_point_id.get(new_view_id, {})
            unmapped_reg = kp_idx_reg not in rec_map.kp_to_point_id.get(reg_id, {})
            if unmapped_new and unmapped_reg:
                pts_new.append(kp_new[kp_idx_new].pt)
                pts_reg.append(kp_reg[kp_idx_reg].pt)
                idx_triplets.append((kp_idx_new, kp_idx_reg, m))
        
        if len(pts_new) == 0:
            continue

        pts_new = np.float64(pts_new)
        pts_reg = np.float64(pts_reg)

        disp = median_pixel_disp(pts_new, pts_reg)
        if disp < PIXEL_DISP_MIN:
            continue
        par_deg = compute_parallax_deg(Rn @ Rr.T, pts_new, pts_reg, rec_map.K)
        if par_deg < MIN_PARALLAX_DEG:
            continue

        Xs = triangulate_points_cv(Pn, Pr, pts_new, pts_reg)

        ok_ch = cheirality_mask(Rn, tn, Rr, tr, Xs)
        ok_rep, _ = filter_by_reprojection(Pn, Pr, pts_new, pts_reg, Xs, REPROJ_THRESH_PX)
        ok = ok_ch & ok_rep

        for idx, good in enumerate(ok):
            if not good:
                continue
            kp_idx_new, kp_idx_reg, _ = idx_triplets[idx]
            pid = rec_map.next_point_id
            rec_map.next_point_id += 1
            rec_map.points_3d[pid] = Xs[idx]
            rec_map.point_observers[pid] = [(new_view_id, kp_idx_new), (reg_id, kp_idx_reg)]
            rec_map.kp_to_point_id[new_view_id][kp_idx_new] = pid
            rec_map.kp_to_point_id[reg_id][kp_idx_reg] = pid
            new_points_added += 1

    print(f"Added new points: {new_points_added}")

# =========================
# 7) 번들 조정 (BA)
# =========================

def run_bundle_adjustment(rec_map, features, fix_first_camera=True, max_nfev=50):
    print("Running bundle adjustment (scipy.optimize.least_squares)...")

    cam_ids = sorted(list(rec_map.registered_images))
    if fix_first_camera and len(cam_ids) > 0:
        fixed_cam_id = cam_ids[0]
    else:
        fixed_cam_id = None

    cam_index = {}
    cam_params = []
    for cid in cam_ids:
        R, t = rec_map.poses[cid]
        rvec = R_to_rodrigues(R)
        if (fixed_cam_id is not None) and (cid == fixed_cam_id):
            cam_index[cid] = None
            fixed_pose = (rvec.copy(), t.reshape(3,).copy())
        else:
            cam_index[cid] = len(cam_params)
            cam_params.append(np.hstack([rvec, t.reshape(3,)]))
    cam_params = np.array(cam_params, dtype=np.float64).reshape(-1,6)

    pt_ids = sorted(list(rec_map.points_3d.keys()))
    pt_index = {pid: idx for idx, pid in enumerate(pt_ids)}
    X_init = np.array([rec_map.points_3d[pid] for pid in pt_ids], dtype=np.float64).reshape(-1,3)

    obs_cam_ids = []
    obs_pt_ids = []
    obs_uv = []
    for pid in pt_ids:
        for (img_id, kp_idx) in rec_map.point_observers[pid]:
            if img_id not in rec_map.poses:
                continue
            uv = features[img_id]['kp'][kp_idx].pt
            obs_cam_ids.append(img_id)
            obs_pt_ids.append(pid)
            obs_uv.append(uv)

    obs_uv = np.array(obs_uv, dtype=np.float64).reshape(-1,2)

    if len(obs_uv) == 0:
        print("[BA] No observations; skip.")
        return

    def pack(cam_params, X):
        return np.hstack([cam_params.ravel(), X.ravel()])

    def unpack(theta):
        n_cam_vars = cam_params.size
        cam_params_new = theta[:n_cam_vars].reshape(cam_params.shape)
        X_new = theta[n_cam_vars:].reshape(X_init.shape)
        return cam_params_new, X_new

    def residuals(theta):
        cam_params_new, X_new = unpack(theta)
        res = []

        pose_cache = {}
        for cid in cam_ids:
            if cid == fixed_cam_id:
                rvec, tvec = fixed_pose
            else:
                idx = cam_index[cid]
                rvec = cam_params_new[idx, :3]
                tvec = cam_params_new[idx, 3:]
            R = rodrigues_to_R(rvec)
            t = tvec.reshape(3,1)
            pose_cache[cid] = (R, t)

        for k in range(len(obs_uv)):
            cid = obs_cam_ids[k]
            pid = obs_pt_ids[k]
            R, t = pose_cache[cid]
            X = X_new[pt_index[pid]].reshape(1,3)
            uv_hat = project_points(X, R, t, rec_map.K).reshape(2,)
            res.extend(obs_uv[k] - uv_hat)
        return np.array(res, dtype=np.float64)

    theta0 = pack(cam_params, X_init)

    result = least_squares(
        residuals, theta0, method='trf',
        loss='huber', f_scale=1.0,
        max_nfev=max_nfev, verbose=1
    )

    if not result.success:
        print(f"[BA] Optimization did not converge: {result.message}")

    cam_params_opt, X_opt = unpack(result.x)

    for cid in cam_ids:
        if cid == fixed_cam_id:
            rvec, tvec = fixed_pose
        else:
            idx = cam_index[cid]
            rvec = cam_params_opt[idx, :3]
            tvec = cam_params_opt[idx, 3:]
        R = rodrigues_to_R(rvec)
        t = tvec.reshape(3,1)
        rec_map.poses[cid] = (R, t)

    for pid in pt_ids:
        rec_map.points_3d[pid] = X_opt[pt_index[pid]]

    print("BA finished.")

def visualize_map(rec_map, save_path: str = "sfm_viz.png", width: int = 1600, height: int = 1200, dpi: int = 100, bg_color=(1,1,1)):
    print("Saving visualization image...")
    fig = plt.figure(figsize=(width/float(dpi), height/float(dpi)), dpi=dpi)
    fig.patch.set_facecolor(bg_color)
    ax = fig.add_subplot(111, projection='3d', facecolor=bg_color)

    if len(rec_map.points_3d) > 0:
        pts = np.array(list(rec_map.points_3d.values()))
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='b', marker='.', s=2, depthshade=False)

    for img_id, (R, t) in rec_map.poses.items():
        C = (-R.T @ t).ravel()
        ax.scatter(C[0], C[1], C[2], c='r', marker='^')
        z = R[2,:]
        ax.quiver(C[0], C[1], C[2], z[0], z[1], z[2], length=0.5, color='r')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])
    try:
        if len(rec_map.points_3d) > 0:
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
            centers = (mins + maxs) / 2.0
            span = float(np.max(maxs - mins) or 1.0)
            ax.set_xlim(centers[0]-span/2, centers[0]+span/2)
            ax.set_ylim(centers[1]-span/2, centers[1]+span/2)
            ax.set_zlim(centers[2]-span/2, centers[2]+span/2)
    except Exception:
        pass
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved to: {save_path}")

def main():
    image_dir = "/home/mori/lab/issol-ku/introduction_to_cv/Structure_from_motion/data/input/photo/mnms6"
    image_paths = sorted(glob.glob(f"{image_dir}/*.jpg") + 
                         glob.glob(f"{image_dir}/*.jpeg") + 
                         glob.glob(f"{image_dir}/*.png"))
    if len(image_paths) < 2:
        print("[ERROR] Need at least 2 images.")
        return

    features = extract_features(image_paths)
    if len(features) < 2:
        print("[ERROR] Feature extraction failed or too few images.")
        return

    rec_map = ReconstructionMap()
    for i, p in enumerate(image_paths):
        if i in features:
            rec_map.image_id_to_path[i] = p
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                rec_map.image_sizes[i] = img.shape[:2]

    matches = match_features(features)
    if len(matches) == 0:
        print("[ERROR] No valid pairs.")
        return

    init_pair = find_best_initial_pair(features, matches, rec_map.K)
    if init_pair == (-1, -1):
        print("[ERROR] Failed to find initial pair.")
        return
        
    initialize_map(rec_map, features, matches, init_pair)
    
    # Incremental loopx
    while True:
        visualize_map(rec_map, save_path=save_path, width=3000, height=3000, dpi=120)
        next_id, cnt = find_next_view_to_register(rec_map, features, matches)
        if next_id == -1:
            print("No more views to register.")
            break
        print(f"--- Next view: {next_id} (2D-3D matches estimate: {cnt}) ---")

        ok = register_view_with_pnp(rec_map, features, matches, next_id)
        if not ok:
            print(f"[SKIP] View {next_id} registration failed.")
            break
            
        triangulate_new_points_for_view(rec_map, features, matches, next_id)

        if len(rec_map.registered_images) % 5 == 0:
            run_bundle_adjustment(rec_map, features, fix_first_camera=True, max_nfev=80)

    run_bundle_adjustment(rec_map, features, fix_first_camera=True, max_nfev=100)

    visualize_map(rec_map, save_path=save_path, width=3000, height=3000, dpi=120)

if __name__ == "__main__":
    main()
