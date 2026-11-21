"""Scale and align a COLMAP model to the ChArUco board: metres, board plane at z = 0.

    python align_board.py SPLAT_DIR

Reads SPLAT_DIR/dense (undistorted PINHOLE model, as text in dense/sparse_txt), detects board
corners in dense/images, triangulates them from the COLMAP poses, fits a similarity onto the
board's known grid, and writes SPLAT_DIR/board_sim3.json plus the transformed text model in
dense/sparse_metric. The residual is the reconstruction's metric error on the desk plane.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

BOARD = cv2.aruco.CharucoBoard(
    (9, 6), 0.028, 0.021, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
)


def qvec_to_rot(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def rot_to_qvec(R):
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return np.array(
        [
            w,
            (R[2, 1] - R[1, 2]) / (4 * w),
            (R[0, 2] - R[2, 0]) / (4 * w),
            (R[1, 0] - R[0, 1]) / (4 * w),
        ]
    )


def read_model(folder: Path):
    cams = {}
    for line in (folder / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cid, model, w, h, *p = line.split()
        assert model == "PINHOLE", model
        fx, fy, cx, cy = map(float, p)
        cams[int(cid)] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    images = {}
    lines = [l for l in (folder / "images.txt").read_text().splitlines() if not l.startswith("#")]
    for header in lines[0::2]:
        if not header.strip():
            continue
        iid, qw, qx, qy, qz, tx, ty, tz, cid, name = header.split()
        R = qvec_to_rot(np.array([qw, qx, qy, qz], float))
        images[name] = (cams[int(cid)], R, np.array([tx, ty, tz], float))
    return images


def triangulate(observations):
    """Linear triangulation from [(K, R, t, uv)]."""
    rows = []
    for K, R, t, uv in observations:
        P = K @ np.hstack([R, t[:, None]])
        rows += [uv[0] * P[2] - P[0], uv[1] * P[2] - P[1]]
    _, _, vt = np.linalg.svd(np.array(rows))
    X = vt[-1]
    return X[:3] / X[3]


def umeyama(src, dst):
    """Similarity s, R, t minimizing |s R src + t - dst|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / a.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def main() -> None:
    root = Path(sys.argv[1])
    dense = root / "dense"
    images = read_model(dense / "sparse_txt")
    detector = cv2.aruco.CharucoDetector(BOARD)
    obs: dict[int, list] = {}
    for name, (K, R, t) in images.items():
        gray = cv2.imread(str(dense / "images" / name), cv2.IMREAD_GRAYSCALE)
        corners, ids, _, _ = detector.detectBoard(gray)
        if ids is None or len(ids) < 6:
            continue
        for cid, uv in zip(ids.ravel(), corners.reshape(-1, 2), strict=True):
            obs.setdefault(int(cid), []).append((K, R, t, uv))

    grid = BOARD.getChessboardCorners()
    src, dst, counts = [], [], []
    for cid, o in obs.items():
        if len(o) < 3:
            continue
        X = triangulate(o)
        errs = [
            np.linalg.norm((K @ (R @ X + t))[:2] / (K @ (R @ X + t))[2] - uv) for K, R, t, uv in o
        ]
        if np.median(errs) > 2.0:
            continue
        src.append(X)
        dst.append(grid[cid])
        counts.append(len(o))
    src, dst = np.array(src), np.array(dst)
    s, R, t = umeyama(src, dst)
    resid = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1) * 1000
    # The board frame's z may point into the desk; flip about x so cameras sit at z > 0.
    centers = np.array([-Rc.T @ tc for _, Rc, tc in images.values()])
    if np.median((s * (R @ centers.T)).T[:, 2] + t[2]) < 0:
        F = np.diag([1.0, -1.0, -1.0])
        R, t = F @ R, F @ t
    print(
        f"corners {len(src)}/40 (median {int(np.median(counts))} views each), "
        f"scale {s:.5f} m per unit, residual median {np.median(resid):.2f} mm, "
        f"max {resid.max():.2f} mm"
    )

    # Rewrite the text model in board coordinates: X_board = s R X + t.
    out = dense / "sparse_metric"
    out.mkdir(exist_ok=True)
    (out / "cameras.txt").write_text((dense / "sparse_txt" / "cameras.txt").read_text())
    lines = (dense / "sparse_txt" / "images.txt").read_text().splitlines()
    new = []
    body = [l for l in lines if not l.startswith("#")]
    for i, l in enumerate(body):
        if i % 2 == 1 or not l.strip():
            new.append(l)
            continue
        iid, qw, qx, qy, qz, tx, ty, tz, cid, name = l.split()
        Rc = qvec_to_rot(np.array([qw, qx, qy, qz], float))
        tc = np.array([tx, ty, tz], float)
        # Camera: x = Rc X + tc with X = R^T (Xb - t) / s  ->  x' = s x = Rc R^T Xb + (s tc - Rc R^T t)
        Rn = Rc @ R.T
        tn = s * tc - Rn @ t
        q = rot_to_qvec(Rn)
        new.append(
            " ".join([iid, *[f"{v:.12g}" for v in q], *[f"{v:.12g}" for v in tn], cid, name])
        )
    (out / "images.txt").write_text("\n".join(new) + "\n")
    pts = []
    for l in (dense / "sparse_txt" / "points3D.txt").read_text().splitlines():
        if l.startswith("#") or not l.strip():
            continue
        f = l.split()
        X = s * R @ np.array(f[1:4], float) + t
        pts.append(
            " ".join([f[0], *[f"{v:.9g}" for v in X], *f[4:7], f"{float(f[7]) * s:.6g}", *f[8:]])
        )
    (out / "points3D.txt").write_text("\n".join(pts) + "\n")
    (root / "board_sim3.json").write_text(
        json.dumps(
            {
                "scale": s,
                "R": R.tolist(),
                "t": t.tolist(),
                "residual_mm_median": float(np.median(resid)),
                "residual_mm_max": float(resid.max()),
                "corners": len(src),
                "board_center": ((s * (R @ src.T)).T + t).mean(0).tolist(),
                "note": "X_board = scale * R @ X_colmap + t; board plane z = 0, origin at board corner 0",
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
