"""
Surface reconstruction cross-comparison using Open3D.

Three algorithms are run on every .xyz file:
  1. Screened Poisson Surface Reconstruction
  2. Ball-Pivoting Algorithm (BPA)
  3. Alpha Shape

Each result is saved as  <stem>_<algo>.ply  and all three meshes are
displayed together (offset side-by-side) in a single window, coloured
distinctly, alongside the original point cloud.

Usage:
    python reconstruct.py                  # all .xyz files in this folder
    python reconstruct.py file1.xyz ...    # specific files
    python reconstruct.py --no-viz ...     # skip visualisation (batch mode)
"""

import os
import sys
import time

import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _parse_points(path: str, delimiter: str | None) -> list:
    pts = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = line.split(delimiter) if delimiter else line.split()
            if len(parts) >= 3:
                try:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
    return pts


def _make_pcd(pts: list) -> o3d.geometry.PointCloud:
    arr = np.array(pts, dtype=np.float64).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    return pcd


def load_xyz(path: str) -> o3d.geometry.PointCloud:
    delimiter = None
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("//"):
                if ";" in line:
                    delimiter = ";"
                elif "," in line:
                    delimiter = ","
                break
    pts = _parse_points(path, delimiter)
    return _make_pcd(pts)


# ---------------------------------------------------------------------------
# Normal estimation
# ---------------------------------------------------------------------------

def estimate_normals(pcd: o3d.geometry.PointCloud, knn: int = 30) -> None:
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd.orient_normals_consistent_tangent_plane(k=knn)


# ---------------------------------------------------------------------------
# Algorithm 1 – Screened Poisson
# ---------------------------------------------------------------------------

def reconstruct_poisson(
    pcd: o3d.geometry.PointCloud,
    depth: int = 9,
    density_quantile: float = 0.05,
) -> o3d.geometry.TriangleMesh:
    """
    Screened Poisson Surface Reconstruction.
    Produces a watertight mesh; low-density boundary artefacts are trimmed.
    """
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    densities = np.asarray(densities)
    mask = densities < np.quantile(densities, density_quantile)
    mesh.remove_vertices_by_mask(mask)
    mesh.compute_vertex_normals()
    return mesh


# ---------------------------------------------------------------------------
# Algorithm 2 – Ball-Pivoting Algorithm (BPA)
# ---------------------------------------------------------------------------

def reconstruct_bpa(
    pcd: o3d.geometry.PointCloud,
    scale_factors: tuple = (1.0, 2.0, 4.0),
) -> o3d.geometry.TriangleMesh:
    """
    Ball-Pivoting Algorithm.
    Works well when sampling density is reasonably uniform.
    Radii are chosen automatically from the average nearest-neighbour distance.
    """
    avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())
    radii = [avg_dist * s for s in scale_factors]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    mesh.compute_vertex_normals()
    return mesh


# ---------------------------------------------------------------------------
# Algorithm 3 – Alpha Shape
# ---------------------------------------------------------------------------

def reconstruct_alpha_shape(
    pcd: o3d.geometry.PointCloud,
    alpha: float | None = None,
) -> o3d.geometry.TriangleMesh:
    """
    Alpha Shape reconstruction (Delaunay-based).
    Alpha is estimated as 5 % of the bounding-box diagonal when not specified.
    Smaller alpha → finer detail but more holes; larger → smoother, fewer holes.
    """
    if alpha is None:
        bbox = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(
            np.asarray(bbox.max_bound) - np.asarray(bbox.min_bound)
        )
        alpha = diag * 0.05
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    return mesh


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

ALGORITHMS: dict[str, dict] = {
    "poisson": {
        "fn": reconstruct_poisson,
        "label": "Screened Poisson",
        "color": [0.85, 0.85, 0.85],   # light grey
    },
    "bpa": {
        "fn": reconstruct_bpa,
        "label": "Ball-Pivoting (BPA)",
        "color": [1.00, 0.75, 0.40],   # warm orange
    },
    "alpha_shape": {
        "fn": reconstruct_alpha_shape,
        "label": "Alpha Shape",
        "color": [0.45, 0.85, 0.55],   # green
    },
}


# ---------------------------------------------------------------------------
# Visualisation – tile geometries side by side along X
# ---------------------------------------------------------------------------

def _tile_geometries(
    geometries: list,
    gap: float = 0.1,
) -> list:
    """
    Translate each geometry so they sit next to each other along the X axis
    without overlapping.  Returns new translated copies (originals unchanged).
    """
    tiled = []
    cursor_x = 0.0
    for geom in geometries:
        bbox = geom.get_axis_aligned_bounding_box()
        extents = np.asarray(bbox.max_bound) - np.asarray(bbox.min_bound)
        offset = np.array([cursor_x - float(bbox.min_bound[0]), 0.0, 0.0])
        tiled.append(geom.translate(offset, relative=True))
        cursor_x += extents[0] + gap
    return tiled


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------

def process_file(path: str, visualise: bool = True) -> None:
    name = os.path.basename(path)
    stem = os.path.splitext(path)[0]

    print(f"\n{'=' * 65}")
    print(f"File : {name}")

    pcd = load_xyz(path)
    n_pts = len(pcd.points)
    if n_pts < 10:
        print(f"  [SKIP] Too few points ({n_pts}).")
        return
    print(f"  Points loaded : {n_pts:,}")

    if n_pts > 200_000:
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        print(f"  Downsampled   : {len(pcd.points):,} points")

    print("  Estimating normals …")
    estimate_normals(pcd)

    # ------------------------------------------------------------------
    # Run every algorithm
    # ------------------------------------------------------------------
    results: dict[str, dict | None] = {}

    print(f"\n  {'Algorithm':<26} {'Vertices':>10} {'Triangles':>10} {'Time (s)':>10}")
    print(f"  {'-'*26} {'-'*10} {'-'*10} {'-'*10}")

    for key, cfg in ALGORITHMS.items():
        t0 = time.perf_counter()
        try:
            mesh: o3d.geometry.TriangleMesh = cfg["fn"](pcd)
            elapsed = time.perf_counter() - t0
            n_v = len(mesh.vertices)
            n_t = len(mesh.triangles)
            print(f"  {cfg['label']:<26} {n_v:>10,} {n_t:>10,} {elapsed:>10.2f}")
            results[key] = {"mesh": mesh, "elapsed": elapsed}
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  {cfg['label']:<26} {'ERROR':>10} {'':>10} {elapsed:>10.2f}  ({exc})")
            results[key] = None

    # ------------------------------------------------------------------
    # Save meshes
    # ------------------------------------------------------------------
    print()
    for key, res in results.items():
        if res is None:
            continue
        out = f"{stem}_{key}.ply"
        o3d.io.write_triangle_mesh(out, res["mesh"])
        print(f"  Saved : {os.path.basename(out)}")

    # ------------------------------------------------------------------
    # Visualise – original cloud + all three meshes tiled side by side
    # ------------------------------------------------------------------
    if not visualise:
        return

    geoms_to_show = []
    labels = []

    pcd_vis = o3d.geometry.PointCloud(pcd)
    pcd_vis.paint_uniform_color([0.20, 0.45, 0.85])
    geoms_to_show.append(pcd_vis)
    labels.append("Original cloud (blue)")

    for key, res in results.items():
        if res is None:
            continue
        m = o3d.geometry.TriangleMesh(res["mesh"])
        m.paint_uniform_color(ALGORITHMS[key]["color"])
        geoms_to_show.append(m)
        labels.append(ALGORITHMS[key]["label"])

    tiled = _tile_geometries(geoms_to_show, gap=0.15)

    layout = "  |  ".join(f"{i+1}: {lbl}" for i, lbl in enumerate(labels))
    print(f"\n  Layout (left→right): {layout}")
    print("  Displaying (close window to continue) …")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"Reconstruction Comparison — {name}",
        width=1600,
        height=900,
    )
    for geom in tiled:
        vis.add_geometry(geom)
    # Disable backface culling so triangles are visible from both sides
    vis.get_render_option().mesh_show_back_face = True
    vis.run()
    vis.destroy_window()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]
    visualise = "--no-viz" not in args
    args = [a for a in args if a != "--no-viz"]

    if args:
        files = args
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        files = sorted(
            os.path.join(script_dir, f)
            for f in os.listdir(script_dir)
            if f.lower().endswith(".xyz")
        )

    if not files:
        print("No .xyz files found.")
        return

    print(f"Processing {len(files)} file(s)  [visualise={visualise}]")
    for f in files:
        print(f"  {f}")

    for f in files:
        if not os.path.isfile(f):
            print(f"[WARN] Not found: {f}")
            continue
        process_file(f, visualise=visualise)

    print("\nDone.")


if __name__ == "__main__":
    main()
