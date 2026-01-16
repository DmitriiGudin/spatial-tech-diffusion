#!/usr/bin/env python3
"""
density_utils.py

Approximate population density (persons / km^2) at an arbitrary lon/lat location
and year, using a cKDTree over ZIP centroids precomputed by build_ZIP_tree.py.

Requires:
    data/processed/zip_kdtree_data.npz

Expected arrays inside .npz (either naming is accepted):
    - coords_km : (N, 2) [x_km, y_km]
    - pop2010 or population_2010 : (N,)
    - pop2020 or population_2020 : (N,)
"""

from pathlib import Path
import numpy as np
from typing import Sequence
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Optional dependencies for mesh-regime
try:
    import matplotlib.tri as mtri
except Exception:
    mtri = None

try:
    from pyproj import Transformer
except Exception:
    Transformer = None


# -------------------------------------------------------------------
# Mesh-regime cache
# -------------------------------------------------------------------

# Cache nodal density vectors for a given (mesh_id, year, epsg_project)
# value: dict with keys: "rho_nodes", "finder", "tri_nodes", "mesh_xy_km"
_MESH_DENSITY_CACHE: dict[tuple[int, float, int], dict] = {}

# -------------------------------------------------------------------
# Mesh-regime cache (batch nodal density)
# -------------------------------------------------------------------

# Cache mesh geometry mapping needed for nodal density computation.
# key = (id(mesh), epsg_project)
# value: dict with keys:
#   - tri_nodes (ntri,3)
#   - areas_tri (ntri,)
#   - A_nodes (N,)         control-volume areas sum(area_T/3)
#   - zip_tri_ids (nzip,)  triangle id for each ZIP centroid in ZIP-plane (-1 outside)
#   - zip_inside_idx (m,)  indices of ZIP centroids that are inside mesh (zip_tri_ids>=0)
#   - epsg_project
_MESH_NODAL_GEOM_CACHE: dict[tuple[int, int], dict] = {}

# Cache computed rho_nodes for (mesh, epsg_project, year)
# key = (id(mesh), epsg_project, year_float)
# value: dict with keys: rho_nodes, N_nodes
_MESH_NODAL_YEAR_CACHE: dict[tuple[int, int, float], dict] = {}

# -------------------------------------------------------------------
# Clear caches when changing the procedure somewhere
# -------------------------------------------------------------------
def clear_density_caches() -> None:
    _MESH_DENSITY_CACHE.clear()
    _MESH_NODAL_GEOM_CACHE.clear()
    _MESH_NODAL_YEAR_CACHE.clear()


def _triangle_areas_km2(mesh) -> np.ndarray:
    """Areas for each triangle, mesh coords are km."""
    tri = mesh.t  # (3, ntri)
    p = mesh.p    # (2, N)
    x1, y1 = p[0, tri[0]], p[1, tri[0]]
    x2, y2 = p[0, tri[1]], p[1, tri[1]]
    x3, y3 = p[0, tri[2]], p[1, tri[2]]
    return 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def _assemble_p1_mass_matrix(mesh) -> csr_matrix:
    """
    Assemble global P1 (CG) mass matrix on a triangular mesh.
    Local: (A/12) * [[2,1,1],[1,2,1],[1,1,2]]
    """
    tri = mesh.t.T  # (ntri, 3)
    n = mesh.p.shape[1]
    A = _triangle_areas_km2(mesh)  # (ntri,)

    # entries per triangle: 9
    I = np.repeat(tri, 3, axis=1).reshape(-1)          # (ntri*9,)
    J = np.tile(tri, (1, 3)).reshape(-1)               # (ntri*9,)

    # build local matrices
    # diag entries: 2*A/12 = A/6
    # off diag: 1*A/12
    vals = np.empty((tri.shape[0], 9), dtype=float)
    vals[:, :] = (A / 12.0)[:, None]
    # set diagonal positions (0,0),(1,1),(2,2) in flattened 3x3: 0,4,8
    vals[:, 0] *= 2.0
    vals[:, 4] *= 2.0
    vals[:, 8] *= 2.0

    M = coo_matrix((vals.reshape(-1), (I, J)), shape=(n, n)).tocsr()
    return M


def _barycentric_weights(x, y, x1, y1, x2, y2, x3, y3) -> tuple[float, float, float]:
    """Barycentric coordinates of (x,y) w.r.t triangle (x1,y1),(x2,y2),(x3,y3)."""
    detT = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if detT == 0:
        return (0.0, 0.0, 0.0)
    w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / detT
    w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / detT
    w3 = 1.0 - w1 - w2
    return (w1, w2, w3)


def _build_mesh_based_density_cache(mesh, year: float, epsg_project: int) -> dict:
    """
    Build nodal density rho_nodes (persons/km^2) on the given skfem MeshTri
    by:
      - binning ZIP centroid populations into mesh triangles (in ZIP-plane)
      - forming triangle density rho_T = pop_T / area_T
      - L2-projecting rho_T onto P1 nodes: M rho = b
    """
    if mtri is None:
        raise ImportError("matplotlib is required for mesh-based density (matplotlib.tri).")
    if Transformer is None:
        raise ImportError("pyproj is required for mesh-based density (pyproj.Transformer).")

    year = float(year)
    epsg_project = int(epsg_project)

    # --- 1) Map mesh nodes to ZIP-plane (same mapping as KDTree) ---
    # mesh coords are EPSG:5070 in km; need lon/lat then lonlat_to_km()
    pts_xy_km = mesh.p.T  # (N,2) in km

    inv = Transformer.from_crs(f"EPSG:{epsg_project}", "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(pts_xy_km[:, 0] * 1000.0, pts_xy_km[:, 1] * 1000.0)

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    zip_nodes_xy = np.column_stack([
        EARTH_RADIUS_KM * lon_rad * np.cos(lat_rad),
        EARTH_RADIUS_KM * lat_rad
    ]).astype(float)

    tri_nodes = mesh.t.T  # (ntri,3)

    # --- 2) Bin ZIP centroids into triangles (ZIP-plane) ---
    pops_t = _extrapolate_population_vector(POP2010, POP2020, year)
    tri_obj = mtri.Triangulation(zip_nodes_xy[:, 0], zip_nodes_xy[:, 1], tri_nodes)
    finder = tri_obj.get_trifinder()

    tri_ids = finder(COORDS_KM[:, 0], COORDS_KM[:, 1])  # -1 outside
    inside = tri_ids >= 0
    ntri = tri_nodes.shape[0]

    pop_tri = np.bincount(
        tri_ids[inside].astype(np.int64),
        weights=pops_t[inside],
        minlength=ntri
    ).astype(float)  # persons per triangle

    # --- 3) Triangle density rho_T = pop_T / area_T ---
    areas = _triangle_areas_km2(mesh)  # km^2
    rho_T = np.zeros_like(pop_tri)
    good = areas > 0
    rho_T[good] = pop_tri[good] / areas[good]  # persons/km^2

    # --- 4) L2 projection: M rho = b ---
    # b_i = ∫ rho_T phi_i = sum_T rho_T(T) * (area_T/3) for each vertex in T
    # Since rho_T*area_T = pop_T, this is simply: b adds pop_T/3 to each of 3 vertices.
    n = mesh.p.shape[1]
    b = np.zeros(n, dtype=float)
    for t in range(ntri):
        if pop_tri[t] == 0:
            continue
        v0, v1, v2 = tri_nodes[t]
        add = pop_tri[t] / 3.0
        b[v0] += add
        b[v1] += add
        b[v2] += add

    M = _assemble_p1_mass_matrix(mesh)

    rho_nodes = spsolve(M, b)  # persons/km^2
    rho_nodes = np.asarray(rho_nodes, dtype=float)

    # numerical cleanup
    rho_nodes[~np.isfinite(rho_nodes)] = 0.0
    rho_nodes = np.clip(rho_nodes, 0.0, None)

    # Build geometric finder in EPSG:5070 km coords for queries
    elem_finder = mesh.element_finder()

    return {
        "rho_nodes": rho_nodes,
        "elem_finder": elem_finder,
        "tri_nodes": tri_nodes,
        "mesh_xy_km": pts_xy_km,   # (N,2)
        "epsg_project": epsg_project,
    }


def _get_mesh_density_cache(mesh, year: float, epsg_project: int) -> dict:
    key = (id(mesh), float(year), int(epsg_project))
    if key not in _MESH_DENSITY_CACHE:
        _MESH_DENSITY_CACHE[key] = _build_mesh_based_density_cache(mesh, year=year, epsg_project=epsg_project)
    return _MESH_DENSITY_CACHE[key]


def _get_density_mesh_based(
    lon: float,
    lat: float,
    year: float,
    mesh,
    epsg_project: int = 5070,
) -> float:
    """
    Evaluate mesh-based density rho_h(lon,lat) by:
      - projecting lon/lat -> EPSG:5070 -> km
      - locating containing triangle
      - barycentric interpolation of nodal rho
    """
    if Transformer is None:
        raise ImportError("pyproj is required for mesh-based density (pyproj.Transformer).")

    cache = _get_mesh_density_cache(mesh, year=year, epsg_project=epsg_project)
    rho_nodes = cache["rho_nodes"]
    tri_nodes = cache["tri_nodes"]
    pts_xy_km = cache["mesh_xy_km"]
    finder = cache["elem_finder"]

    # project query lon/lat -> EPSG:5070 meters -> km
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_project}", always_xy=True)
    x_m, y_m = fwd.transform(float(lon), float(lat))
    x_km, y_km = x_m / 1000.0, y_m / 1000.0

    try:
        tri_id = int(finder(np.array([x_km]), np.array([y_km]))[0])
    except ValueError:
        # skfem throws if point is outside the mesh
        return 0.0
    
    if tri_id < 0:
        return 0.0

    v0, v1, v2 = tri_nodes[tri_id]
    x1, y1 = pts_xy_km[v0]
    x2, y2 = pts_xy_km[v1]
    x3, y3 = pts_xy_km[v2]

    w1, w2, w3 = _barycentric_weights(x_km, y_km, x1, y1, x2, y2, x3, y3)
    dens = w1 * rho_nodes[v0] + w2 * rho_nodes[v1] + w3 * rho_nodes[v2]
    return float(max(dens, 0.0))

# -------------------------------------------------------------------
# Paths / constants
# -------------------------------------------------------------------

BASE = Path("data")
TREE_NPZ = BASE / "processed" / "zip_kdtree_data.npz"

EARTH_RADIUS_KM = 6371.0

# -------------------------------------------------------------------
# Load KDTree data
# -------------------------------------------------------------------

_data = np.load(TREE_NPZ)

if "coords_km" not in _data.files:
    raise KeyError(f"'coords_km' not found in {TREE_NPZ}. Keys are: {_data.files}")

COORDS_KM = _data["coords_km"].astype(float)

# Accept either key naming convention
if "population_2010" in _data.files:
    POP2010 = _data["population_2010"].astype(float)
elif "pop2010" in _data.files:
    POP2010 = _data["pop2010"].astype(float)
else:
    raise KeyError(f"2010 population array not found in {TREE_NPZ}. Keys are: {_data.files}")

if "population_2020" in _data.files:
    POP2020 = _data["population_2020"].astype(float)
elif "pop2020" in _data.files:
    POP2020 = _data["pop2020"].astype(float)
else:
    raise KeyError(f"2020 population array not found in {TREE_NPZ}. Keys are: {_data.files}")

if COORDS_KM.shape[0] != POP2010.shape[0] or COORDS_KM.shape[0] != POP2020.shape[0]:
    raise ValueError("Array length mismatch between coords_km and population arrays.")

TREE = cKDTree(COORDS_KM)

# -------------------------------------------------------------------
# Coordinate transform: lon/lat -> x/y in km
# MUST match build_ZIP_tree.py
# -------------------------------------------------------------------

def lonlat_to_km(lon: float, lat: float) -> np.ndarray:
    """
    Convert lon/lat in degrees to (x,y) in kilometers using the SAME mapping
    as build_ZIP_tree.py:

        x = R * lon_rad * cos(lat_rad)
        y = R * lat_rad

    Note: this is not a conformal projection, but is fast and consistent
    for local distance queries inside the KDTree.
    """
    lon = float(lon)
    lat = float(lat)

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    x = EARTH_RADIUS_KM * lon_rad * np.cos(lat_rad)
    y = EARTH_RADIUS_KM * lat_rad
    return np.array([x, y], dtype=float)

# -------------------------------------------------------------------
# Temporal extrapolation: exponential between 2010 and 2020
# -------------------------------------------------------------------

def _extrapolate_population_vector(p2010: np.ndarray,
                                  p2020: np.ndarray,
                                  year: float) -> np.ndarray:
    """
    Exponential interpolation/extrapolation:

        p(t) = p2010 * exp(g * (t - 2010)),
        g = (1/10) * log(p2020 / p2010)

    Rules:
    - If both > 0: exponential.
    - If only one > 0: constant in time.
    - Else: 0.
    - Clip at 0 and round to nearest integer.
    """
    year = float(year)
    p2010 = np.asarray(p2010, dtype=float)
    p2020 = np.asarray(p2020, dtype=float)

    out = np.zeros_like(p2010, dtype=float)

    mask_both = (p2010 > 0) & (p2020 > 0)
    if np.any(mask_both):
        ratio = p2020[mask_both] / p2010[mask_both]
        g = np.log(ratio) / 10.0
        out[mask_both] = p2010[mask_both] * np.exp(g * (year - 2010.0))

    mask_10_only = (p2010 > 0) & (p2020 <= 0)
    out[mask_10_only] = p2010[mask_10_only]

    mask_20_only = (p2010 <= 0) & (p2020 > 0)
    out[mask_20_only] = p2020[mask_20_only]

    out = np.clip(out, 0.0, None)
    out = np.round(out)
    return out

# -------------------------------------------------------------------
# Batch density helpers
# -------------------------------------------------------------------

def _nodal_control_volume_areas_km2(mesh) -> np.ndarray:
    """
    A_i = sum_{T contains i} area(T)/3   (mass-lumped P1 area per vertex)
    """
    tri_nodes = mesh.t.T  # (ntri,3)
    n = mesh.p.shape[1]
    areas = _triangle_areas_km2(mesh)  # (ntri,)

    A_nodes = np.zeros(n, dtype=float)
    # add area/3 to each of the 3 vertices
    w = areas / 3.0
    np.add.at(A_nodes, tri_nodes[:, 0], w)
    np.add.at(A_nodes, tri_nodes[:, 1], w)
    np.add.at(A_nodes, tri_nodes[:, 2], w)
    return A_nodes


def _build_mesh_nodal_geom_cache(mesh, epsg_project: int) -> dict:
    """
    Precompute all geometry-only objects needed to map ZIP centroids to mesh triangles
    (in ZIP-plane), and the nodal control-volume areas A_nodes.
    """
    if mtri is None:
        raise ImportError("matplotlib is required for mesh-based density (matplotlib.tri).")
    if Transformer is None:
        raise ImportError("pyproj is required for mesh-based density (pyproj.Transformer).")

    epsg_project = int(epsg_project)

    tri_nodes = mesh.t.T  # (ntri,3)
    areas_tri = _triangle_areas_km2(mesh)  # (ntri,)
    A_nodes = _nodal_control_volume_areas_km2(mesh)  # (N,)

    # Map mesh nodes (EPSG:epsg_project km) -> lon/lat -> ZIP-plane xy (lonlat_to_km mapping)
    pts_xy_km = mesh.p.T  # (N,2) in km EPSG coords
    inv = Transformer.from_crs(f"EPSG:{epsg_project}", "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(pts_xy_km[:, 0] * 1000.0, pts_xy_km[:, 1] * 1000.0)

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    zip_nodes_xy = np.column_stack([
        EARTH_RADIUS_KM * lon_rad * np.cos(lat_rad),
        EARTH_RADIUS_KM * lat_rad
    ]).astype(float)

    # Build triangulation in ZIP-plane and find which triangle each ZIP centroid falls into
    tri_obj = mtri.Triangulation(zip_nodes_xy[:, 0], zip_nodes_xy[:, 1], tri_nodes)
    finder = tri_obj.get_trifinder()

    zip_tri_ids = finder(COORDS_KM[:, 0], COORDS_KM[:, 1]).astype(np.int64)  # -1 outside
    zip_inside_idx = np.where(zip_tri_ids >= 0)[0].astype(np.int64)

    return {
        "tri_nodes": tri_nodes,
        "areas_tri": areas_tri,
        "A_nodes": A_nodes,
        "zip_tri_ids": zip_tri_ids,
        "zip_inside_idx": zip_inside_idx,
        "epsg_project": epsg_project,
    }


def _get_mesh_nodal_geom_cache(mesh, epsg_project: int) -> dict:
    key = (id(mesh), int(epsg_project))
    if key not in _MESH_NODAL_GEOM_CACHE:
        _MESH_NODAL_GEOM_CACHE[key] = _build_mesh_nodal_geom_cache(mesh, epsg_project=epsg_project)
    return _MESH_NODAL_GEOM_CACHE[key]

# -------------------------------------------------------------------
# Batch density
# -------------------------------------------------------------------

def get_batch_nodal_density(
    mesh,
    years: Sequence[float],
    *,
    epsg_project: int = 5070,
    return_masses: bool = True,
    use_cache: bool = True,
) -> dict:
    """
    Compute mass-lumped nodal density rho_i(year) on the mesh for a batch of years.

    Definitions (mass-lumped P1):
        N_i(year) = sum_{T contains i} pop_T(year)/3
        A_i       = sum_{T contains i} area(T)/3
        rho_i(year) = N_i(year) / A_i   (persons / km^2)

    Population per triangle pop_T(year) is obtained by binning ZIP centroid populations
    (extrapolated to 'year') into triangles in the ZIP-plane.

    Returns dict with:
        years: (nyears,)
        rho_nodes: (nyears, N)
        N_nodes: (nyears, N)         [if return_masses]
        A_nodes: (N,)                [if return_masses]
        tri_nodes: (ntri,3)
    """
    yrs = np.asarray(list(years), dtype=float)
    if yrs.size == 0:
        raise ValueError("years must be a non-empty sequence of floats.")

    geom = _get_mesh_nodal_geom_cache(mesh, epsg_project=epsg_project)
    tri_nodes = geom["tri_nodes"]          # (ntri,3)
    A_nodes = geom["A_nodes"]              # (N,)
    zip_tri_ids = geom["zip_tri_ids"]      # (nzip,)
    zip_inside_idx = geom["zip_inside_idx"]
    ntri = tri_nodes.shape[0]
    N = mesh.p.shape[1]

    rho_nodes_all = np.zeros((yrs.size, N), dtype=float)
    N_nodes_all = np.zeros((yrs.size, N), dtype=float) if return_masses else None

    # Pre-slice for speed
    tri_ids_inside = zip_tri_ids[zip_inside_idx]

    for k, year in enumerate(yrs):
        year_f = float(year)

        # Try year-level cache
        if use_cache:
            key = (id(mesh), int(epsg_project), float(year_f))
            hit = _MESH_NODAL_YEAR_CACHE.get(key)
            if hit is not None:
                rho_nodes_all[k, :] = hit["rho_nodes"]
                if return_masses:
                    N_nodes_all[k, :] = hit["N_nodes"]
                continue

        # ZIP populations extrapolated to this year, for ALL ZIPs
        pops_t = _extrapolate_population_vector(POP2010, POP2020, year_f)

        # Bin people into triangles
        # pop_tri[t] = sum pops of ZIP centroids that fall into triangle t
        pop_tri = np.bincount(
            tri_ids_inside.astype(np.int64),
            weights=pops_t[zip_inside_idx],
            minlength=ntri
        ).astype(float)

        # Mass-lump to nodes: N_i = sum_{T contains i} pop_T/3
        N_nodes = np.zeros(N, dtype=float)
        add = pop_tri / 3.0
        np.add.at(N_nodes, tri_nodes[:, 0], add)
        np.add.at(N_nodes, tri_nodes[:, 1], add)
        np.add.at(N_nodes, tri_nodes[:, 2], add)

        # Density
        rho_nodes = N_nodes / np.maximum(A_nodes, 1e-30)
        rho_nodes[~np.isfinite(rho_nodes)] = 0.0
        rho_nodes = np.clip(rho_nodes, 0.0, None)

        rho_nodes_all[k, :] = rho_nodes
        if return_masses:
            N_nodes_all[k, :] = N_nodes

        if use_cache:
            _MESH_NODAL_YEAR_CACHE[(id(mesh), int(epsg_project), float(year_f))] = {
                "rho_nodes": rho_nodes,
                "N_nodes": N_nodes,
            }

    out = {
        "years": yrs,
        "rho_nodes": rho_nodes_all,
        "tri_nodes": tri_nodes,
    }
    if return_masses:
        out["N_nodes"] = N_nodes_all
        out["A_nodes"] = A_nodes
    return out