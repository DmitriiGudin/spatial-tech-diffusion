#!/usr/bin/env python3
"""
mesh_utils.py

Build 2D triangular meshes from state/province polygons.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from matplotlib.ticker import LogLocator, LogFormatterSciNotation

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from shapely.prepared import prep
from pyproj import Transformer

import time
import logging

import gmsh
from skfem import MeshTri
logging.getLogger("skfem").setLevel(logging.ERROR)

from contextlib import contextmanager

from fem_utils import load_mesh_km_from_msh
from density_utils import get_batch_nodal_density, lonlat_to_km

@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[TIMER] {label}: {dt:.3f} s")
        
_PROJECTORS: dict[int, Transformer] = {}
_INV_PROJECTORS: dict[int, Transformer] = {}

# ============================================================
# Config
# ============================================================

@dataclass
class MeshBuildConfig:
    h_km: float = 50.0
    simplify_km: float = 10.0

    # EPSG:5070 = NAD83 / Conus Albers (meters)
    epsg_project: int = 5070

    gmsh_algo_2d: int = 6  # 5 = Delaunay, 6 = Frontal-Delaunay
    msh_version: float = 4.1

    # Tolerances for ring cleanup
    ring_eps: float = 1e-12


# ============================================================
# Geometry helpers
# ============================================================

def pick_largest_polygon(geom) -> Polygon:
    """If MultiPolygon, keep largest by area."""
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        polys = [p for p in geom.geoms if (not p.is_empty) and p.area > 0]
        if not polys:
            raise ValueError("No valid polygons found in MultiPolygon.")
        return max(polys, key=lambda p: p.area)
    raise TypeError(f"Unsupported geometry type: {type(geom)}")


def simplify_polygon_km(poly_km: Polygon, tol_km: float) -> Polygon:
    """
    Simplify boundary in km to avoid tiny features => tiny triangles.
    preserve_topology=True, clean with buffer(0) if needed.
    """
    if tol_km > 0:
        out = poly_km.simplify(tol_km, preserve_topology=True)
    else:
        out = poly_km

    if not out.is_valid:
        out = out.buffer(0)

    if out.is_empty or (not isinstance(out, Polygon)):
        out = pick_largest_polygon(out)

    return out


def project_polygon_to_km(poly_lonlat: Polygon, epsg: int) -> Polygon:
    """
    EPSG:4326 (lon/lat degrees) -> EPSG:epsg (meters) -> scale to km.
    """
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    def _proj_coords(coords):
        xs_m, ys_m = transformer.transform(
            [c[0] for c in coords],
            [c[1] for c in coords],
        )
        xs_km = [x / 1000.0 for x in xs_m]
        ys_km = [y / 1000.0 for y in ys_m]
        return list(zip(xs_km, ys_km))

    exterior = _proj_coords(poly_lonlat.exterior.coords)
    interiors = [_proj_coords(ring.coords) for ring in poly_lonlat.interiors]
    return Polygon(exterior, interiors)


def _get_projector(epsg_project: int) -> Transformer:
    tr = _PROJECTORS.get(epsg_project)
    if tr is None:
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_project}", always_xy=True)
        _PROJECTORS[epsg_project] = tr
    return tr


def _get_inverse_projector(epsg_project: int) -> Transformer:
    tr = _INV_PROJECTORS.get(epsg_project)
    if tr is None:
        tr = Transformer.from_crs(f"EPSG:{epsg_project}", "EPSG:4326", always_xy=True)
        _INV_PROJECTORS[epsg_project] = tr
    return tr


def _project_lonlat_to_km(lon: np.ndarray, lat: np.ndarray, epsg_project: int) -> tuple[np.ndarray, np.ndarray]:
    tr = _get_projector(epsg_project)
    x_m, y_m = tr.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x_m, float) / 1000.0, np.asarray(y_m, float) / 1000.0


# ============================================================
# Admin-1 loading / selection
# ============================================================

def load_admin1_states(admin1_shp: Path) -> gpd.GeoDataFrame:
    """
    Load the Natural Earth Admin-1 (states/provinces) shapefile.
    Expect it to include US states with postal codes in a column.

    Common column names:
    - "iso_3166_2" like "US-CA"
    - "postal" or "postal_code" (sometimes)
    - "name" / "name_en" for state name

    We'll robustly derive a 2-letter state code where possible.
    """
    gdf = gpd.read_file(admin1_shp).to_crs("EPSG:4326")

    # Derive a 'state_code' column if possible.
    state_code = None

    if "iso_3166_2" in gdf.columns:
        # e.g. US-CA
        state_code = gdf["iso_3166_2"].astype(str).str.split("-").str[-1]
    elif "postal" in gdf.columns:
        state_code = gdf["postal"].astype(str)
    elif "postal_code" in gdf.columns:
        state_code = gdf["postal_code"].astype(str)

    if state_code is None:
        raise ValueError(
            "Could not find a usable state code column. "
            "Tried: iso_3166_2, postal, postal_code. "
            f"Available columns: {list(gdf.columns)}"
        )

    gdf = gdf.copy()
    gdf["state_code"] = state_code.str.upper().str.strip()

    # Keep only US states/territories that have 2-letter codes
    gdf = gdf[gdf["state_code"].str.fullmatch(r"[A-Z]{2}", na=False)]

    return gdf


def build_region_polygon_from_states(
    admin1_shp: Path,
    state_codes: Iterable[str],
) -> Polygon:
    """
    Union the listed states into a single polygon (lon/lat), keep largest component.
    """
    gdf = load_admin1_states(admin1_shp)
    codes = {c.upper().strip() for c in state_codes}
    sel = gdf[gdf["state_code"].isin(codes)]
    if len(sel) == 0:
        raise ValueError(f"No states found for codes={sorted(codes)}")

    geom = unary_union(sel.geometry.values)
    if geom.is_empty:
        raise ValueError("Union of selected states is empty.")

    return pick_largest_polygon(geom)


def build_conus_polygon(admin1_shp: Path) -> Polygon:
    """
    Build CONUS polygon from Admin-1 shapefile by taking all US state polygons
    except AK and HI, then union and keep largest component.
    """
    gdf = load_admin1_states(admin1_shp)

    # Exclude AK/HI
    gdf = gdf[~gdf["state_code"].isin(["AK", "HI"])]

    geom = unary_union(gdf.geometry.values)
    if geom.is_empty:
        raise ValueError("CONUS union is empty.")

    return pick_largest_polygon(geom)


# ============================================================
# Gmsh meshing (kilometers)
# ============================================================

def _clean_ring(coords, eps: float):
    """
    coords: iterable of (x,y)
    Returns list of (x,y) with:
      - last point removed if it equals first
      - consecutive duplicates removed
      - raises if fewer than 3 points remain
    """
    coords = list(coords)
    if len(coords) < 3:
        raise ValueError("Ring has <3 points.")

    # Drop repeated last point if closed
    if (abs(coords[0][0] - coords[-1][0]) < eps) and (abs(coords[0][1] - coords[-1][1]) < eps):
        coords = coords[:-1]

    cleaned = [coords[0]]
    for x, y in coords[1:]:
        x0, y0 = cleaned[-1]
        if (abs(x - x0) >= eps) or (abs(y - y0) >= eps):
            cleaned.append((x, y))

    if len(cleaned) < 3:
        raise ValueError("Ring collapsed after cleaning (too many duplicates).")

    return cleaned


def _add_polygon_to_gmsh(model, poly_km: Polygon, h_km: float, ring_eps: float):
    """
    Add polygon (with holes) to gmsh.model.geo in kilometers. Return surface tag.
    """
    geo = model.geo

    def add_loop_from_coords(coords, close_explicitly=True):
        coords = _clean_ring(coords, eps=ring_eps)

        pts = [geo.addPoint(float(x), float(y), 0.0, float(h_km)) for (x, y) in coords]

        lines = []
        for i in range(len(pts) - 1):
            lines.append(geo.addLine(pts[i], pts[i + 1]))
        if close_explicitly:
            lines.append(geo.addLine(pts[-1], pts[0]))

        return geo.addCurveLoop(lines)

    # Exterior
    outer_loop = add_loop_from_coords(poly_km.exterior.coords)

    # Holes: reverse orientation to be safe
    hole_loops = []
    for ring in poly_km.interiors:
        hole_loops.append(add_loop_from_coords(list(ring.coords)[::-1]))

    surf = geo.addPlaneSurface([outer_loop] + hole_loops)
    return surf


def build_mesh_from_polygon_km(
    poly_km: Polygon,
    out_msh: Path,
    cfg: MeshBuildConfig,
    verbose: bool = True,
    model_name: str = "mesh_km",
) -> None:
    """
    Mesh a planar polygon in kilometers with Gmsh and write .msh.
    """
    out_msh.parent.mkdir(parents=True, exist_ok=True)
    h = float(cfg.h_km)

    with timed(f"build_mesh_from_polygon_km -> {out_msh.name} (h={h:g} km)"):
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
            gmsh.model.add(model_name)
    
            surf = _add_polygon_to_gmsh(gmsh.model, poly_km, h, ring_eps=cfg.ring_eps)
            gmsh.model.geo.removeAllDuplicates()
            gmsh.model.geo.synchronize()
    
            # Physical group for downstream FEM IO
            gmsh.model.addPhysicalGroup(2, [surf], tag=1)
            gmsh.model.setPhysicalName(2, 1, "domain")
    
            # Background constant size field: parameter name differs by gmsh builds.
            # Most builds accept "VIn" for Constant.
            try:
                fid = gmsh.model.mesh.field.add("Constant")
                gmsh.model.mesh.field.setNumber(fid, "VIn", h)
                gmsh.model.mesh.field.setAsBackgroundMesh(fid)
            except Exception:
                fid = gmsh.model.mesh.field.add("MathEval")
                gmsh.model.mesh.field.setString(fid, "F", str(h))
                gmsh.model.mesh.field.setAsBackgroundMesh(fid)
    
            # Enforce at geometric points too
            point_entities = gmsh.model.getEntities(0)
            if point_entities:
                gmsh.model.mesh.setSize(point_entities, h)
    
            gmsh.option.setNumber("Mesh.Algorithm", cfg.gmsh_algo_2d)
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.8*h)
            gmsh.option.setNumber("Mesh.MeshSizeMax", 0.8*h)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            
            # Additional options improving quality of triangles
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
            gmsh.option.setNumber("Mesh.Smoothing", 10)  # optional; 0–20 typical
    
            gmsh.model.mesh.generate(2)
            
            # Post-ops (could be useful):
            try:
                gmsh.model.mesh.optimize("Netgen")
            except Exception:
                pass
            try:
                gmsh.model.mesh.optimize("Laplace2D")
            except Exception:
                pass
    
            gmsh.option.setNumber("Mesh.MshFileVersion", cfg.msh_version)
            gmsh.write(str(out_msh))
        finally:
            gmsh.finalize()


# ============================================================
# End-to-end convenience builders
# ============================================================

def build_mesh_from_admin1_region(
    admin1_shp: Path,
    state_codes: Iterable[str],
    out_msh: Path,
    cfg: MeshBuildConfig,
    verbose: bool = True,
    model_name: str = "region_mesh_km",
) -> None:
    """
    Build a mesh for a region defined by a list of 2-letter state codes.
    """
    poly_lonlat = build_region_polygon_from_states(admin1_shp, state_codes)
    poly_km = project_polygon_to_km(poly_lonlat, cfg.epsg_project)
    poly_km = simplify_polygon_km(poly_km, cfg.simplify_km)
    build_mesh_from_polygon_km(poly_km, out_msh, cfg, verbose=verbose, model_name=model_name)


# ============================================================
# Plotting (mesh -> lon/lat)
# ============================================================

import meshio
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm


def plot_msh_triangles_lonlat(msh_path: Path, out_png: Path, epsg_project: int = 5070):
    """
    Read .msh whose coordinates are in km (EPSG:5070 space scaled to km),
    convert km->m, inverse-project to lon/lat, and plot triangles.
    """
    mesh = meshio.read(msh_path)

    tri_cells = None
    for c in mesh.cells:
        if c.type == "triangle":
            tri_cells = c.data
            break
    if tri_cells is None:
        raise ValueError("No triangle cells found in mesh.")

    pts_xy_km = mesh.points[:, :2]
    pts_xy_m = pts_xy_km * 1000.0  # inverse projection expects meters

    inv = _get_inverse_projector(epsg_project)
    lon, lat = inv.transform(pts_xy_m[:, 0], pts_xy_m[:, 1])

    tri = mtri.Triangulation(lon, lat, tri_cells)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.triplot(tri, linewidth=0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Mesh: {msh_path.name} (lon/lat)")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved mesh plot to {out_png}")


# ============================================================
# Population and mesh diagnostics
# ============================================================

def print_mesh_quality_diagnostics(mesh: MeshTri, *, label: str = "mesh") -> None:
    """
    Print:
      - #angles > 90° (counted across ALL triangle angles), plus ratio
      - side-length percentiles across all triangle sides (km)
      - triangle-area percentiles across all triangles (km^2)
    Assumes mesh coordinates are in km.
    """
    tri = mesh.t  # (3, ntri) node indices
    p = mesh.p    # (2, N) node coords in km

    ntri = tri.shape[1]
    if ntri == 0:
        print(f"[MESH DIAG] {label}: no triangles.")
        return

    # Triangle vertex coordinates
    x1, y1 = p[0, tri[0]], p[1, tri[0]]
    x2, y2 = p[0, tri[1]], p[1, tri[1]]
    x3, y3 = p[0, tri[2]], p[1, tri[2]]

    # Side lengths (km)
    # a = |v2-v3|, b = |v1-v3|, c = |v1-v2|
    a = np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
    b = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
    c = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Angles via law of cosines (radians)
    eps = 1e-15
    denomA = np.maximum(2.0 * b * c, eps)
    denomB = np.maximum(2.0 * a * c, eps)
    denomC = np.maximum(2.0 * a * b, eps)

    cosA = (b * b + c * c - a * a) / denomA
    cosB = (a * a + c * c - b * b) / denomB
    cosC = (a * a + b * b - c * c) / denomC

    cosA = np.clip(cosA, -1.0, 1.0)
    cosB = np.clip(cosB, -1.0, 1.0)
    cosC = np.clip(cosC, -1.0, 1.0)

    Aang = np.arccos(cosA)
    Bang = np.arccos(cosB)
    Cang = np.arccos(cosC)

    # Count obtuse angles: > 90° = > pi/2
    obtuse_mask = (Aang > 0.5 * np.pi) | (Bang > 0.5 * np.pi) | (Cang > 0.5 * np.pi)
    n_obtuse_tri = int(np.count_nonzero(obtuse_mask))
    ratio_obtuse_tri = n_obtuse_tri / max(ntri, 1)

    # Side-length percentiles over ALL sides of ALL triangles
    sides_all = np.concatenate([a, b, c]).astype(float)

    # Area percentiles over triangles
    areas = triangle_areas_km2_from_skfem(mesh).astype(float)

    q = [0, 5, 25, 50, 75, 95, 100]
    side_q = np.percentile(sides_all, q)
    area_q = np.percentile(areas, q)

    print("\n" + "=" * 90)
    print(f"[MESH DIAG] {label}")
    print(f"[MESH DIAG] obtuse triangles (>90° angle): {n_obtuse_tri} / {ntri}  (ratio={ratio_obtuse_tri:.6f})")

    print("[MESH DIAG] side lengths (km) percentiles:")
    print("  " + ", ".join([f"{qq:>3d}%={val:.6g}" for qq, val in zip(q, side_q)]))

    print("[MESH DIAG] triangle areas (km^2) percentiles:")
    print("  " + ", ".join([f"{qq:>3d}%={val:.6g}" for qq, val in zip(q, area_q)]))
    print("=" * 90 + "\n")


def polygon_lonlat_to_zipkm(poly_lonlat: Polygon):
    """
    Map a lon/lat polygon into the SAME (x,y) km plane used by density_utils KDTree.
    """
    from density_utils import lonlat_to_km as zip_lonlat_to_km

    def map_ring(ring_coords):
        pts = [zip_lonlat_to_km(lon, lat) for (lon, lat) in ring_coords]
        return [(float(p[0]), float(p[1])) for p in pts]

    ext = map_ring(poly_lonlat.exterior.coords)
    holes = [map_ring(r.coords) for r in poly_lonlat.interiors]
    return Polygon(ext, holes)


def true_population_inside_region(poly_lonlat: Polygon, year: float) -> float:
    """
    "True" population inside a region polygon = sum of ZIP centroid populations (extrapolated to year)
    whose centroid point lies inside the region polygon, in the KDTree coordinate plane.
    """
    from density_utils import COORDS_KM, POP2010, POP2020, _extrapolate_population_vector

    poly_zipkm = polygon_lonlat_to_zipkm(poly_lonlat)
    ppoly = prep(poly_zipkm)

    pops_t = _extrapolate_population_vector(POP2010, POP2020, year)  # (N,)
    inside = np.array([ppoly.contains(Point(float(x), float(y))) for (x, y) in COORDS_KM], dtype=bool)
    return float(np.sum(pops_t[inside]))


def triangle_areas_km2_from_skfem(mesh) -> np.ndarray:
    """
    Compute areas for each triangle in km^2 (mesh coordinates are km).
    mesh: skfem.MeshTri
    """
    tri = mesh.t  # (3, ntri)
    p = mesh.p    # (2, N)
    x1, y1 = p[0, tri[0]], p[1, tri[0]]
    x2, y2 = p[0, tri[1]], p[1, tri[1]]
    x3, y3 = p[0, tri[2]], p[1, tri[2]]
    return 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def estimated_population_integral_on_mesh(
    msh_path: Path,
    year: float,
    epsg_project: int = 5070,
    *,
    mesh: Optional[MeshTri] = None,
) -> float:
    """
    Mesh-only mode:
        pop_est(year) = sum_i rho_i(year) * A_i
    """
    from fem_utils import load_mesh_km_from_msh

    if mesh is None:
        mesh = load_mesh_km_from_msh(msh_path)

    out = get_batch_nodal_density(mesh, [year], epsg_project=epsg_project,
                                  return_masses=True, use_cache=True)
    rho_nodes = out["rho_nodes"][0]
    A_nodes = out["A_nodes"]
    return float(np.dot(rho_nodes, A_nodes))


def print_population_mass_check(
    admin1_shp: Path,
    state_codes: Iterable[str],
    msh_path: Path,
    year: float = 2023.0,
    epsg_project: int = 5070,
    *,
    mesh: Optional[MeshTri] = None,
) -> None:
    with timed("print_population_mass_check"):
        poly_lonlat = build_region_polygon_from_states(admin1_shp, state_codes)
        true_pop = true_population_inside_region(poly_lonlat, year=year)

        est_pop = estimated_population_integral_on_mesh(
            msh_path=msh_path, year=year, epsg_project=epsg_project, mesh=mesh
        )

        denom = max(true_pop, 1.0)
        print(f"[POP CHECK] mode=MESH-NODAL-MASSLUMP year={year:.0f} "
              f"region={','.join(list(state_codes))} mesh={Path(msh_path).name}")
        print(f"[POP CHECK] true ZIP-sum inside polygon: {true_pop:,.0f}")
        print(f"[POP CHECK] estimated population on mesh: {est_pop:,.0f}")
        print(f"[POP CHECK] ratio est/true = {est_pop / denom:.6f}")
    
    
def _safe_element_finder(mesh: MeshTri, chunk_size: int = 100_000):
    """
    Chunked wrapper around skfem's element_finder for large point clouds.
    Returns tri_ids (len=npoints) with -1 outside.
    """
    finder = mesh.element_finder()

    xmin, xmax = float(mesh.p[0].min()), float(mesh.p[0].max())
    ymin, ymax = float(mesh.p[1].min()), float(mesh.p[1].max())
    pad = 1e-9

    def _find(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        out = -np.ones(x.shape[0], dtype=np.int64)

        bbox = (x >= xmin - pad) & (x <= xmax + pad) & (y >= ymin - pad) & (y <= ymax + pad)
        idx_all = np.where(bbox)[0]
        if idx_all.size == 0:
            return out

        CH = int(chunk_size)
        for s in range(0, idx_all.size, CH):
            idx = idx_all[s:s+CH]
            try:
                tri = finder(x[idx], y[idx])
                out[idx] = tri.astype(np.int64)
            except ValueError:
                # Rare: if skfem throws due to some outside points in the chunk
                for ii in idx:
                    try:
                        tri1 = finder(np.array([x[ii]]), np.array([y[ii]]))
                        out[ii] = int(tri1[0])
                    except ValueError:
                        out[ii] = -1
        return out

    return _find


def true_population_per_triangle_fast(
    tri_nodes_zipxy: np.ndarray,   # (ntri,3) vertex indices
    zip_nodes_xy: np.ndarray,      # (Nnodes,2) node coords in ZIP-plane
    zip_xy_in: np.ndarray,         # (Nzip_in,2) ZIP centroids in ZIP-plane
    pops_in: np.ndarray,           # (Nzip_in,) populations
) -> np.ndarray:
    # Build triangulation in ZIP-plane
    tri = mtri.Triangulation(
        zip_nodes_xy[:, 0],
        zip_nodes_xy[:, 1],
        tri_nodes_zipxy
    )

    # Fast point -> triangle lookup
    finder = tri.get_trifinder()
    tri_ids = finder(zip_xy_in[:, 0], zip_xy_in[:, 1])  # -1 outside

    inside = tri_ids >= 0
    ntri = tri_nodes_zipxy.shape[0]

    pop_true_tri = np.bincount(
        tri_ids[inside].astype(np.int64),
        weights=pops_in[inside],
        minlength=ntri
    ).astype(float)

    print("[DEBUG] trifinder assigned ZIP points:", int(np.count_nonzero(inside)), "/", int(zip_xy_in.shape[0]))
    return pop_true_tri


def plot_triangle_population_comparison(
    admin1_shp: Path,
    state_codes: Iterable[str],
    msh_path: Path,
    out_png: Path,
    year: float = 2023.0,
    epsg_project: int = 5070,
    vmax_quantile: float = 0.99,
    *,
    mesh: Optional[MeshTri] = None,
    h_km: Optional[float] = None,   # <-- pass cfg.h_km from caller
) -> None:
    """
    Two-panel plot:
      Left : true population per triangle (ZIP centroid sum inside triangle)
      Right: estimated population per node (mass-lumped) shown as circles centered at nodes

    Circle radius in km: r_km = sqrt(3)/4 * h_km (so “comparable” to triangles of size ~h_km).
    """
    with timed("plot_triangle_population_comparison"):
        from fem_utils import load_mesh_km_from_msh
        from density_utils import COORDS_KM, POP2010, POP2020, _extrapolate_population_vector
        from matplotlib.patches import Ellipse
        from matplotlib.collections import PatchCollection

        # ---- load mesh ----
        if mesh is None:
            mesh = load_mesh_km_from_msh(msh_path)

        tri_nodes = mesh.t.T  # (ntri, 3)
        pts_xy_km = mesh.p.T  # (N,2) in km (EPSG:5070)

        # ---- lon/lat for plotting ----
        inv = _get_inverse_projector(epsg_project)
        lon, lat = inv.transform(pts_xy_km[:, 0] * 1000.0, pts_xy_km[:, 1] * 1000.0)
        lon = np.asarray(lon, float)
        lat = np.asarray(lat, float)

        # ---- estimated population per NODE (mass-lumped) ----
        out = get_batch_nodal_density(
            mesh, [year],
            epsg_project=epsg_project,
            return_masses=True,
            use_cache=True
        )
        dens_nodes = out["rho_nodes"][0]  # persons / km^2
        A_nodes = out["A_nodes"]          # km^2
        pop_est_node = dens_nodes * A_nodes  # persons per node “cell”

        # ---- true population per TRIANGLE ----
        pops_t = _extrapolate_population_vector(POP2010, POP2020, year)
        zip_xy = COORDS_KM

        poly_lonlat = build_region_polygon_from_states(admin1_shp, state_codes)
        poly_zipkm = polygon_lonlat_to_zipkm(poly_lonlat)
        ppoly_zipkm = prep(poly_zipkm)

        zip_inside_region = np.array(
            [ppoly_zipkm.contains(Point(float(x), float(y))) for (x, y) in zip_xy],
            dtype=bool
        )
        zip_xy_in = zip_xy[zip_inside_region]
        pops_in = pops_t[zip_inside_region]

        # Map mesh nodes to ZIP-plane
        zip_nodes_xy = np.array([lonlat_to_km(lon[i], lat[i]) for i in range(len(lon))], dtype=float)

        pop_true_tri = true_population_per_triangle_fast(
            tri_nodes_zipxy=tri_nodes,
            zip_nodes_xy=zip_nodes_xy,
            zip_xy_in=zip_xy_in,
            pops_in=pops_in,
        )

        print(f"[true-tri] triangles nonempty: {np.count_nonzero(pop_true_tri > 0)} / {pop_true_tri.size}")
        print(f"[true-tri] total pop from triangle binning: {pop_true_tri.sum():,.0f}")
        print(f"[true-tri] total pop from region-filtered ZIPs: {pops_in.sum():,.0f}")
        print(f"[true-tri] ratio(binned / region ZIP sum): {pop_true_tri.sum() / max(pops_in.sum(), 1.0):.6f}")

        print(f"[est-node] total pop from nodal masses: {pop_est_node.sum():,.0f}")

        # ---- plotting primitives ----
        triang = mtri.Triangulation(lon, lat, tri_nodes)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        # shared log scale across (triangle pops) and (node pops)
        eps = 1.0
        true_pos = pop_true_tri[pop_true_tri > 0]
        node_pos = pop_est_node[pop_est_node > 0]

        if true_pos.size == 0 and node_pos.size == 0:
            vmin = eps
            vmax = eps * 10.0
        else:
            vmax = float(np.quantile(np.concatenate([true_pos, node_pos]), vmax_quantile))
            vmax = max(vmax, eps)
            vmin = eps

        norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

        # ---- LEFT: triangle choropleth ----
        tpc0 = axes[0].tripcolor(
            triang,
            np.maximum(pop_true_tri, eps),
            shading="flat",
            norm=norm
        )
        axes[0].set_title(f"True population / triangle (ZIP centroid sum), year={year:.0f}")
        axes[0].set_xlabel("Longitude (deg)")
        axes[0].set_ylabel("Latitude (deg)")

        # ---- RIGHT: node circles ----
        # Require h_km to set comparable radius
        if h_km is None:
            raise ValueError("plot_triangle_population_comparison: please pass h_km (e.g. cfg.h_km).")

        r_km = (np.sqrt(3) / 4) * float(h_km)

        # Convert km radius to deg radius (local anisotropic conversion)
        KM_PER_DEG_LAT = 111.32

        dy_deg = np.full_like(lat, r_km / KM_PER_DEG_LAT, dtype=float)  # <-- vector now
        dx_deg = r_km / (KM_PER_DEG_LAT * np.clip(np.cos(np.deg2rad(lat)), 1e-6, None))

        patches = []
        values = np.maximum(pop_est_node, eps)

        for x, y, wx, hy in zip(lon, lat, 2.0 * dx_deg, 2.0 * dy_deg):
            patches.append(Ellipse((float(x), float(y)), width=float(wx), height=float(hy)))

        pc = PatchCollection(patches, array=values, norm=norm, linewidths=0.0)
        axes[1].add_collection(pc)
        axes[1].autoscale_view()

        axes[1].set_title(f"Estimated population / node (mass-lumped), year={year:.0f}\n")
        axes[1].set_xlabel("Longitude (deg)")
        axes[1].set_ylabel("Latitude (deg)")

        # make map aspect sane
        for ax in axes:
            ax.set_aspect("equal", adjustable="box")

        # shared colorbar (use left artist)
        cbar = fig.colorbar(tpc0, ax=axes, location="right", fraction=0.035, pad=0.02)
        cbar.set_label("Population (persons, log scale)")

        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"Saved population comparison plot to {out_png}")
        print("True total (tri):", float(pop_true_tri.sum()))
        print("Est  total (node):", float(pop_est_node.sum()))
        print("Ratio est/true:", float(pop_est_node.sum() / max(pop_true_tri.sum(), 1.0)))


def _events_inside_mesh_mask(
    mesh: MeshTri,
    lon: np.ndarray,
    lat: np.ndarray,
    epsg_project: int,
    chunk_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      inside_mask: (K,) bool
      tri_ids:     (K,) int64   (-1 outside)
      year_int:    (K,) int32   calendar year extracted from dates elsewhere if you want
    """
    x_km, y_km = _project_lonlat_to_km(lon, lat, epsg_project=epsg_project)
    finder = _safe_element_finder(mesh, chunk_size=chunk_size)
    tri_ids = finder(x_km, y_km).astype(np.int64)
    inside = tri_ids >= 0
    return inside, tri_ids, x_km, y_km


def fetch_cpi_monthly_fred(
    start: str,
    end: str,
    *,
    cache_csv: Optional[Path] = None,
) -> pd.Series:
    """
    Fetch monthly CPI (CPIAUCSL) from FRED via pandas_datareader.

    Returns a pandas Series indexed by month start timestamps (freq ~ MS).
    If cache_csv is provided and exists, uses it unless it doesn't cover [start,end].
    """
    start_dt = pd.to_datetime(start).to_period("M").to_timestamp()
    end_dt = pd.to_datetime(end).to_period("M").to_timestamp()

    # Try cache first
    if cache_csv is not None and cache_csv.exists():
        try:
            dfc = pd.read_csv(cache_csv, parse_dates=["date"])
            dfc = dfc.dropna(subset=["date", "CPIAUCSL"]).copy()
            dfc = dfc.sort_values("date")
            s = pd.Series(dfc["CPIAUCSL"].to_numpy(float), index=dfc["date"])
            s = s[~s.index.duplicated(keep="last")]
            if (s.index.min() <= start_dt) and (s.index.max() >= end_dt):
                return s.loc[start_dt:end_dt]
        except Exception:
            pass  # fall through to re-download

    # Download from FRED
    try:
        import pandas_datareader.data as web  # type: ignore
    except Exception as e:
        raise ImportError(
            "pandas_datareader is required for CPI download. "
            "Install with: pip install pandas-datareader"
        ) from e

    df = web.DataReader("CPIAUCSL", "fred", start_dt, end_dt)
    if df.empty:
        raise RuntimeError("CPI download returned empty dataframe.")

    s = df["CPIAUCSL"].copy()
    # Normalize index to month-start timestamps
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()

    if cache_csv is not None:
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        out = pd.DataFrame({"date": s.index, "CPIAUCSL": s.to_numpy(float)})
        out.to_csv(cache_csv, index=False)

    return s


def choose_cpi_base(
    cpi: pd.Series,
    *,
    base_year: int = 2026,
    base_month: Optional[int] = None,
) -> tuple[pd.Timestamp, float]:
    """
    Choose CPI base date/value.
    Preference: latest month in base_year (optionally a specific base_month).
    Fallback: latest available CPI month overall.
    """
    cpi = cpi.dropna().copy()
    if cpi.empty:
        raise ValueError("CPI series is empty after dropping NaNs.")

    idx = pd.to_datetime(cpi.index).to_period("M").to_timestamp()
    cpi.index = idx

    if base_month is not None:
        target = pd.Timestamp(year=base_year, month=int(base_month), day=1)
        if target in cpi.index:
            return target, float(cpi.loc[target])
        # if not present, fall back (don’t silently interpolate)
        print(f"[CPI] WARNING: CPI for {base_year:04d}-{base_month:02d} not available; falling back to latest.")

    in_year = cpi[cpi.index.year == int(base_year)]
    if not in_year.empty:
        dt = in_year.index.max()
        return dt, float(in_year.loc[dt])

    dt = cpi.index.max()
    print(f"[CPI] WARNING: no CPI data found for year {base_year}; using latest available {dt.strftime('%Y-%m')}.")
    return dt, float(cpi.loc[dt])


def plot_adoptions_vs_costs(
    events_df: pd.DataFrame,
    out_png: Path,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    price_col: str = "price",
    date_col: str = "date",
    missing_price_value: float = -1.0,
    base_year: int = 2026,
    base_month: Optional[int] = None,
    cpi_cache_csv: Optional[Path] = None,
) -> dict:
    """
    Three-panel plot:
      1) monthly adoption counts
      2) monthly price distribution summary (p10 / median / p90), CPI-adjusted to base
      3) monthly count of valid price observations (finite and > 0)

    Notes:
      - Prices <= 0 or == missing_price_value are excluded from price stats and count panel.
      - Inflation adjustment is done monthly via CPI (FRED) and applied to the quantiles/median (fast).
    """
    with timed("plot_adoptions_vs_costs"):
        df = events_df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).copy()
        if df.empty:
            raise ValueError("No valid events after parsing dates.")

        if start_date is None:
            start_date = df[date_col].min().strftime("%Y-%m-%d")
        if end_date is None:
            end_date = df[date_col].max().strftime("%Y-%m-%d")

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)].copy()
        if df.empty:
            raise ValueError("No events in requested date window.")

        # Monthly bins (month start)
        m0 = start_dt.to_period("M").to_timestamp()
        m1 = end_dt.to_period("M").to_timestamp()
        months = pd.date_range(m0, m1, freq="MS")

        # --- Monthly adoption counts (all events) ---
        ev_month = df[date_col].dt.to_period("M").dt.to_timestamp()
        idx = ((ev_month.dt.year - m0.year) * 12 + (ev_month.dt.month - m0.month)).to_numpy(np.int64)
        counts = np.bincount(idx, minlength=len(months)).astype(float)

        # --- Price parsing + validity mask (for price stats only) ---
        if price_col not in df.columns:
            raise ValueError(f"Column '{price_col}' not found in events_df.")

        prices = pd.to_numeric(df[price_col], errors="coerce").to_numpy(float)

        # "valid available prices" = finite and positive
        # also exclude explicit sentinel if you keep one (e.g. -1)
        ok_price = np.isfinite(prices) & (prices > 0.0) & (prices != float(missing_price_value))

        df_price = df.loc[ok_price, [date_col]].copy()
        df_price["price"] = prices[ok_price]
        df_price["month"] = df_price[date_col].dt.to_period("M").dt.to_timestamp()

        # --- Monthly p10/median/p90 + observation counts (raw dollars) ---
        # (months are not many; groupby+quantile is fine)
        if df_price.empty:
            p10_raw = np.full(len(months), np.nan, dtype=float)
            p50_raw = np.full(len(months), np.nan, dtype=float)
            p90_raw = np.full(len(months), np.nan, dtype=float)
            n_price = np.zeros(len(months), dtype=float)
        else:
            g = df_price.groupby("month")["price"]
            q = g.quantile([0.10, 0.50, 0.90]).unstack()
            q = q.reindex(months)  # align to full month index

            p10_raw = q.get(0.10, pd.Series(index=months, dtype=float)).to_numpy(float)
            p50_raw = q.get(0.50, pd.Series(index=months, dtype=float)).to_numpy(float)
            p90_raw = q.get(0.90, pd.Series(index=months, dtype=float)).to_numpy(float)

            n_price = g.size().reindex(months).fillna(0.0).to_numpy(float)

        # --- CPI monthly series over same month window ---
        cpi_start = months.min().strftime("%Y-%m-%d")
        cpi_end = months.max().strftime("%Y-%m-%d")
        cpi = fetch_cpi_monthly_fred(cpi_start, cpi_end, cache_csv=cpi_cache_csv)

        cpi = cpi.reindex(pd.to_datetime(months).to_period("M").to_timestamp())
        if cpi.isna().any():
            cpi = cpi.ffill()

        base_dt, cpi_base = choose_cpi_base(cpi, base_year=base_year, base_month=base_month)
        cpi_vals = cpi.to_numpy(float)
        scale = float(cpi_base) / np.maximum(cpi_vals, 1e-12)

        # Apply CPI adjustment to the distribution summaries
        p10_adj = p10_raw * scale
        p50_adj = p50_raw * scale
        p90_adj = p90_raw * scale

        # ---- Plot ----
        out_png.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(
            3, 1, figsize=(12, 10), sharex=True, constrained_layout=True,
            gridspec_kw=dict(height_ratios=[2.0, 2.0, 1.2])
        )

        # 1) Adoptions
        axes[0].plot(months, counts, marker="o", linestyle="-", linewidth=1.5, markersize=3)
        axes[0].set_ylabel("Adoptions / month")
        axes[0].grid(True, alpha=0.25)
        axes[0].set_title("Monthly adoptions and CPI-adjusted price distribution summaries")

        # 2) Prices: p10 / median / p90
        axes[1].plot(months, p10_adj, marker="o", linestyle="-", linewidth=1.5, markersize=3, label="p10")
        axes[1].plot(months, p50_adj, marker="o", linestyle="-", linewidth=1.5, markersize=3, label="median")
        axes[1].plot(months, p90_adj, marker="o", linestyle="-", linewidth=1.5, markersize=3, label="p90")
        axes[1].set_ylabel(f"Cost (CPI-adjusted to {base_year} base)")
        axes[1].set_ylim(bottom=0.0)
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="best")

        axes[1].text(
            0.01, 0.95,
            f"CPI base used: {base_dt.strftime('%Y-%m')} (CPI={float(cpi_base):.3f})",
            transform=axes[1].transAxes,
            va="top"
        )

        # 3) Price observation counts
        axes[2].bar(months, n_price, width=25)  # width in days-ish; OK for monthly bars
        axes[2].set_ylabel("# price obs")
        axes[2].grid(True, alpha=0.25, axis="y")
        axes[2].set_xlabel("Month")

        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"Saved adoptions vs costs plot to {out_png}")

        return dict(
            start_date=str(start_date),
            end_date=str(end_date),
            n_months=int(len(months)),
            n_events_total=int(len(df)),
            n_events_with_price=int(np.count_nonzero(ok_price)),
            n_months_with_price=int(np.count_nonzero(n_price > 0)),
            cpi_base_month=str(base_dt.strftime("%Y-%m")),
            cpi_base_value=float(cpi_base),
            base_year=int(base_year),
            base_month=int(base_month) if base_month is not None else None,
        )
    
    
def plot_mesh_costs(
    msh_path: Path,
    events_df: pd.DataFrame,
    out_png: Path,
    *,
    year: int,
    h_km: float,                    # REQUIRED for circle radius, like other plots
    epsg_project: int = 5070,
    price_col: str = "price",
    date_col: str = "date",
    missing_price_value: float = -1.0,
    chunk_size: int = 50_000,
    # CPI adjustment (optional)
    cpi_adjust: bool = True,
    base_year: int = 2026,
    base_month: Optional[int] = None,
    cpi_cache_csv: Optional[Path] = None,
) -> dict:
    """
    Node-biased plot: circles centered at nodes, colored by per-node median cost.

    - Uses only strictly positive prices (finite and > 0, and != missing_price_value).
    - Uses only events inside the mesh.
    - Per-node median built by assigning each event price to the 3 triangle vertices.
    - Nodes with no samples get the statewide median (median across all positive observations inside mesh).
    - Colorbar is linear, vmin=0, vmax=max(node_median).
    - If there are no positive price observations inside mesh for that year -> no plot; print warning.
    """
    with timed("plot_mesh_costs"):
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.patches import Ellipse
        from matplotlib.collections import PatchCollection

        mesh = load_mesh_km_from_msh(msh_path)
        N = mesh.p.shape[1]
        tri = mesh.t  # (3, ntri)

        df = events_df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col, "longitude", "latitude"]).copy()
        if df.empty:
            raise ValueError("No valid events after parsing date/coords.")

        # year filter
        df = df[df[date_col].dt.year.astype(int) == int(year)].copy()
        if df.empty:
            print(f"[mesh_costs] WARNING: no events in year={year}. Skipping plot.")
            return dict(year=int(year), made_plot=False, n_price_obs_inside=0)

        # parse prices and enforce STRICT positivity
        if price_col not in df.columns:
            raise ValueError(f"Column '{price_col}' not found in events_df.")
        prices_raw = pd.to_numeric(df[price_col], errors="coerce").to_numpy(float)

        ok_price = (
            np.isfinite(prices_raw) &
            (prices_raw > 0.0) &
            (prices_raw != float(missing_price_value))
        )
        df = df.loc[ok_price].copy()
        if df.empty:
            print(f"[mesh_costs] WARNING: no positive valid prices in year={year}. Skipping plot.")
            return dict(year=int(year), made_plot=False, n_price_obs_inside=0)

        # inside mesh
        lon = df["longitude"].to_numpy(float)
        lat = df["latitude"].to_numpy(float)
        inside, tri_ids, _, _ = _events_inside_mesh_mask(mesh, lon, lat, epsg_project, chunk_size=chunk_size)
        df = df.loc[inside].copy()
        tri_ids = tri_ids[inside].astype(np.int64)

        if df.empty or tri_ids.size == 0:
            print(f"[mesh_costs] WARNING: no positive priced events inside mesh in year={year}. Skipping plot.")
            return dict(year=int(year), made_plot=False, n_price_obs_inside=0)

        prices = pd.to_numeric(df[price_col], errors="coerce").to_numpy(float)

        # Optional CPI adjustment PER EVENT (month-specific), then keep strictly positive again
        if cpi_adjust:
            m0 = df[date_col].min().to_period("M").to_timestamp()
            m1 = df[date_col].max().to_period("M").to_timestamp()
            months = pd.date_range(m0, m1, freq="MS")

            cpi = fetch_cpi_monthly_fred(
                months.min().strftime("%Y-%m-%d"),
                months.max().strftime("%Y-%m-%d"),
                cache_csv=cpi_cache_csv
            )
            cpi = cpi.reindex(pd.to_datetime(months).to_period("M").to_timestamp())
            if cpi.isna().any():
                cpi = cpi.ffill()

            base_dt, cpi_base = choose_cpi_base(cpi, base_year=base_year, base_month=base_month)

            ev_month = df[date_col].dt.to_period("M").dt.to_timestamp()
            cpi_ev = cpi.reindex(ev_month).to_numpy(float)

            scale = float(cpi_base) / np.maximum(cpi_ev, 1e-12)
            prices = prices * scale

            # enforce positivity post-transform (paranoia)
            pos = np.isfinite(prices) & (prices > 0.0)
            prices = prices[pos]
            tri_ids = tri_ids[pos]
            df = df.iloc[np.where(pos)[0]].copy()
            
        # ------------------------------------------------------------------
        # Trim extreme prices within the year to suppress broken 1-2 sample nodes
        # Drop bottom 20% and top 20% of (positive) prices for this year.
        # ------------------------------------------------------------------
        trim_q = 0.20
        if prices.size >= 10:  # avoid doing something silly on tiny samples
            lo = float(np.quantile(prices, trim_q))
            hi = float(np.quantile(prices, 1.0 - trim_q))

            # Guard against degenerate quantiles (all prices equal, etc.)
            if np.isfinite(lo) and np.isfinite(hi) and (hi > lo):
                keep = (prices >= lo) & (prices <= hi) & np.isfinite(prices) & (prices > 0.0)

                n_before = int(prices.size)
                prices = prices[keep]
                tri_ids = tri_ids[keep]
                df = df.iloc[np.where(keep)[0]].copy()

                print(f"[mesh_costs] trimmed prices in year={year}: "
                      f"kept {int(prices.size):,}/{n_before:,} "
                      f"(lo=q{trim_q:.2f}={lo:.6g}, hi=q{1-trim_q:.2f}={hi:.6g})")
            else:
                print(f"[mesh_costs] WARNING: trimming skipped (degenerate quantiles) "
                      f"lo={lo}, hi={hi}")
        else:
            print(f"[mesh_costs] WARNING: trimming skipped (too few prices: {int(prices.size)})")

        if prices.size == 0:
            print(f"[mesh_costs] WARNING: all prices removed by trimming in year={year}. Skipping plot.")
            return dict(year=int(year), made_plot=False, n_price_obs_inside=0)

        if prices.size == 0:
            print(f"[mesh_costs] WARNING: no positive prices after adjustment in year={year}. Skipping plot.")
            return dict(year=int(year), made_plot=False, n_price_obs_inside=0)

        # statewide median (STRICTLY POSITIVE sample set by construction)
        statewide_median = float(np.median(prices))
        if not np.isfinite(statewide_median) or statewide_median <= 0.0:
            print(f"[mesh_costs] WARNING: statewide median not positive ({statewide_median}). Skipping plot.")
            return dict(year=int(year), made_plot=False, n_price_obs_inside=int(prices.size))

        # ----- per-node medians via triangle->3 vertices replication -----
        v0 = tri[0, tri_ids]
        v1 = tri[1, tri_ids]
        v2 = tri[2, tri_ids]

        node_ids = np.concatenate([v0, v1, v2]).astype(np.int64)
        samples = np.concatenate([prices, prices, prices]).astype(float)

        # sort by node id, compute medians per group
        order = np.argsort(node_ids)
        node_ids_s = node_ids[order]
        samples_s = samples[order]

        node_median = np.full(N, statewide_median, dtype=float)
        node_count = np.zeros(N, dtype=np.int64)

        cuts = np.flatnonzero(np.diff(node_ids_s)) + 1
        starts = np.concatenate([[0], cuts])
        ends = np.concatenate([cuts, [node_ids_s.size]])

        for s, e in zip(starts, ends):
            nid = int(node_ids_s[s])
            vals = samples_s[s:e]
            # vals should all be positive, but enforce anyway
            vals = vals[np.isfinite(vals) & (vals > 0.0)]
            if vals.size:
                node_median[nid] = float(np.median(vals))
                node_count[nid] = int(vals.size)

        # harden: ensure no non-positive node values leak through
        bad = ~(np.isfinite(node_median) & (node_median > 0.0))
        if np.any(bad):
            node_median[bad] = statewide_median

        vmax = float(np.nanmax(node_median))
        if not np.isfinite(vmax) or vmax <= 0.0:
            print(f"[mesh_costs] WARNING: vmax not positive ({vmax}). Skipping plot.")
            return dict(year=int(year), made_plot=False, n_price_obs_inside=int(prices.size))

        # ----- build lon/lat circles like your other plots -----
        pts_xy_km = mesh.p.T
        inv = _get_inverse_projector(epsg_project)
        lon_nodes, lat_nodes = inv.transform(pts_xy_km[:, 0] * 1000.0, pts_xy_km[:, 1] * 1000.0)
        lon_nodes = np.asarray(lon_nodes, float)
        lat_nodes = np.asarray(lat_nodes, float)

        r_km = (np.sqrt(3.0) / 4.0) * float(h_km)
        KM_PER_DEG_LAT = 111.32
        dy_deg = np.full_like(lat_nodes, r_km / KM_PER_DEG_LAT, dtype=float)
        dx_deg = r_km / (KM_PER_DEG_LAT * np.clip(np.cos(np.deg2rad(lat_nodes)), 1e-6, None))

        patches = [
            Ellipse((float(x), float(y)),
                    width=float(2.0 * wx),
                    height=float(2.0 * hy))
            for x, y, wx, hy in zip(lon_nodes, lat_nodes, dx_deg, dy_deg)
        ]

        norm = Normalize(vmin=0.0, vmax=vmax)
        vals_plot = np.clip(node_median, 0.0, vmax)

        pc = PatchCollection(patches, array=vals_plot, norm=norm, linewidths=0.0)

        out_png.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")

        title = f"Node median cost ({year})"
        if cpi_adjust:
            title += f" (CPI-adjusted to {base_year})"
        ax.set_title(title)

        cbar = fig.colorbar(pc, ax=ax, location="right", fraction=0.046, pad=0.02)
        cbar.set_label("Median cost (linear)")

        ax.text(
            0.01, 0.01,
            f"Statewide median used for missing nodes: {statewide_median:.3g}\n"
            f"Positive price obs inside mesh: {int(prices.size):,}",
            transform=ax.transAxes, va="bottom"
        )
        
        print("[mesh_costs] price stats (positive inside mesh):",
              "min", float(np.min(prices)),
              "p1", float(np.quantile(prices, 0.01)),
              "p50", float(np.quantile(prices, 0.50)),
              "p99", float(np.quantile(prices, 0.99)),
              "max", float(np.max(prices)))
        
        print("[mesh_costs] node_median stats:",
              "min", float(np.min(node_median)),
              "p1", float(np.quantile(node_median, 0.01)),
              "p50", float(np.quantile(node_median, 0.50)),
              "p99", float(np.quantile(node_median, 0.99)),
              "max", float(np.max(node_median)),
              "n<=0", int(np.count_nonzero(node_median <= 0)))

        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"[mesh_costs] saved plot to {out_png}")

        return dict(
            year=int(year),
            made_plot=True,
            n_price_obs_inside=int(prices.size),
            statewide_median=float(statewide_median),
            vmax=float(vmax),
            n_nodes_with_samples=int(np.count_nonzero(node_count > 0)),
            cpi_adjust=bool(cpi_adjust),
            base_year=int(base_year),
            base_month=int(base_month) if base_month is not None else None,
        )


def plot_total_adoptions_bimodal_log_fit(
    msh_path: Path,
    events_df: pd.DataFrame,
    out_png: Path,
    *,
    epsg_project: int = 5070,
    start_date: Optional[str] = None,   # e.g. "1998-01-01"
    end_date: Optional[str] = None,     # e.g. "2023-12-31"
    eps_count: float = 1.0,
    min_months_each_side: int = 12,     # months
    chunk_size: int = 50_000,
) -> dict:
    """
    Total adoptions over time inside mesh domain.
    y-axis is log(count+eps), time axis is linear.
    Fit a 2-segment piecewise linear model (single breakpoint) by brute-force SSE scan.
    Prints the breakpoint and intersection time of the two fitted lines.

    Returns dict with fit info.
    """
    with timed("plot_total_adoptions_bimodal_log_fit"):
        mesh = load_mesh_km_from_msh(msh_path)
    
        df = events_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "longitude", "latitude"]).copy()
        if df.empty:
            raise ValueError("No valid events after parsing date/coords.")
        
        # date window
        if start_date is None:
            start_date = df["date"].min().strftime("%Y-%m-%d")
        if end_date is None:
            end_date = df["date"].max().strftime("%Y-%m-%d")
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()
        if df.empty:
            raise ValueError("No events in requested date window.")
        
        # inside mesh
        lon = df["longitude"].to_numpy(float)
        lat = df["latitude"].to_numpy(float)
        inside, tri_ids, _, _ = _events_inside_mesh_mask(mesh, lon, lat, epsg_project, chunk_size=chunk_size)
        df = df.loc[inside].copy()
        if df.empty:
            raise ValueError("No events inside mesh domain.")
        
        # ---- monthly binning: map each event to an integer month index ----
        # month0 = first month in window (normalized to month start)
        m0 = start_dt.to_period("M").to_timestamp()
        m1 = end_dt.to_period("M").to_timestamp()
        months = pd.date_range(m0, m1, freq="MS")  # month starts
        t = (months.year + (months.month - 0.5) / 12.0).to_numpy(float)  # mid-month in "year units"
        
        # event month index (vectorized)
        ev_month_start = df["date"].dt.to_period("M").dt.to_timestamp()
        idx = ((ev_month_start.dt.year - m0.year) * 12 + (ev_month_start.dt.month - m0.month)).to_numpy(np.int64)
        
        y = np.bincount(idx, minlength=len(months)).astype(float)
        y_log = np.log(y + float(eps_count))
    
        # Continuous piecewise linear fit in log space
        n = t.size
        best = None
        candidates = range(min_months_each_side - 1, n - min_months_each_side)
    
        for k in candidates:
            tc = float(t[k])
    
            # Build design matrix X for params theta = [a, b, d]
            # Left rows (<=k):  y = a + b t
            # Right rows (>k):  y = a + b tc + d (t - tc)
            X = np.zeros((n, 3), dtype=float)
    
            # left segment
            X[:k+1, 0] = 1.0
            X[:k+1, 1] = t[:k+1]
            X[:k+1, 2] = 0.0
    
            # right segment
            X[k+1:, 0] = 1.0
            X[k+1:, 1] = tc
            X[k+1:, 2] = (t[k+1:] - tc)
    
            theta, *_ = np.linalg.lstsq(X, y_log, rcond=None)
            a, b, d = map(float, theta)
    
            pred = X @ theta
            sse = float(np.sum((y_log - pred) ** 2))
    
            if (best is None) or (sse < best["sse"]):
                best = dict(k=int(k), sse=sse, a=a, b=b, d=d, tc=tc)
    
        if best is None:
            raise RuntimeError("Could not find a valid breakpoint candidate.")
    
        k = int(best["k"])
        a, b, d = best["a"], best["b"], best["d"]
        tc = float(best["tc"])
    
        # Derived second-line intercept in the original (a+bt) form:
        # y = c + d t for t > tc, where c = a + (b-d)*tc
        c = float(a + (b - d) * tc)
    
        bp_left = months[k].strftime("%Y-%m")
        bp_right = months[min(k + 1, len(months) - 1)].strftime("%Y-%m")
    
        print("\n=== Continuous 2-segment log(count) fit (intersection at breakpoint) ===")
        print(f"[bimodal] window dates: {start_date}–{end_date}")
        print(f"[bimodal] events inside mesh: {len(df):,}")
        print(f"[bimodal] breakpoint at t_c = {tc:.6f} (between {bp_left} and {bp_right}, k={k})")
        print(f"[bimodal] left : log(y+eps) = {a:.6g} + {b:.6g}*t")
        print(f"[bimodal] right: log(y+eps) = {c:.6g} + {d:.6g}*t   (continuous at t_c)")
    
        # ---- plotting ----
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(t, y + float(eps_count), marker="o", linestyle="None", markersize=3)
    
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Time (year units; monthly bins)")
        ax.set_ylabel(f"Monthly adoptions inside mesh (log scale; plotted as count+{eps_count:g})")
        ax.set_title("Monthly adoptions with continuous 2-segment fit (fit in log space)")
    
        def yhat_log(tt: np.ndarray) -> np.ndarray:
            tt = np.asarray(tt, float)
            out = np.empty_like(tt)
            left = tt <= tc
            out[left] = a + b * tt[left]
            out[~left] = a + b * tc + d * (tt[~left] - tc)
            return out
    
        def yhat_count(tt: np.ndarray) -> np.ndarray:
            return np.maximum(np.exp(yhat_log(tt)), float(eps_count))
    
        t_min, t_max = float(t.min()), float(t.max())
        t_line = np.linspace(t_min, t_max, 400)
        ax.plot(t_line, yhat_count(t_line), linewidth=2.5)
    
        # breakpoint marker (this is the intersection by construction)
        ax.axvline(tc, linestyle="--", linewidth=2)
        ax.text(tc + 0.02, float(np.nanmax(y + eps_count)) * 0.6, f"t_c={tc:.2f}", rotation=90)
    
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
    
        print(f"[bimodal] saved plot to {out_png}")
        print("Diagnostic dictionary:")
        
        return_dict = dict(
            start_date=str(start_date),
            end_date=str(end_date),
            n_months=int(n),
            n_events_inside=int(len(df)),
        
            breakpoint_k=int(k),
            breakpoint_month_left=bp_left,
            breakpoint_month_right=bp_right,
            breakpoint_year=float(tc),
        
            # left segment params
            a=float(a),
            b=float(b),
        
            # right segment params in both forms
            c=float(c),
            d=float(d),
        
            sse=float(best["sse"]),
            eps_count=float(eps_count),
            min_months_each_side=int(min_months_each_side),
            model="continuous_piecewise_linear_log",
        )
        print (return_dict)
    
        return return_dict


def plot_node_adoptions_vs_population_powerlaw(
    msh_path: Path,
    events_df: pd.DataFrame,
    out_png: Path,
    *,
    epsg_project: int = 5070,
    pop_year: Optional[float] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    chunk_size: int = 50_000,
    bins: int = 80,
    min_pop: float = 1.0,
    min_adopt: float = 1.0,
    cutoff_ratio: float = 1e-3,
    rasterize: bool = True,
    use_random_subsample_for_fit: Optional[int] = None,
) -> dict:
    """
    Build node totals:
      y_node = total adoptions assigned to node (event -> triangle -> 1/3 to vertices)
      x_node = node population at pop_year via rho_nodes * A_nodes

    Make 2D histogram with log axes; fit power law:
      y ≈ C * x^alpha  (fit in log space)
    """
    with timed("plot_node_adoptions_vs_population_powerlaw"):
        mesh = load_mesh_km_from_msh(msh_path)
        N = mesh.p.shape[1]
        tri = mesh.t  # (3, ntri)
    
        df = events_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "longitude", "latitude"]).copy()
        if df.empty:
            raise ValueError("No valid events after parsing date/coords.")
    
        df["year"] = df["date"].dt.year.astype(int)
    
        if start_year is None:
            start_year = int(df["year"].min())
        if end_year is None:
            end_year = int(df["year"].max())
    
        df = df[(df["year"] >= int(start_year)) & (df["year"] <= int(end_year))].copy()
        if df.empty:
            raise ValueError("No events in requested year window.")
    
        if pop_year is None:
            pop_year = float(df["year"].max())
    
        lon = df["longitude"].to_numpy(float)
        lat = df["latitude"].to_numpy(float)
    
        inside, tri_ids, _, _ = _events_inside_mesh_mask(mesh, lon, lat, epsg_project, chunk_size=chunk_size)
        tri_ids = tri_ids[inside]
        if tri_ids.size == 0:
            raise ValueError("No events inside mesh domain.")
    
        # node totals by 1/3 distribution to triangle vertices
        y_node = np.zeros(N, dtype=float)
    
        v0 = tri[0, tri_ids]
        v1 = tri[1, tri_ids]
        v2 = tri[2, tri_ids]
        w = np.full(tri_ids.shape[0], 1.0 / 3.0, dtype=float)
    
        np.add.at(y_node, v0, w)
        np.add.at(y_node, v1, w)
        np.add.at(y_node, v2, w)
    
        # node populations at pop_year from density_utils
        out = get_batch_nodal_density(mesh, [float(pop_year)], epsg_project=epsg_project, return_masses=True, use_cache=True)
        rho_nodes = np.asarray(out["rho_nodes"][0], float)  # persons/km^2
        A_nodes = np.asarray(out["A_nodes"], float)         # km^2
        pop_node = rho_nodes * A_nodes                      # persons per node control volume
    
        # ---- filter zeros for PLOTTING (keep broad) ----
        mask_plot = (
            (pop_node > float(min_pop)) &
            (y_node > float(min_adopt)) &
            np.isfinite(pop_node) &
            np.isfinite(y_node)
        )
        x_plot = pop_node[mask_plot].astype(float)
        y_plot = y_node[mask_plot].astype(float)
        
        if x_plot.size < 10:
            raise RuntimeError(f"Too few node samples for plotting after filtering (kept {x_plot.size}).")
        
        # ---- additional filter for FIT only: adoption-rate cutoff ----
        ratio = y_plot / np.maximum(x_plot, 1.0)
        mask_fit = ratio >= float(cutoff_ratio)
        
        x_fit = x_plot[mask_fit]
        y_fit = y_plot[mask_fit]
        
        if x_fit.size < 10:
            print(f"[power] WARNING: cutoff_ratio={cutoff_ratio:g} leaves only {x_fit.size} nodes for fit; "
                  f"falling back to fitting on all plotted nodes.")
            x_fit, y_fit = x_plot, y_plot
        
        # Optional: subsample for fit stability/speed if you have huge node counts
        if use_random_subsample_for_fit is not None and x_fit.size > int(use_random_subsample_for_fit):
            rng = np.random.default_rng(0)
            idx = rng.choice(x_fit.size, size=int(use_random_subsample_for_fit), replace=False)
            x_fit = x_fit[idx]
            y_fit = y_fit[idx]
        
        # ---- fit power law in log space: log y = c + alpha log x ----
        lx = np.log(x_fit)
        ly = np.log(y_fit)
        A = np.column_stack([np.ones_like(lx), lx])
        coeff, *_ = np.linalg.lstsq(A, ly, rcond=None)
        c, alpha = float(coeff[0]), float(coeff[1])
        C = float(np.exp(c))
        
        print("\n=== Node adoption vs population power law ===")
        print(f"[power] window years: {start_year}–{end_year}")
        print(f"[power] pop_year used: {pop_year:.0f}")
        print(f"[power] nodes plotted: {x_plot.size:,} / {N:,}")
        print(f"[power] nodes used in fit: {x_fit.size:,} (cutoff_ratio={cutoff_ratio:g})")
        print(f"[power] fit: adoptions ≈ {C:.6g} * pop^{alpha:.6g}")
        print(f"[power] alpha = {alpha:.6g}")
    
        # 2D histogram in log space (axes log)
        import matplotlib.pyplot as plt
    
        out_png.parent.mkdir(parents=True, exist_ok=True)
    
        fig, ax = plt.subplots(figsize=(9, 7))
    
        # log-spaced bins
        xbins = np.logspace(np.log10(x_plot.min()), np.log10(x_plot.max()), int(bins))
        ybins = np.logspace(np.log10(y_plot.min()), np.log10(y_plot.max()), int(bins))
    
        h = ax.hist2d(x_plot, y_plot, bins=[xbins, ybins], norm=LogNorm())
        if rasterize:
            h[3].set_rasterized(True)
    
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"Node population at year {int(pop_year)} (persons)")
        ax.set_ylabel("Total adoptions assigned to node (count)")
        ax.set_title("Node adoptions vs node population (log-log histogram)")
    
        # overlay fitted power-law line
        xline = np.logspace(np.log10(x_plot.min()), np.log10(x_plot.max()), 200)
        yline = C * (xline ** alpha)
        ax.plot(xline, yline, linewidth=2.5)
    
        # colorbar formatting
        cbar = fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Bin count (log scale)")
        cbar.locator = LogLocator(base=10)
        cbar.formatter = LogFormatterSciNotation(base=10)
        cbar.update_ticks()
    
        ax.grid(True, alpha=0.2, which="both")
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
    
        print(f"[power] saved plot to {out_png}")
    
        print('Diagnostic dictionary:')
        return_dict = dict(
            alpha=float(alpha),
            C=float(C),
            pop_year=float(pop_year),
        
            n_nodes_total=int(N),
            n_nodes_plotted=int(x_plot.size),
            n_nodes_fit=int(x_fit.size),
        
            start_year=int(start_year),
            end_year=int(end_year),
        
            min_pop=float(min_pop),
            min_adopt=float(min_adopt),
            cutoff_ratio=float(cutoff_ratio),
        
            used_subsample=int(use_random_subsample_for_fit) if use_random_subsample_for_fit is not None else None,
        )
        print(return_dict)
        
        return return_dict


# ============================================================
# Examples
# ============================================================

def main():
    base = Path("data")

    admin1_shp = base / "raw" / "maps" / "ne_10m_admin_1_states_provinces_lakes" / "ne_10m_admin_1_states_provinces_lakes.shp"

    out_dir_mesh = base / "processed"
    out_dir_fig = base / "figures"
    out_dir_mesh.mkdir(parents=True, exist_ok=True)
    out_dir_fig.mkdir(parents=True, exist_ok=True)

    # ---- Create the mesh ----
    h_km, simplify_km = 6, 18
    cfg_ca = MeshBuildConfig(h_km=h_km, simplify_km=h_km, epsg_project=5070)
    msh_ca = out_dir_mesh / "ny_"+str(h_km)+"_"+str(simplify_km)+"_km.msh"
    png_ca = out_dir_fig / "ny_"+str(h_km)+"_"+str(simplify_km)+"_mesh.png"
    print("Building NY mesh ("+str(h_km)+" km)...")
    build_mesh_from_admin1_region(admin1_shp, ["NY"], msh_ca, cfg_ca, verbose=True, model_name="ny_mesh_km")
    mesh_ca = load_mesh_km_from_msh(msh_ca)
    
    # ---- Mesh quality diagnostics ----
    print_mesh_quality_diagnostics(mesh_ca, label=f"NY (h={cfg_ca.h_km:g} km)")
    print("Plotting NY mesh...")
    plot_msh_triangles_lonlat(msh_ca, png_ca, epsg_project=cfg_ca.epsg_project)

    # ---- Population mass check + triangle comparison ----
    print_population_mass_check(admin1_shp, ["NY"], msh_ca, year=2023, epsg_project=cfg_ca.epsg_project, mesh=mesh_ca)
    pop_cmp_png = out_dir_fig / "ny_pop_true_vs_est_year2023.png"
    plot_triangle_population_comparison(admin1_shp, ["NY"], msh_ca, pop_cmp_png, year=2023, epsg_project=cfg_ca.epsg_project, mesh=mesh_ca, h_km=cfg_ca.h_km)
    
    # ---- Load adoption events ----
    csv_events = base / "processed" / "solar_installations_all.csv"
    events_df = pd.read_csv(csv_events)
    events_df["state"] = events_df["state"].astype(str).str.strip()
    events_ca_df = events_df.loc[events_df["state"] == "NY"].copy()
    
    # ---- Adoptions per month + cost (inflation-adjusted heuristic) ----
    out_png1 = out_dir_fig / "ny_adoptions_vs_costs_cpi2026.png"
    plot_adoptions_vs_costs(
        events_df=events_ca_df,
        out_png=out_png1,
        start_date=None,
        end_date=None,
        price_col="price",
        missing_price_value=-1.0,
        base_year=2026,
        base_month=None,  # set e.g. 1 for Jan-2026 if you want to force it
        cpi_cache_csv=out_dir_fig / "cpi_cache.csv",
    )
    
    # ---- Median node costs ----
    out_png2 = out_dir_fig / "node_median_costs_2023.png"
    plot_mesh_costs(
        msh_path=msh_ca,
        events_df=events_ca_df,
        out_png=out_png2,
        epsg_project=cfg_ca.epsg_project,
        year=2023,
        h_km=h_km,
        cpi_adjust=True,
        base_year=2026,
        cpi_cache_csv=out_dir_fig / "cpi_cache.csv",
    )
    
    # ---- Total adoptions over time + bimodal log fit ----
    out_png3 = out_dir_fig / "ny_total_adoptions_bimodal_logfit.png"
    plot_total_adoptions_bimodal_log_fit(
        msh_path=msh_ca,
        events_df=events_ca_df,
        out_png=out_png3,
        epsg_project=cfg_ca.epsg_project,
        start_date=None,
        end_date=None,
        eps_count=1,
        min_months_each_side=0,
    )
        
    # ---- Node adoptions vs node population + power-law fit ----
    out_png4 = out_dir_fig / "ny_node_adoptions_vs_pop_powerlaw.png"
    plot_node_adoptions_vs_population_powerlaw(
        msh_path=msh_ca,
        events_df=events_ca_df,
        out_png=out_png4,
        epsg_project=cfg_ca.epsg_project,
        start_year=1998,
        end_year=2023,
        pop_year=2023,
        bins=90,
        min_pop=1,
        min_adopt=1,
    )


if __name__ == "__main__":
    main()