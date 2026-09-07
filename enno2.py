# %%
import heapq
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, box
from shapely.ops import voronoi_diagram


def zip_shapefile(shp_path: Path) -> Path:
    zip_path = shp_path.with_suffix(".zip")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
            part = shp_path.with_suffix(ext)
            if part.exists():
                zf.write(part, arcname=part.name)

    return zip_path


def infer_grid_spacing(points: gpd.GeoDataFrame) -> tuple[float, float]:
    xs = np.sort(points.geometry.x.unique())
    ys = np.sort(points.geometry.y.unique())
    dx = float(np.median(np.diff(xs)[np.diff(xs) > 0]))
    dy = float(np.median(np.diff(ys)[np.diff(ys) > 0]))
    return dx, dy


def points_to_cells(points: gpd.GeoDataFrame, dx: float, dy: float) -> gpd.GeoDataFrame:
    hx, hy = dx / 2, dy / 2
    out = points.copy()
    out["geometry"] = [box(x - hx, y - hy, x + hx, y + hy) for x, y in zip(points.geometry.x, points.geometry.y)]
    return out


def points_to_grid_cells(points: gpd.GeoDataFrame, dx: float | None = None, dy: float | None = None):
    points = points.copy()
    if dx is None or dy is None:
        dx, dy = infer_grid_spacing(points)

    hx = dx / 2.0
    hy = dy / 2.0

    points["geometry"] = [box(x - hx, y - hy, x + hx, y + hy) for x, y in zip(points.geometry.x, points.geometry.y)]
    return points


def _null_str(series, width):
    s = series.astype("string").fillna("")
    return s.str.slice(0, width)


def _null_float(series):
    return pd.to_numeric(series, errors="coerce").astype(float)


def assign_cells_to_samples(
    cells: gpd.GeoDataFrame,
    samples: gpd.GeoDataFrame,
    value_col: str,
    sample_id_col: str,
    lambda_dist: float = 0.0,
) -> gpd.GeoDataFrame:
    """
    Seeded region-growing on a regular grid.
    Each cell gets exactly one sample assignment.
    Cost is mainly value mismatch to the seed sample.
    """

    cells = cells.copy().reset_index(drop=True)
    samples = samples.copy()

    # cell centres
    cells["cx"] = cells.geometry.centroid.x
    cells["cy"] = cells.geometry.centroid.y

    # regular-grid integer coordinates
    xs = np.sort(cells["cx"].unique())
    ys = np.sort(cells["cy"].unique())
    x_to_ix = {x: i for i, x in enumerate(xs)}
    y_to_ix = {y: i for i, y in enumerate(ys)}

    cells["gx"] = cells["cx"].map(x_to_ix).astype(int)
    cells["gy"] = cells["cy"].map(y_to_ix).astype(int)

    print(cells[value_col])
    print(samples[value_col])

    tmp = cells[["gx", "gy"]].reset_index()
    by_pos = {(row.gx, row.gy): row.index for row in tmp.itertuples(index=False)}
    neighbours = {}
    for i, row in cells.iterrows():
        gx, gy = row["gx"], row["gy"]
        ns = []
        for dx_, dy_ in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            j = by_pos.get((gx + dx_, gy + dy_))
            if j is not None:
                ns.append(j)
        neighbours[i] = ns

    # match each seed sample to the containing cell
    seed_map = {}
    cell_union = cells[["geometry"]].copy()
    join = gpd.sjoin(samples, cell_union, how="inner", predicate="within").drop_duplicates(subset=[sample_id_col])

    for _, s in join.iterrows():
        seed_map[int(s["index_right"])] = {
            "sample_id": s[sample_id_col],
            "sample_value": float(s[value_col]),
        }

    if not seed_map:
        raise ValueError("No samples could be matched to grid cells")

    assigned_sample = np.full(len(cells), None, dtype=object)
    best_cost = np.full(len(cells), np.inf, dtype=float)

    heap = []
    for cell_ix, seed in seed_map.items():
        assigned_sample[cell_ix] = seed["sample_id"]
        best_cost[cell_ix] = 0.0
        heapq.heappush(heap, (0.0, cell_ix, seed["sample_id"], seed["sample_value"]))

    vals = pd.to_numeric(cells[value_col], errors="coerce").to_numpy()

    while heap:
        cost, i, sample_id, sample_value = heapq.heappop(heap)
        if cost > best_cost[i]:
            continue

        for j in neighbours[i]:
            step_cost = abs(vals[j] - sample_value) + lambda_dist
            new_cost = cost + step_cost
            if new_cost < best_cost[j]:
                best_cost[j] = new_cost
                assigned_sample[j] = sample_id
                heapq.heappush(heap, (new_cost, j, sample_id, sample_value))

    out = cells.copy()
    out[sample_id_col] = assigned_sample
    return out


def build_seeded_field_partition(
    grid_points: gpd.GeoDataFrame,
    field_polygon,
    soil_samples: gpd.GeoDataFrame,
    value_col: str,
    sample_id_col: str,
    field_col: str = "Schlag",
    lambda_dist: float = 0.0,
) -> gpd.GeoDataFrame:
    dx, dy = infer_grid_spacing(grid_points)
    cells = points_to_cells(grid_points, dx, dy)

    # return cells

    field_gdf = gpd.GeoDataFrame({field_col: [soil_samples[field_col].iloc[0]]}, geometry=[field_polygon], crs=grid_points.crs)
    cells = gpd.overlay(cells, field_gdf, how="intersection")

    assigned = assign_cells_to_samples(
        cells=cells,
        samples=soil_samples,
        value_col=value_col,
        sample_id_col=sample_id_col,
        lambda_dist=lambda_dist,
    )

    dissolved = assigned.dissolve(by=sample_id_col, aggfunc="first").reset_index()
    dissolved = dissolved.merge(
        soil_samples.drop(columns=soil_samples.geometry.name),
        on=sample_id_col,
        how="left",
        suffixes=("", "_cells"),
    )
    # filter our _cells columns
    dissolved = dissolved.loc[:, ~dissolved.columns.str.endswith("_cells")]
    return gpd.GeoDataFrame(dissolved, geometry="geometry", crs=grid_points.crs)


def apply_seeding_fields(
    grid_points: gpd.GeoDataFrame,
    fields: gpd.GeoDataFrame,
    soil_samples: gpd.GeoDataFrame,
    value_col: str,
    sample_id_col: str,
    field_col: str = "Schlag",
    lambda_dist: float = 0.0,
):
    out = []
    for field in fields[field_col].unique():
        field_polygon = fields[fields[field_col] == field].geometry.iloc[0]
        grid_points = grid_points[grid_points.geometry.within(field_polygon)]
        samples = soil_samples[soil_samples.geometry.within(field_polygon)]
        print(f"grid_points: {grid_points.shape}, samples: {samples.shape}, field: {field}")
        # return build_seeded_field_partition(
        cells = build_seeded_field_partition(
            grid_points=grid_points,
            field_polygon=field_polygon,
            soil_samples=soil_samples,
            value_col=value_col,
            sample_id_col=sample_id_col,
            field_col=field_col,
            lambda_dist=lambda_dist,
        )
        out.append(cells)
    return pd.concat(out)


def apply_ph_liming_rule(gdf, ph_col="pH (CaCl2)"):
    ph = gdf[ph_col].astype(float)

    conditions = [
        ph <= 4.0,
        (ph > 4.0) & (ph <= 4.5),
        (ph > 4.5) & (ph <= 5.0),
        (ph > 5.0) & (ph <= 5.5),
        (ph > 5.5) & (ph <= 6.0),
        (ph > 6.0) & (ph <= 6.5),
        ph > 6.5,
    ]

    delta_values = [2.3, 1.8, 1.3, 0.8, 0.3, -0.2, -0.2]
    kalk_values = [5.5, 4.0, 2.5, 1.3, 0.4, 0.0, 0.0]

    out = gdf.copy()
    out["Fehler pH (CaCl2)"] = np.select(conditions, delta_values, default=np.nan)
    out["Kalk 85 t/ha"] = np.select(conditions, kalk_values, default=np.nan)
    return out


def export_shp(gdf: gpd.GeoDataFrame, path: Path):
    if gdf.empty:
        raise ValueError("GeoDataFrame is empty")
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS")
    if not gdf.geom_type.fillna("").isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("All geometries must be Polygon or MultiPolygon")

    gdf = gdf.copy().to_crs("EPSG:4326")
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    schema_fields = [
        ("ID", "str:5"),
        ("SCHLAGNAME", "str:30"),
        ("PH_WERT", "float:10.8"),
        ("B_WERT", "float:10.8"),
        ("CU_WERT", "float:10.8"),
        ("MN_WERT", "float:10.8"),
        ("ZN_WERT", "float:10.8"),
        ("S_WERT", "float:10.8"),
        ("NA_WERT", "float:10.8"),
        ("MG_WERT", "float:10.8"),
        ("K_WERT", "float:10.8"),
        ("P_WERT", "float:10.8"),
        ("FEIN", "float:10.8"),
        ("SOM_WERT", "float:10.8"),
        ("PROBEDATUM", "str:20"),
        ("LABORPRNR", "str:30"),
        ("PROBENUMME", "str:30"),
        ("Sand_WERT", "float:10.8"),
        ("Silt_WERT", "float:10.8"),
        ("Ertrag_WERT", "float:10.8"),
        ("Salz_WERT", "float:10.8"),
        ("SOC_WERT", "float:10.8"),
        ("PPI_WERT", "float:10.8"),
        ("ph_delta", "float:10.8"),
        ("K85_tha", "float:10.8"),
    ]

    widths = {name: int(spec.split(":")[1].split(".")[0]) for name, spec in schema_fields if spec.startswith("str:")}

    # Start with empty target frame in exact field order
    out = gpd.GeoDataFrame(index=gdf.index, geometry=gdf.geometry, crs=gdf.crs)

    for name, spec in schema_fields:
        if spec.startswith("str:"):
            out[name] = ""
        else:
            out[name] = np.nan

    # Minimal mapping from your current data into the target schema.
    # Everything not mapped remains empty/NaN.
    mapping = {
        "id": "ID",
        "Schlag": "SCHLAGNAME",
        "pH (CaCl2)": "PH_WERT",
        "Boron (mg/kg)": "B_WERT",
        "Chemistry__Copper (mg/kg)": "CU_WERT",
        "Chemistry__Manganese (mg/kg)": "MN_WERT",
        "Chemistry__Zinc (mg/kg)": "ZN_WERT",
        "Chemistry__Sulphur (mg/100g)": "S_WERT",
        "Chemistry__Sodium (mg/kg)": "NA_WERT",
        "Chemistry__Magnesium (mg/100g)": "MG_WERT",
        "Chemistry__Potassium (mg/100g)": "K_WERT",
        "Chemistry__Phosphate (mg/100g)": "P_WERT",
        "Physics__Clay (%)": "FEIN",
        "Chemistry__Soil organic matter (%)": "SOM_WERT",
        "sampled_at_latest": "PROBEDATUM",
        "analysis_sample_id": "LABORPRNR",
        "primary_sample_id": "PROBENUMME",
        "Physics__Sand (%)": "Sand_WERT",
        "Physics__Silt (%)": "Silt_WERT",
        "Yield__Ertrag (t/ha)": "Ertrag_WERT",
        "Chemistry__Salt content (mg/100g)": "Salz_WERT",
        "Chemistry__Soil organic carbon (%)": "SOC_WERT",
        "PPI": "PPI_WERT",
        "Fehler pH (CaCl2)": "ph_delta",
        "Kalk 85 t/ha": "K85_tha",
    }

    unknown_targets = set(mapping.values()) - set(out.columns)
    if unknown_targets:
        raise ValueError(f"Mapping targets not present in output schema: {sorted(unknown_targets)}")

    # print(gdf["Fehler pH (CaCl2)"])

    for src, dst in mapping.items():
        if src in gdf.columns:
            out[dst] = gdf[src]

    # Type coercion exactly to DBF-friendly types
    for name, spec in schema_fields:
        if spec.startswith("str:"):
            out[name] = _null_str(out[name], widths[name])
        else:
            out[name] = _null_float(out[name])

    schema = {
        "geometry": "Polygon",
        "properties": dict(schema_fields),
    }
    print(out.columns)

    out_path = path.with_suffix(".shp")
    out = out[[name for name, _ in schema_fields] + [out.geometry.name]]
    out.to_file(
        out_path,
        driver="ESRI Shapefile",
        schema=schema,
        encoding="UTF-8",
        engine="fiona",
    )

    # Best way to match the reference CRS text exactly
    ref_prj = "target.prj"
    shutil.copyfile(ref_prj, out_path.with_suffix(".prj"))

    zip_shapefile(out_path)
    return out_path


def _maybe_project(points: gpd.GeoDataFrame, fields: gpd.GeoDataFrame):
    if points.crs is None:
        raise ValueError("samples.geojson has no CRS")
    if fields.crs is None:
        raise ValueError("fields.geojson has no CRS")

    if fields.crs != points.crs:
        fields = fields.to_crs(points.crs)

    if points.crs.is_geographic:
        target_crs = points.estimate_utm_crs()
        points = points.to_crs(target_crs)
        fields = fields.to_crs(target_crs)

    return points, fields


def build_field_voronoi(samples: gpd.GeoDataFrame, fields: gpd.GeoDataFrame, field_col="Schlag") -> gpd.GeoDataFrame:
    if field_col not in samples.columns:
        raise ValueError(f"Missing '{field_col}' in samples")
    if field_col not in fields.columns:
        raise ValueError(f"Missing '{field_col}' in fields")

    samples, fields = _maybe_project(samples, fields)
    original_crs = samples.crs

    fields_diss = fields[[field_col, fields.geometry.name]].dissolve(by=field_col).reset_index()

    out = []

    for schlag, pts in samples.groupby(field_col, dropna=True):
        fld = fields_diss.loc[fields_diss[field_col] == schlag]
        if fld.empty:
            print(f"Skipping {schlag}: no matching field geometry")
            continue

        field_geom = fld.geometry.iloc[0]
        pts = pts.loc[~pts.geometry.is_empty & pts.geometry.notna()].copy()

        if pts.empty:
            print(f"Skipping {schlag}: no sample geometries")
            continue

        # Degenerate case: one point -> the whole field gets that point's attributes
        if len(pts) == 1:
            row = pts.drop(columns=pts.geometry.name).iloc[[0]].copy()
            row = gpd.GeoDataFrame(row, geometry=[field_geom], crs=original_crs)
            out.append(row)
            continue

        mp = MultiPoint(list(pts.geometry))
        vd = voronoi_diagram(mp, envelope=field_geom.envelope, edges=False)

        cells = [geom for geom in vd.geoms if geom.geom_type == "Polygon"]
        cells_gdf = gpd.GeoDataFrame({"geometry": cells}, crs=original_crs)

        # restrict to the field boundary
        cells_gdf = gpd.overlay(
            cells_gdf,
            gpd.GeoDataFrame({field_col: [schlag]}, geometry=[field_geom], crs=original_crs),
            how="intersection",
        )

        if cells_gdf.empty:
            print(f"Skipping {schlag}: Voronoi clip produced no polygons")
            continue

        # assign each polygon back to its generating point by spatial containment
        pts_keyed = pts.reset_index(drop=True).reset_index().rename(columns={"index": "_pt_ix"})
        joined = gpd.sjoin(cells_gdf, pts_keyed, how="left", predicate="contains")

        # In normal Voronoi output each cell contains exactly one generator point
        joined = joined.dropna(subset=["_pt_ix"]).drop(columns=["index_right"])
        joined["_pt_ix"] = joined["_pt_ix"].astype(int)

        attrs = pts_keyed.drop(columns=pts_keyed.geometry.name)
        merged = joined.merge(attrs, on="_pt_ix", how="left").drop(columns=["_pt_ix"])

        # keep polygon geometry only
        merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=original_crs)
        out.append(merged)

    if not out:
        return gpd.GeoDataFrame(columns=samples.columns, geometry="geometry", crs=original_crs)

    result = pd.concat(out, ignore_index=True)
    return gpd.GeoDataFrame(result, geometry="geometry", crs=original_crs)


import geopandas as gpd
import numpy as np
from shapely.geometry import Point


def densify_grid_over_fields(
    fields: gpd.GeoDataFrame,
    spacing: float,
    field_id_col: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Create a regular grid of points inside each field polygon.

    Scope:
    - Uses a fixed spacing in the CRS units of `fields`
    - Keeps field attributes on each generated point
    - Does not interpolate any values from existing grid points
    """
    if fields.empty:
        return gpd.GeoDataFrame(columns=list(fields.columns), geometry=[], crs=fields.crs)

    out = []

    for _, row in fields.iterrows():
        poly = row.geometry
        if poly is None or poly.is_empty:
            continue

        minx, miny, maxx, maxy = poly.bounds
        xs = np.arange(minx, maxx + spacing, spacing)
        ys = np.arange(miny, maxy + spacing, spacing)

        pts = [Point(x, y) for x in xs for y in ys]
        # mask = [poly.covers(p) for p in pts]
        # pts = [p for p, keep in zip(pts, mask) if keep]
        print(f"pts: {len(pts)} points in field {row[field_id_col]}")

        if not pts:
            continue

        attrs = row.drop(labels="geometry").to_dict()
        gdf = gpd.GeoDataFrame(
            [attrs] * len(pts),
            geometry=pts,
            crs=fields.crs,
        )

        if field_id_col is not None and field_id_col in row.index:
            gdf[field_id_col] = row[field_id_col]

        out.append(gdf)

    if not out:
        return gpd.GeoDataFrame(columns=list(fields.columns), geometry=[], crs=fields.crs)

    return gpd.GeoDataFrame(
        pd.concat(out, ignore_index=True),
        geometry="geometry",
        crs=fields.crs,
    )


# %%


def assign_sample_attributes_to_grid(
    grid_points: gpd.GeoDataFrame,
    samples: gpd.GeoDataFrame,
    max_distance: float | None = None,
    keep_grid_geometry_only: bool = True,
) -> gpd.GeoDataFrame:
    """
    Assign each grid point all attributes of its nearest sample.

    Scope:
    - constant assignment by nearest sample
    - no interpolation
    - returns one row per grid point

    Parameters
    ----------
    max_distance
        Optional maximum search distance in CRS units.
        Grid points without a nearby sample get NaNs.
    keep_grid_geometry_only
        Drops the sample geometry column from the result.
    """
    if grid_points.crs != samples.crs:
        raise ValueError("grid_points and samples must use the same CRS")

    samples_right = samples.copy()

    # if keep_grid_geometry_only:
    #     samples_right = samples_right.drop(columns="geometry")

    out = gpd.sjoin_nearest(
        grid_points,
        samples_right,
        how="left",
        max_distance=max_distance,
        distance_col="distance_to_sample",
        lsuffix="grid_points",
        rsuffix="",
    )

    if "index_right" in out.columns:
        out = out.drop(columns="index_right")
    out = out.drop(columns="distance_to_sample")
    return out


samples_path = Path("samples.geojson")
fields_path = Path("fields.geojson")
grid_points_path = Path("grid.geojson")
output_path = Path("Bodenproben.shp")

samples = gpd.read_file(samples_path)
grid_points = gpd.read_file(grid_points_path)
fields = gpd.read_file(fields_path)

translate = {"Boron": "Boron (mg/kg)", "Schlag": "Schlag", "Chemistry__pH CaCl2": "pH (CaCl2)"}
samples = samples.rename(columns=translate)
fields = fields.rename(columns={"Schlag": "Schlag"})
# grid_points = grid_points.rename(columns={"Chemistry__pH CaCl2": "pH (CaCl2)"})

schlag_col = "Schlag"
remove = ["Roßd. Sandweg klein" "Roßd. Sandw Dreieck"]

samples = samples.loc[:, ~samples.columns.duplicated()]
samples = samples[~samples[schlag_col].isin(remove)].copy()
fields = fields[~fields[schlag_col].isin(remove)].copy()


samples = apply_ph_liming_rule(samples, ph_col="pH (CaCl2)")

# grid_points = densify_grid_over_fields(fields, spacing=0.001 / 5, field_id_col="Schlag")
# grid_points = assign_sample_attributes_to_grid(grid_points, samples)

voronoi_polygons = build_field_voronoi(samples, fields, field_col=schlag_col)
voronoi_polygons = voronoi_polygons.to_crs(samples.crs)
# attributes by spatial containment
voronoi_polygons = assign_sample_attributes_to_grid(voronoi_polygons, samples)
print(voronoi_polygons.columns)
export_shp(voronoi_polygons, Path("2026-03-27_Soilytix-Bodenproben.shp"))
# cells = apply_seeding_fields(
#     grid_points=grid_points,
#     fields=fields,
#     soil_samples=samples,
#     value_col="pH (CaCl2)",
#     sample_id_col="primary_sample_id",
#     field_col=schlag_col,
#     lambda_dist=0.0,
# )


# %%
out = []
for field in fields[schlag_col].unique():
    field_polygon = fields[fields[schlag_col] == field].geometry.iloc[0]
    print(f"field_polygon: {field} — {field_polygon.area}")
    grid_points_ = grid_points[grid_points.geometry.within(field_polygon)]
    samples_ = samples[samples.geometry.within(field_polygon)]
    print(f"grid_points: {grid_points_.shape}, samples: {samples_.shape}, field: {field}")
    # return build_seeded_field_partition(
    cells = build_seeded_field_partition(
        grid_points=grid_points_,
        field_polygon=field_polygon,
        soil_samples=samples_,
        value_col="pH (CaCl2)",
        sample_id_col="primary_sample_id",
        field_col="Schlag",
        lambda_dist=0.0,
    )
    out.append(cells)
out = pd.concat(out)

# export_shp(cells, output_path)
# %%
out.columns
# %%
export_shp(out, output_path)
# %%
output_path.resolve()
# %%
