# %%
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint
from shapely.ops import voronoi_diagram


def zip_shapefile(shp_path: Path) -> Path:
    zip_path = shp_path.with_suffix(".zip")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
            part = shp_path.with_suffix(ext)
            if part.exists():
                zf.write(part, arcname=part.name)

    return zip_path


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


def export_shp(gdf, path: Path):
    field_map = {
        "id": "id",
        "project_id": "proj_id",
        "sampling_area_id": "samp_ar_id",
        "Physics__Clay (%)": "phy_clay",
        "Physics__Sand (%)": "phy_sand",
        "Physics__Silt (%)": "phy_silt",
        "primary_sample_id": "prim_s_id",
        "sampled_at_latest": "samp_date",
        "analysis_sample_id": "anal_s_id",
        "pH (CaCl2)": "ph_cacl2",
        "Yield__Ertrag (t/ha)": "yield_tha",
        "Chemistry__Zinc (mg/kg)": "zn_mgkg",
        "Boron (mg/kg)": "b_mgkg",
        "Chemistry__Copper (mg/kg)": "cu_mgkg",
        "Chemistry__Sodium (mg/kg)": "na_mgkg",
        "Chemistry__Manganese (mg/kg)": "mn_mgkg",
        "Chemistry__Sulphur (mg/100g)": "s_mg100g",
        "Chemistry__Magnesium (mg/100g)": "mg_mg100g",
        "Chemistry__Phosphate (mg/100g)": "p_mg100g",
        "Chemistry__Potassium (mg/100g)": "k_mg100g",
        "Chemistry__Salt content (mg/100g)": "salt_mg1",
        "Chemistry__Soil organic carbon (%)": "soc_pct",
        "Chemistry__Soil organic matter (%)": "som_pct",
        "nutzbare_Feldkapazität__VBK50___FK1M": "nfk_vbk50",
        "potenzielle_Kationenaustauschkapazität__VBK50___FK1M": "cec_vbk50",
        "organische_Kohlenstoffvorräte_im_Boden__VBK50___organischer_Kohlenstoffvorrat": "cstock_vb",
        "Müncheberger_Soil_Quality_Rating__MSQR": "msqr",
        "clay": "clay",
        "soil_organic_matter": "som",
        "pH": "ph",
        "soil_organic_carbon": "soc",
        "MSQR": "msqr_val",
        "nFK": "nfk",
        "elevation": "elev",
        "slope": "slope",
        "Schlag": "field",
        "PPI": "ppi",
        "Fehler pH (CaCl2)": "ph_err",
        "Kalk 85 t/ha": "kalk85_tha",
    }

    keep = [c for c in field_map if c in gdf.columns] + [gdf.geometry.name]
    filtered = gdf[keep].rename(columns=field_map)

    geojson_output = path.with_suffix(".geojson")
    filtered.to_file(geojson_output, driver="GeoJSON")

    shp_output = path.with_suffix(".shp")
    filtered.to_file(shp_output, driver="ESRI Shapefile")
    zip_shapefile(shp_output)


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


def main():
    samples_path = Path("samples.geojson")
    fields_path = Path("fields.geojson")
    output_path = Path("samples_voronoi.shp")

    samples = gpd.read_file(samples_path)
    fields = gpd.read_file(fields_path)

    translate = {"pH": "pH (CaCl2)", "Boron": "Boron (mg/kg)", "Schlag": "Schlag"}
    samples = samples.rename(columns=translate)
    fields = fields.rename(columns={"Schlag": "Schlag"})

    schlag_col = "Schlag"
    keep = [
        "Güssow Ottos Mais",
        "Rossd. am Kanal lang",
        "Schlag. Sandlake",
    ]

    samples = samples[samples[schlag_col].isin(keep)].copy()
    fields = fields[fields[schlag_col].isin(keep)].copy()

    samples = apply_ph_liming_rule(samples, ph_col="pH (CaCl2)")
    voronoi_polygons = build_field_voronoi(samples, fields, field_col=schlag_col)

    export_shp(voronoi_polygons, output_path)


# %%
main()
# %%
