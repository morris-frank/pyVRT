# %%
import argparse
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd


def zip_shapefile(shp_path: Path) -> Path:
    zip_path = shp_path.with_suffix(".zip")
    stem = shp_path.stem

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
            part = shp_path.with_suffix(ext)
            if part.exists():
                zf.write(part, arcname=part.name)

    return zip_path


import numpy as np


def apply_ph_liming_rule(gdf, ph_col="pH (CaCl2)"):
    """
    Apply the liming rule to each row of a GeoDataFrame.

    Rules:
        Ist-pH <= 4.0  -> Ziel-pH 6.3, delta +2.3, Kalk 5.5 t/ha
        <= 4.5         -> Ziel-pH 6.3, delta +1.8, Kalk 4.0 t/ha
        <= 5.0         -> Ziel-pH 6.3, delta +1.3, Kalk 2.5 t/ha
        <= 5.5         -> Ziel-pH 6.3, delta +0.8, Kalk 1.3 t/ha
        <= 6.0         -> Ziel-pH 6.3, delta +0.3, Kalk 0.4 t/ha
        <= 6.5         -> Ziel-pH 6.3, delta -0.2, Kalk 0.0 t/ha

    For pH > 6.5, Kalk is set to 0.0 and Ziel-pH remains 6.3.
    """

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
    # out["ziel_pH"] = 6.3
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
    filtered = gdf[list(field_map.keys()) + [gdf.geometry.name]]
    filtered = filtered.rename(columns=field_map)

    geojson_output = path.with_suffix(".geojson")
    filtered.to_file(geojson_output, driver="GeoJSON")
    shp_output = path.with_suffix(".shp")
    filtered.to_file(shp_output, driver="ESRI Shapefile")
    zip_path = zip_shapefile(shp_output)


def process_field_grid_shapefiles(
    gdf,
    field_id_col,
    input_path_template,
    output_path: Path,
    ph_col="pH",
):
    """
    For each unique field ID in `gdf`:
    - load the corresponding point shapefile
    - apply the liming rule to all points using `ph_col`
    - save a new shapefile next to the input shapefile

    Parameters
    ----------
    gdf : GeoDataFrame or DataFrame
        Table containing the field IDs to process.
    field_id_col : str
        Column in `gdf` containing the field identifier.
    input_path_template : str
        Path template with `{field_id}` placeholder, e.g.
        "/data/grids/{field_id}/grid_points.shp"
        or
        "/data/grids/{field_id}.shp"
    ph_col : str
        Name of the pH column in the point shapefiles.
    output_suffix : str
        Suffix appended to the input shapefile stem.

    Returns
    -------
    list[Path]
        Paths of written shapefiles.
    """
    written_files = []

    points_outs = []
    for field_id in gdf[field_id_col].dropna().unique():
        # Güssow Ottos Mais -> G_ssow_Ottos_Mais
        field_id = field_id.replace(" ", "_").replace("-", "_").replace("ü", "_").replace("ä", "_").replace("ö", "_").replace("ß", "_")
        # replace "__" with "_"
        field_id = field_id.replace("__", "_")
        input_path = Path(input_path_template.format(field_id=field_id))

        if not input_path.exists():
            print(f"Skipping {field_id}: missing {input_path}")
            continue

        points = gpd.read_file(input_path)
        translate = {"pH": "pH (CaCl2)", "Boron": "Boron (mg/kg)", "Schlag": "Schlag"}
        points = points.rename(columns=translate)

        if ph_col not in points.columns:
            print(f"Skipping {field_id}: missing column {ph_col} in {input_path}")
            continue

        points_out = apply_ph_liming_rule(points, ph_col=ph_col)
        points_outs.append(points_out)

    combined = pd.concat(points_outs)
    export_shp(combined, output_path.with_name(f"{output_path.stem}_mapped.shp"))


def main():
    input_path = Path("samples.geojson")
    geojson_output = Path("samples.raw.geojson")
    shp_output = Path("samples.raw.shp")

    gdf = gpd.read_file(input_path)

    schlag_col = "Schlag"
    schlag_filter = gdf[schlag_col].isin(
        [
            "Güssow Ottos Mais",
            "Rossd. am Kanal lang",
            "Schlag. Sandlake",
        ]
    )
    filtered = gdf[schlag_filter]

    filtered = apply_ph_liming_rule(filtered)

    export_shp(filtered, geojson_output)
    # vrt_path_template = "/Users/mfr/code/soilytix/pyVRT/output/field_id={field_id}/svm_grid_features.shp"
    # process_field_grid_shapefiles(filtered, "Schlag", vrt_path_template, output_path=shp_output, ph_col="pH (CaCl2)")


# %%
main()

# %%
