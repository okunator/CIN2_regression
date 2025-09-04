import geopandas as gpd
import numpy as np
import pandas as pd
from histolytics.spatial_geom.medial_lines import (
    medial_lines,
    sliding_perpendicular_lines,
)
from histolytics.spatial_geom.shape_metrics import shape_metric
from histolytics.spatial_ops.ops import get_objs
from histolytics.utils.gdf import set_crs

TO_MM_CONVERSION = 0.5 / 1e3  # conversion to mm
TO_MM_SQUARED_CONVERSION = 0.125 / 1e6  # conversion to mm^2


def epithelial_structures_pipeline(
    tis: gpd.GeoDataFrame, nuc: gpd.GeoDataFrame
) -> tuple[pd.Series, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Pipeline to compute various metrics related to (healthy and non-healthy) epithelial structures.

    The following metrics are computed:
        - Lesion Total Area
        - Gland Area
        - Squamous Area
        - Invaded Gland Area
        - Gland to Biopsy Area Proportion
        - Squamous to Biopsy Area Proportion
        - Lesion to Biopsy Area Proportion
        - Lesion to Gland Area Proportion
        - Lesion to Squamous Area Proportion
        - Lesion to Healthy Epithelial Area Proportion
        - Invaded Gland to Healthy Epithelial Area Proportion
        - Invaded Gland to All Gland Area Proportion
        - Neoplastic Nuclei Proportion
        - Squamous Nuclei Proportion
        - Glandular Nuclei Proportion
        - Neoplastic to Glandular Nuclei Proportion
        - Neoplastic to Healthy Epithelial Nuclei Proportion
        - Lesion Compactness
        - Lesion Circularity
        - Lesion Convexity
        - Lesion Elongation
        - Lesion Eccentricity
        - Lesion Fractal_Dimension
        - Lesion Medial Line Length
        - Lesion Mean Depth
        - Lesion Std Depth

    Parameters:
        tis (GeoDataFrame): GeoDataFrame containing tissue polygons.
        nuc (GeoDataFrame): GeoDataFrame containing nuclei polygons.

    Returns:
        tuple: A tuple containing the following elements:
            - pd.Series: A series containing the computed metrics.
            - gpd.GeoDataFrame: The GeoDataFrame containing the medial lines.
            - gpd.GeoDataFrame: The GeoDataFrame containing the depth lines.
            - gpd.GeoDataFrame: The GeoDataFrame containing the lesion tissue polygons.
    """
    tis = set_crs(tis)
    nuc = set_crs(nuc)

    lesion_stats, cin_tis = neo_tissue_pipeline(tis, nuc)
    medial_res, medials, perp_lines = medial_and_depth_pipeline(cin_tis, "Lesion")
    lesion_shapes = lesion_shape_pipeline(cin_tis, "Lesion")

    return (
        pd.concat([medial_res, lesion_shapes, lesion_stats]),
        medials,
        perp_lines,
        cin_tis,
    )


def medial_and_depth_pipeline(
    tis, tissue_name: str
) -> tuple[pd.Series, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Compute the medial line and depth lines of the tissue polygons.

    The computed features are:
        - Medial Line Length (in mm)
        - Mean Depth (in mm)
        - Std Depth (in mm)

    Parameters:
        tis (GeoDataFrame): GeoDataFrame containing tissue polygons.
        tissue_name (str): Name of the tissue type.

    Returns:
        tuple: A tuple containing the following elements:
            - pd.Series: A series containing the computed features.
            - gpd.GeoDataFrame: The GeoDataFrame containing the medial lines.
            - gpd.GeoDataFrame: The GeoDataFrame containing the depth lines.
    """
    medials = medial_lines(tis)

    # get perpendicular lines to the medial lines
    perp_lines = sliding_perpendicular_lines(
        medials, tis.union_all(), step_distance=300, perp_length=5000
    )

    # get the sum of medial line lengths as the total length
    medial_len = medials.length.sum()

    # get the mean length of the perp_lines as the mean breadth of the lesion
    mean_depth = perp_lines.length.mean()
    median_depth = perp_lines.length.median()
    std_depth = perp_lines.length.std()

    res = pd.Series(
        {
            f"{tissue_name} Medial Line Length": medial_len * TO_MM_CONVERSION,
            f"{tissue_name} Mean Depth": mean_depth * TO_MM_CONVERSION,
            f"{tissue_name} Median Depth": median_depth * TO_MM_CONVERSION,
            f"{tissue_name} Std Depth": std_depth * TO_MM_CONVERSION,
        }
    )

    return res, medials, perp_lines


def lesion_shape_pipeline(
    tis: gpd.GeoDataFrame, tissue_name: str, area_threshold_factor: int = 15
) -> pd.Series:
    """This function computes shape metrics for a tissue gdf.

    The following metrics are computed:
        - Lesion Compactness
        - Lesion Circularity
        - Lesion Convexity
        - Lesion Elongation
        - Lesion Eccentricity
        - Lesion Fractal_Dimension
        - Lesion Total Area (in mm^2)

    Note:
        If there is a tissue segmentation that is significantly larger (15x) than others,
        all the metrics will be computed only on that tissue segmentation except for the area.
        The area will be computed as the sum of all tissue segmentation areas.

    Parameters:
        tis (GeoDataFrame): GeoDataFrame containing tissue polygons
        tissue_name (str): Name of the tissue type
        area_threshold_factor (int): Factor to determine significant area difference

    Returns:
        pd.Series: A series containing the computed shape metrics.
    """
    shape_metrics = [
        "area",
        "compactness",
        "circularity",
        "convexity",
        "elongation",
        "eccentricity",
        "fractal_dimension",
    ]

    # compute shape metrics for the lesion
    tis = shape_metric(tis, shape_metrics, parallel=False)

    # Calculate area statistics
    areas = tis["area"].values
    max_area = areas.max() * TO_MM_SQUARED_CONVERSION

    # Sort areas in descending order to get second biggest
    sorted_areas = np.sort(areas)[::-1]
    max_to_second_ratio = 1.0
    if len(sorted_areas) >= 2:
        second_biggest_area = sorted_areas[1] * TO_MM_SQUARED_CONVERSION
        max_to_second_ratio = max_area / second_biggest_area

    # If max area is significantly larger than others, use max for other metrics
    use_max = max_to_second_ratio > area_threshold_factor

    # Compute metrics
    result = {}
    for metric in shape_metrics:
        if metric != "area":  # Skip area metric
            if use_max:
                result[f"{tissue_name} {metric}".title()] = tis[metric].max()
            else:
                result[f"{tissue_name} {metric}".title()] = tis[metric].mean()
    result[f"{tissue_name} Total Area"] = areas.sum() * TO_MM_SQUARED_CONVERSION

    return pd.Series(result)


def neo_tissue_pipeline(
    tis: gpd.GeoDataFrame, nuc: gpd.GeoDataFrame
) -> tuple[pd.Series, gpd.GeoDataFrame]:
    """Pipeline to compute various metrics related to lesion tissue and neoplastic nuclei.

    The following metrics are computed:
        - Gland Area (in mm^2)
        - Squamous Area (in mm^2)
        - Invaded Gland Area (in mm^2)
        - Gland to Biopsy Area Proportion
        - Squamous to Biopsy Area Proportion
        - Lesion to Biopsy Area Proportion
        - Lesion to Gland Area Proportion
        - Lesion to Squamous Area Proportion
        - Lesion to Healthy Epithelial Area Proportion
        - Invaded Gland to Healthy Epithelial Area Proportion
        - Invaded Gland to All Gland Area Proportion
        - Neoplastic Nuclei Proportion
        - Squamous Nuclei Proportion
        - Glandular Nuclei Proportion
        - Neoplastic to Glandular Nuclei Proportion
        - Neoplastic to Healthy Epithelial Nuclei Proportion

    Parameters:
        tis (GeoDataFrame): GeoDataFrame containing tissue polygons.
        nuc (GeoDataFrame): GeoDataFrame containing nuclei polygons.


    Returns:
        pd.Series: A series containing the computed metrics.
        gpd.GeoDataFrame: The GeoDataFrame containing the lesion tissue polygons.
    """

    # get tissues first:
    biopsy_area = tis.area.sum()

    # lesion
    cin_tis = tis[tis["class_name"] == "area_cin"]
    lesion_area = cin_tis.area.sum()
    cin_tis = cin_tis.loc[cin_tis.area > 1e5]

    # glands
    gland = tis[tis["class_name"] == "areagland"]
    gland = gland.loc[gland.area > 1e5]

    # squamous epithelium
    squam = tis[tis["class_name"] == "areasquam"]
    squam = squam.loc[squam.area > 1e5]

    # stroma
    stroma = tis[tis["class_name"] == "areastroma"]
    stroma = stroma.loc[stroma.area > 1e5]

    # compute areal proportions
    gland_area = gland.area.sum() if not gland.empty else 0.0
    squam_area = squam.area.sum() if not squam.empty else 0.0
    gland_proportion = gland_area / biopsy_area if biopsy_area > 0 else 0.0
    squam_proportion = squam_area / biopsy_area if biopsy_area > 0 else 0.0
    lesion_proportion = lesion_area / biopsy_area if biopsy_area > 0 else 0.0

    lesion_to_gland_prop = (
        lesion_area / (gland_area + lesion_area)
        if (gland_area + lesion_area) > 0
        else 0.0
    )
    lesion_to_squam_prop = (
        lesion_area / (squam_area + lesion_area)
        if (squam_area + lesion_area) > 0
        else 0.0
    )
    lesion_to_epithel_prop = (
        lesion_area / (gland_area + squam_area + lesion_area)
        if (gland_area + squam_area + lesion_area) > 0
        else 0.0
    )

    # compute gland invasion portions
    gland_cin_intersect: gpd.GeoDataFrame = get_objs(
        cin_tis.dissolve().explode(index_parts=False), gland
    )
    invaded_gland_area = gland_cin_intersect.area.sum()
    invaded_gland_area_proportion = invaded_gland_area / gland_area
    invaded_gland_to_healthy_epith_prop = invaded_gland_area / (gland_area + squam_area)

    # compute cell type proportions
    n_total = len(nuc)
    cin_nuc = nuc[nuc["class_name"] == "neoplastic"]
    squam_nuc = nuc[nuc["class_name"] == "squamous_epithel"]
    gland_nuc = nuc[nuc["class_name"] == "glandular_epithel"]

    cin_nuc_prop = len(cin_nuc) / n_total if n_total > 0 else 0.0
    squam_nuc_prop = len(squam_nuc) / n_total if n_total > 0 else 0.0
    gland_nuc_prop = len(gland_nuc) / n_total if n_total > 0 else 0.0

    cin_nuc_to_gland_nuc_prop = (
        len(cin_nuc) / (len(cin_nuc) + len(gland_nuc))
        if (len(cin_nuc) + len(gland_nuc)) > 0
        else 0.0
    )
    cin_nuc_to_healthy_epith_prop = (
        len(cin_nuc) / (len(cin_nuc) + len(gland_nuc) + len(squam_nuc))
        if (len(cin_nuc) + len(gland_nuc) + len(squam_nuc)) > 0
        else 0.0
    )

    res = {
        "Gland Area": gland_area * TO_MM_SQUARED_CONVERSION,
        "Squamous Area": squam_area * TO_MM_SQUARED_CONVERSION,
        "Invaded Gland Area": invaded_gland_area * TO_MM_SQUARED_CONVERSION,
        "Gland to Biopsy Area Proportion": gland_proportion,
        "Squamous to Biopsy Area Proportion": squam_proportion,
        "Lesion to Biopsy Area Proportion": lesion_proportion,
        "Lesion to Gland Area Proportion": lesion_to_gland_prop,
        "Lesion to Squamous Area Proportion": lesion_to_squam_prop,
        "Lesion to Healthy Epithelial Area Proportion": lesion_to_epithel_prop,
        "Invaded Gland to Healthy Epithelial Area Proportion": invaded_gland_to_healthy_epith_prop,
        "Invaded Gland to All Gland Area Proportion": invaded_gland_area_proportion,
        "Neoplastic Nuclei Proportion": cin_nuc_prop,
        "Squamous Nuclei Proportion": squam_nuc_prop,
        "Glandular Nuclei Proportion": gland_nuc_prop,
        "Neoplastic to Glandular Nuclei Proportion": cin_nuc_to_gland_nuc_prop,
        "Neoplastic to Healthy Epithelial Nuclei Proportion": cin_nuc_to_healthy_epith_prop,
    }

    return pd.Series(res), cin_tis
