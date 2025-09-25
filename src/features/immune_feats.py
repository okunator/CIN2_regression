from typing import List, Tuple

import geopandas as gpd
import pandas as pd
from histolytics.spatial_clust.density_clustering import density_clustering
from histolytics.spatial_ops.ops import get_interfaces, get_objs
from histolytics.utils.gdf import set_crs, set_uid
from shapely import unary_union

TO_MM_CONVERSION = 0.5 / 1e3  # conversion to mm
TO_MM_SQUARED_CONVERSION = 0.125 / 1e6  # conversion to mm^2


def immune_feat_pipe(tis: gpd.GeoDataFrame, nuc: gpd.GeoDataFrame) -> pd.Series:
    """Compute immune features from tissue and nuclei GeoDataFrames.

    The following features are computed:
    - Overall Immune Cell Proportion
    - Distal Immune to All Immune Proportion
    - LIIs to All Immune Proportion
    - LIIs to All Neoplastic Proportion
    - LIIs within Lesion to All Immune Proportion
    - LIIs within Lesion to All Neoplastic Proportion
    - LIIs at LSI to All Immune Proportion
    - LIIs at LSI to All Neoplastic Proportion
    - LIIs within Lesion to All LIIs proportion
    - Clustered to All Immune Cells Proportion
    - Gland Adjacent to All Clustered Immune Proportion
    - Lesion Adjacent to All Clustered Immune Proportion
    - Squam Adjacent to All Clustered Immune Proportion
    - Number of Immune Cell Clusters
    - Average Number of Cells per Immune Cell Cluster
    - Std Number of Cells per Immune Cell Cluster
    - Number of Cells in Largest Immune Cell Cluster
    - Number of Lesion Adjacent Immune Cell Clusters
    - Number of Gland Adjacent Immune Cell Clusters
    - Number of Squam Adjacent Immune Cell Clusters
    - Number of Distal Immune Cell Clusters
    - Clustered Immune Cell Density
    - LII Density
    - LIIs within Lesion Density
    - LIIs at LSI Density
    - Distal Immune Cell Density
    - Lesion Adjacent Clustered Immune Density
    - Gland Adjacent Clustered Immune Density
    - Squam Adjacent Clustered Immune Density
    - Lesion Cluster to Gland Cluster Size Proportion

    Parameters:
        tis (gpd.GeoDataFrame):
            The GeoDataFrame with the tissue areas.
        nuc (gpd.GeoDataFrame):
            The GeoDataFrame with the nuclei.

    Returns:
        pd.Series:
            A pandas Series with the computed features.
    """
    # Get the different tissue areas
    stroma = set_crs(
        tis.loc[tis["class_name"].isin(["areastroma", "blood"])].make_valid()
    )
    tumor = set_crs(tis.loc[tis["class_name"] == "area_cin"].make_valid())

    # Get the different cells from the different tissue areas
    tumor_cells = nuc.loc[nuc["class_name"] == "neoplastic"]
    immune_cells = nuc.loc[nuc["class_name"] == "inflammatory"]

    # Get immune cells in different tissue contexts
    tumor_cells: gpd.GeoDataFrame = get_objs(tumor, tumor_cells)
    LIL_cells: gpd.GeoDataFrame = get_objs(tumor, immune_cells)

    # Cluster the immune cells
    clust_label = density_clustering(
        immune_cells, eps=200, min_samples=25, method="dbscan"
    )
    immune_cells = immune_cells.assign(label=clust_label)
    clust_immune_cells = set_uid(immune_cells[immune_cells["label"] > 0])
    immune_clusts, tumor_iface, squam_iface, gland_iface = context4clusters(
        clust_immune_cells, tis
    )
    # get the iface union to get the distal stroma area
    iface_union = gpd.GeoSeries(
        unary_union(
            [tumor_iface.union_all(), squam_iface.union_all(), gland_iface.union_all()]
        )
    )

    distal_stroma = stroma.geometry.difference(iface_union.union_all())
    distal_stroma = gpd.GeoDataFrame(geometry=distal_stroma, crs=stroma.crs)
    distal_immune_cells = get_objs(distal_stroma, immune_cells)
    LSI_immune_cells = get_objs(tumor_iface, immune_cells)

    # Separate the immune clusters based on their stromal context
    LSI_c_LILS = immune_clusts.loc[
        immune_clusts["stromal_context"] == "adjascent_tumor"
    ]
    GSI_c_LILS = immune_clusts.loc[
        immune_clusts["stromal_context"] == "adjascent_gland"
    ]
    SSI_c_LILS = immune_clusts.loc[
        immune_clusts["stromal_context"] == "adjascent_squam"
    ]
    distal_clust_cells = immune_clusts.loc[immune_clusts["stromal_context"] == "distal"]

    ##########################################################
    # compute features

    # Calculate overall immune cell proportion
    immune_cell_proportion = len(immune_cells) / len(nuc) if len(nuc) > 0 else 0
    distal_immune_to_all_immune_proportion = (
        len(distal_immune_cells) / len(immune_cells) if len(immune_cells) > 0 else 0
    )
    LII_to_all_immune_proportion = (
        (len(LIL_cells) + len(LSI_immune_cells)) / len(immune_cells)
        if len(immune_cells) > 0
        else 0
    )
    LII_to_all_neoplastic_proportion = (
        (len(LIL_cells) + len(LSI_immune_cells)) / len(tumor_cells)
        if len(tumor_cells) > 0
        else 0
    )
    LII_within_lesion_to_all_immune_proportion = (
        len(LIL_cells) / len(immune_cells) if len(immune_cells) > 0 else 0
    )
    LII_within_lesion_to_all_neoplastic_proportion = (
        len(LIL_cells) / len(tumor_cells) if len(tumor_cells) > 0 else 0
    )
    LSI_immune_to_all_immune_proportion = (
        len(LSI_immune_cells) / len(immune_cells) if len(immune_cells) > 0 else 0
    )
    LSI_immune_to_all_neoplastic_proportion = (
        len(LSI_immune_cells) / len(tumor_cells) if len(tumor_cells) > 0 else 0
    )
    LII_within_lesion_to_LSI_immune_proportion = (
        LII_within_lesion_to_all_neoplastic_proportion
        / (
            LII_within_lesion_to_all_neoplastic_proportion
            + LSI_immune_to_all_neoplastic_proportion
        )
    )

    # Calculate clustered immune cell proportions
    clust_immune_proportion = len(immune_clusts) / len(immune_cells)
    tumor_clust_immune_proportion = (
        len(LSI_c_LILS) / len(immune_clusts) if len(immune_clusts) > 0 else 0
    )
    gland_clust_immune_proportion = (
        len(GSI_c_LILS) / len(immune_clusts) if len(immune_clusts) > 0 else 0
    )
    squam_clust_immune_proportion = (
        len(SSI_c_LILS) / len(immune_clusts) if len(immune_clusts) > 0 else 0
    )

    # Calculate iface areas for density normalization
    LSI_area = tumor_iface.area.sum()
    stroma_area = stroma.area.sum()
    tumor_area = tumor.area.sum()
    distal_stroma_area = distal_stroma.area.sum()

    # Get the immune densities
    LIL_within_density = (
        len(LIL_cells) / (tumor_area * TO_MM_SQUARED_CONVERSION)
        if tumor_area > 0
        else 0
    )
    LIL_LSI_density = (
        len(LSI_immune_cells) / (LSI_area * TO_MM_SQUARED_CONVERSION)
        if LSI_area > 0
        else 0
    )
    LIL_density = (
        (len(LIL_cells) + len(LSI_immune_cells))
        / ((tumor_area + LSI_area) * TO_MM_SQUARED_CONVERSION)
        if (tumor_area + LSI_area) > 0
        else 0
    )

    immune_c_density = (
        len(immune_clusts) / (stroma_area * TO_MM_SQUARED_CONVERSION)
        if stroma_area > 0
        else 0
    )
    # Get the immune density to stromal area
    LSI_c_immune_density_to_stroma = (
        len(LSI_c_LILS) / (stroma_area * TO_MM_SQUARED_CONVERSION)
        if stroma_area > 0
        else 0
    )
    GSI_c_immune_density_to_stroma = (
        len(GSI_c_LILS) / (stroma_area * TO_MM_SQUARED_CONVERSION)
        if stroma_area > 0
        else 0
    )
    SSI_c_immune_density_to_stroma = (
        len(SSI_c_LILS) / (stroma_area * TO_MM_SQUARED_CONVERSION)
        if stroma_area > 0
        else 0
    )
    distal_immune_density = (
        len(immune_cells.loc[immune_cells["label"] == -1])
        / (distal_stroma_area * TO_MM_SQUARED_CONVERSION)
        if distal_stroma_area > 0
        else 0
    )

    # Calculate average number of cells per cluster using groupby mean
    if (len(immune_clusts["label"].unique()) - 1) > 0:
        average_number_of_cells_per_cluster = (
            immune_clusts.groupby("label").size()[immune_clusts["label"] > 0].mean()
        )
        std_number_of_cells_per_cluster = (
            immune_clusts.groupby("label").size()[immune_clusts["label"] > 0].std()
        )
        max_number_of_cells_per_cluster = (
            immune_clusts.groupby("label").size()[immune_clusts["label"] > 0].max()
        )
    else:
        average_number_of_cells_per_cluster = 0
        max_number_of_cells_per_cluster = 0
        std_number_of_cells_per_cluster = 0

    # Compute the sum of cluster sizes for lesion-adjacent and gland-adjacent clusters
    lesion_adjacent_sum = LSI_c_LILS.groupby("label").size().sum()
    gland_adjacent_sum = GSI_c_LILS.groupby("label").size().sum()

    if gland_adjacent_sum > 0:
        lesion_to_gland_cluster_size_proportion = lesion_adjacent_sum / (
            lesion_adjacent_sum + gland_adjacent_sum
        )
    else:
        lesion_to_gland_cluster_size_proportion = 0

    res = {
        "Overall Immune Cell Proportion": immune_cell_proportion,
        "Distal Immune to All Immune Proportion": distal_immune_to_all_immune_proportion,
        "LIIs to All Immune Proportion": LII_to_all_immune_proportion,
        "LIIs to All Neoplastic Proportion": LII_to_all_neoplastic_proportion,
        "LIIs within Lesion to All Immune Proportion": LII_within_lesion_to_all_immune_proportion,
        "LIIs within Lesion to All Neoplastic Proportion": LII_within_lesion_to_all_neoplastic_proportion,
        "LIIs at LSI to All Immune Proportion": LSI_immune_to_all_immune_proportion,
        "LIIs at LSI to All Neoplastic Proportion": LSI_immune_to_all_neoplastic_proportion,
        "LIIs within Lesion to All LIIs proportion": LII_within_lesion_to_LSI_immune_proportion,
        "Clustered to All Immune Cells Proportion": clust_immune_proportion,
        "Gland Adjacent to All Clustered Immune Proportion": gland_clust_immune_proportion,
        "Lesion Adjacent to All Clustered Immune Proportion": tumor_clust_immune_proportion,
        "Squam Adjacent to All Clustered Immune Proportion": squam_clust_immune_proportion,
        "Number of Immune Cell Clusters": len(immune_clusts["label"].unique()) - 1,
        "Average Number of Cells per Immune Cell Cluster": average_number_of_cells_per_cluster,
        "Std Number of Cells per Immune Cell Cluster": std_number_of_cells_per_cluster,
        "Number of Cells in Largest Immune Cell Cluster": max_number_of_cells_per_cluster,
        "Number of Lesion Adjacent Immune Cell Clusters": len(
            LSI_c_LILS["label"].unique()
        ),
        "Number of Gland Adjacent Immune Cell Clusters": len(
            GSI_c_LILS["label"].unique()
        ),
        "Number of Squam Adjacent Immune Cell Clusters": len(
            SSI_c_LILS["label"].unique()
        ),
        "Number of Distal Immune Cell Clusters": len(
            distal_clust_cells["label"].unique()
        ),
        "Clustered Immune Cell Density": immune_c_density,
        "LII Density": LIL_density,
        "LIIs within Lesion Density": LIL_within_density,
        "LIIs at LSI Density": LIL_LSI_density,
        "Distal Immune Cell Density": distal_immune_density,
        "Lesion Adjacent Clustered Immune Density": LSI_c_immune_density_to_stroma,
        "Gland Adjacent Clustered Immune Density": GSI_c_immune_density_to_stroma,
        "Squam Adjacent Clustered Immune Density": SSI_c_immune_density_to_stroma,
        "Lesion Cluster to Gland Cluster Size Proportion": lesion_to_gland_cluster_size_proportion,
    }
    return pd.Series(res)


def cluster_stroma_context(
    area_gdf: gpd.GeoDataFrame,
    cluster_gdf: gpd.GeoDataFrame,
    tissue_type: str = "area_cin",
    stroma_types: List[str] = ["areastroma", "blood"],
    buf_dist: int = 500,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Get the stromal context of the given clustered cells.

    The stromal context is defined as:
    - 'distal': The cluster is not in the interface zones.
    - 'adjascent': The cluster is in the immediate interface zone.
    - 'proximal': The cluster is in the further away interface zone.

    Parameters:
        area_gdf (gpd.GeoDataFrame):
            The GeoDataFrame with the areas.
        cluster_gdf (gpd.GeoDataFrame):
            The GeoDataFrame with the clustered cells.
        tissue_type (str):
            The class name of the tissue.
        stroma_types (List[str]):
            The class names of the stroma.
        buf_dist (int):
            The buffer distance for the interface zone.

    Returns:
        clustered_cells (gpd.GeoDataFrame):
            The GeoDataFrame with the clustered cells.
        iface (gpd.GeoDataFrame):
            The GeoDataFrame with the interface zones.
    """
    clustered_cells = cluster_gdf.copy()
    clustered_cells = clustered_cells.assign(stromal_context="distal")

    # Get the tissue and stroma
    tissue = area_gdf.loc[area_gdf["class_name"] == tissue_type]
    tissue.set_crs(4328, inplace=True, allow_override=True)
    tissue = tissue.loc[tissue.area > 1e5]

    stroma = area_gdf.loc[area_gdf["class_name"].isin(stroma_types)]
    stroma.set_crs(4328, inplace=True, allow_override=True)

    # Get the interface zones
    iface = get_interfaces(tissue, stroma, buffer_dist=buf_dist)

    if iface.empty:
        return clustered_cells, iface

    iface = iface.dissolve().explode(index_parts=False)

    # assign the stromal context
    for lab in sorted(clustered_cells["label"].unique()):
        clust = clustered_cells.loc[clustered_cells["label"] == lab]

        # Get the cells in the interface zones
        clust_iface = get_objs(iface, clust)

        # Assign the stromal context, if over 15% cluster extends to the interface zones
        # the stromal context is set to adjascent. Adjascent is the immediate interface
        # zone, If the cluster is not in the interface zones, the stromal context is set
        # to distal.
        if len(clust_iface) / len(clust) > 0.15:
            clust = clust.assign(stromal_context="adjascent")

        clustered_cells.loc[clust.index, "stromal_context"] = clust["stromal_context"]

    return clustered_cells, iface


def context4clusters(
    clustered_cells: gpd.GeoDataFrame,
    area_gdf: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Find the stromal contexts of the cell clusters.

    The stromal contexts are defined as:
    - 'distal': The cluster is not in the interface zones.
    - 'adjascent': The cluster is in the immediate interface zone.
    - 'proximal': The cluster is in the further away interface zone.

    Parameters:
        cell_gdf (gpd.GeoDataFrame):
            The GeoDataFrame with the cells.
        area_gdf (gpd.GeoDataFrame):
            The GeoDataFrame with the areas.

    Returns:
        final_clusts (gpd.GeoDataFrame):
            The GeoDataFrame with the clustered cells and the stromal contexts.
        iface (gpd.GeoDataFrame):
            The GeoDataFrame with the interface zones of the `tissue_type` areas.
    """
    tumor_immune_clusts, tumor_iface = cluster_stroma_context(
        area_gdf=area_gdf,
        cluster_gdf=clustered_cells,
        tissue_type="area_cin",
        stroma_types=["areastroma", "blood"],
        buf_dist=750,
    )

    gland_immune_clusts, gland_iface = cluster_stroma_context(
        area_gdf=area_gdf,
        cluster_gdf=clustered_cells,
        tissue_type="areagland",
        stroma_types=["areastroma", "blood"],
        buf_dist=750,
    )

    squam_immune_clusts, squam_iface = cluster_stroma_context(
        area_gdf=area_gdf,
        cluster_gdf=clustered_cells,
        tissue_type="areasquam",
        stroma_types=["areastroma", "blood"],
        buf_dist=750,
    )

    # Refine the stromal contexts based on the interface zones
    # Start with a copy of tumor_immune_clusts
    final_clusts = tumor_immune_clusts.copy()
    final_clusts.loc[
        final_clusts["stromal_context"] == "adjascent", "stromal_context"
    ] = "adjascent_tumor"

    # Update 'adjascent' values from squam_immune_clusts, tumor adjacent clusters override
    # 'squam' adjacent clusters if there is overlap
    mask = (squam_immune_clusts["stromal_context"] == "adjascent") & (
        final_clusts["stromal_context"] != "adjascent_tumor"
    )
    final_clusts.loc[mask, "stromal_context"] = "adjascent_squam"

    # Update 'adjascent' values from gland_clusts, tumor adjacent clusters override
    # 'gland' adjacent clusters if there is overlap
    mask = (
        (gland_immune_clusts["stromal_context"] == "adjascent")
        & (final_clusts["stromal_context"] != "adjascent_tumor")
        & (final_clusts["stromal_context"] != "adjascent_squam")
    )
    final_clusts.loc[mask, "stromal_context"] = "adjascent_gland"

    # Ensure 'distal' values that are set as 'distal' in all of the three gdfs are
    # left as 'distal'
    mask = (
        (tumor_immune_clusts["stromal_context"] == "distal")
        & (gland_immune_clusts["stromal_context"] == "distal")
        & (squam_immune_clusts["stromal_context"] == "distal")
    )
    final_clusts.loc[mask, "stromal_context"] = "distal"

    # Ensure 'distal' values that are set as 'distal' in all of the three gdfs are
    # left as 'distal'
    final_clusts.loc[mask, "stromal_context"] = "distal"

    return final_clusts, tumor_iface, squam_iface, gland_iface
