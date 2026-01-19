from __future__ import annotations

from typing import Dict, List

# -------------------------
# Immune response features
# -------------------------
IMMUNE_FEATURES: Dict[str, str] = {
    "immune_cell_propotion": "Immune cells / all cells",
    # Overall lesion-associated immune cells (intralesional + interface)
    "TILS_immune_proportion": "Total LII / all immune",
    "TILS_neoplastic_proportion": "Total LII / neoplastic cells",
    "TIL_cin_density": "Total LII density",
    # Lesion–stroma interface
    "TILS_tsi_TILS_proportion": "Interface LII / total LII",
    "TILS_tsi_immune_proportion": "Interface LII / all immune",
    "TILS_tsi_neoplastic_proportion": "Interface LII / neoplastic cells",
    "TIL_tsi_cin_density": "Interface LII density",
    # Intralesional immune cells (within lesion)
    "TILS_cin_TILS_proportion": "Intralesional LII / total LII",
    "TILS_cin_immune_proportion": "Intralesional LII / all immune",
    "TILS_cin_TILS_tsi_proportion": "Intralesional / interface LII ratio",
    "TILS_cin_neoplastic_proportion": "Intralesional LII / neoplastic cells",
    "TIL_cin_cin_density": "Intralesional LII density",
    # Immune clustering (global)
    "immune_cluster_proportion": "Clustered immune / all immune",
    "immune_cluster_stroma_density": "Clustered immune density (stroma)",
    "n_immune_clusters_total": "Number of immune clusters",
    # Lesion-adjacent clusters
    "tumor_adjascent_immune_cluster_proportion": "Lesion-adjacent clusters / all clusters",
    "tumor_adjascent_immune_cluster_immune_proportion": "Lesion-adjacent immune / all immune",
    "n_tumor_adjascent_immune_clusters": "Lesion-adjacent cluster count",
    "tumor_adjascent_clustered_immune_cells_stroma_density": "Lesion-adjacent clustered immune density",
    # Squamous-adjacent clusters
    "squam_adjascent_immune_cluster_proportion": "Squamous-adjacent clusters / all clusters",
    "squam_adjascent_immune_cluster_immune_proportion": "Squamous-adjacent immune / all immune",
    "n_squam_adjascent_immune_clusters": "Squamous-adjacent cluster count",
    "squam_adjascent_clustered_immune_cells_stroma_density": "Squamous-adjacent clustered immune density",
    # Gland-adjacent clusters
    "gland_adjascent_immune_cluster_proportion": "Gland-adjacent clusters / all clusters",
    "gland_adjascent_immune_cluster_immune_proportion": "Gland-adjacent immune / all immune",
    "n_gland_adjascent_immune_clusters": "Gland-adjacent cluster count",
    "gland_adjascent_clustered_immune_cells_stroma_density": "Gland-adjacent clustered immune density",
    # CIN vs gland cluster comparisons
    "cin_cluster_to_gland_cluster_cellcount_proportion": "CIN/gland cluster cell ratio",
    "cin_cluster_to_gland_cluster_clustercount_proportion": "CIN/gland cluster count ratio",
    # Cluster size
    "mean_immune_cluster_size": "Mean immune cluster size",
    "std_immune_cluster_size": "Immune cluster size variability",
}

# -----------------------------------------
# Neoplastic nuclear and cellular features
# -----------------------------------------
NEOPLASTIC_FEATURES: Dict[str, str] = {
    "neoplastic_chrom_area_mean": "Mean chromatin area",
    "neoplastic_chrom_area_std": "Chromatin area variability",
    "neoplastic_chrom_mean_intensity_mean": "Mean grayscale chromatin intensity",
    "neoplastic_chrom_mean_intensity_std": "Chromatin grayscale intensity variability",
    "neoplastic_chrom_clump_area_nuc_area_ratio_mean": "Mean chromatin-to-nuclear area ratio",
    "neoplastic_chrom_clump_area_nuc_area_ratio_std": "Chromatin-to-nuclear area ratio variability",
    "neoplastic_nuc_mean_intensity_mean": "Mean nuclear grayscale intensity",
    "neoplastic_nuc_mean_intensity_std": "Nuclear grayscale intensity variability",
    "tumor_cells_area_mean": "Mean neoplastic nuclear area",
    "tumor_cells_area_std": "Neoplastic nuclear area variability",
    "tumor_cells_major_axis_len_mean": "Mean major axis length",
    "tumor_cells_major_axis_len_std": "Major axis length variability",
    "tumor_cells_eccentricity_mean": "Mean eccentricity",
    "tumor_cells_eccentricity_std": "Eccentricity variability",
    "tumor_cells_fractal_dimension_mean": "Mean fractal dimension",
    "tumor_cells_fractal_dimension_std": "Fractal dimension variability",
    "tumor_cells_elongation_mean": "Mean elongation",
    "tumor_cells_elongation_std": "Elongation variability",
    "tumor_cells_circularity_mean": "Mean circularity",
    "tumor_cells_circularity_std": "Circularity variability",
}

# -------------------------
# Tissue and lesion features
# -------------------------
TISSUE_FEATURES: Dict[str, str] = {
    "tumor_cell_propotion": "Neoplastic cells / all cells",
    "gland_cell_propotion": "Glandular cells / all cells",
    "squam_cell_propotion": "Squamous cells / all cells",
    "cin_to_biopsy_area_ratio": "Lesion / biopsy area",
    "cin_to_gland_squam_area_ratio": "Lesion / epithelial area",
    "cin_to_gland_area_ratio": "Lesion / glandular area",
    "cin_to_squam_area_ratio": "Lesion / squamous area",
    "squam_to_biopsy_area_ratio": "Squamous / biopsy area",
    "gland_to_biopsy_area_ratio": "Glandular / biopsy area",
    "cin_to_squam_cell_ratio": "Neoplastic / squamous cells",
    "cin_to_gland_cell_ratio": "Neoplastic / glandular cells",
    "cin_to_gland_squam_cell_ratio": "Neoplastic / epithelial cells",
    "area_invaded_gland": "Invaded gland area",
    "invaded_gland_area_proportion": "Invaded gland area proportion",
    "tumor_mean_len_breadth_lines": "Mean lesion breadth",
    "tumor_std_len_breadth_lines": "Lesion breadth variability",
    "tumor_len_medial_line": "Lesion medial line length",
    "area_cin": "Lesion area",
    "area_gland": "Glandular area",
    "area_squam": "Squamous area",
    "cin_eccentricity": "Lesion eccentricity",
    "cin_fractal_dimension": "Lesion fractal dimension",
    "cin_elongation": "Lesion elongation",
    "cin_circularity": "Lesion circularity",
}

# -------------------------
# Clinical / HPV features
# -------------------------
CLINICAL_FEATURES: Dict[str, str] = {
    "clin_size": "Colposcopic lesion size",
    "clin_entrypapa": "Entry Pap smear result",
    "n_hpv_types": "Number of HPV types",
    "has_hpv16": "HPV16 positive",
    "has_hpv31": "HPV31 positive",
}

FEATURE_LABELS: Dict[str, str] = {
    **IMMUNE_FEATURES,
    **NEOPLASTIC_FEATURES,
    **TISSUE_FEATURES,
    **CLINICAL_FEATURES,
}

FEATURE_THEMES: Dict[str, List[str]] = {
    "immune_feats": list(IMMUNE_FEATURES.values()),
    "neo_nuc_feats": list(NEOPLASTIC_FEATURES.values()),
    "tissue_feats": list(TISSUE_FEATURES.values()),
    "clin_feats": list(CLINICAL_FEATURES.values()),
}


def available_feature_themes() -> List[str]:
    return sorted(FEATURE_THEMES.keys())
