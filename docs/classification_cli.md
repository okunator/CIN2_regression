# CIN2 Classification CLI

## Overview
This command-line tool reproduces the classification workflow from the manuscript using a **single input dataset**. The input file must contain patient IDs, outcomes, and all required feature columns (histological, clinical, HPV). The tool runs repeated stratified cross-validation, saves performance plots, and computes SHAP-based feature rankings.

## Quick start
- Prepare a single dataset as CSV or Parquet with required columns listed below.
- Run the CLI using the entrypoint or Python module.

Example usage (entrypoint):
```
cin2-classify --input-data /path/to/your_dataset.csv --output-dir /path/to/output
```

Example usage (module):
```
python -m src.classify --input-data /path/to/your_dataset.csv
```

## Input data requirements
Your dataset must contain the following columns:

### Required identifiers and outcome
- Patient/sample identifier: default column name is dg_sample_number.
- Outcome: default column name is outcome.
  - Must be a binary label (default positive class: reg, negative class: non-reg).
  - If the dataset uses different labels, pass --positive-class and --negative-class.

### Feature columns
The tool uses four feature themes. You can run any subset via --feature-themes.

By default, the tool **renames raw feature columns to manuscript labels**. If your dataset already contains the manuscript labels, run with --no-apply-feature-mapping.

#### Clinical and HPV features (clin_feats)
| Raw column name | Manuscript label |
| --- | --- |
| clin_size | Colposcopic lesion size |
| clin_entrypapa | Entry Pap smear result |
| n_hpv_types | Number of HPV types |
| has_hpv16 | HPV16 positive |
| has_hpv31 | HPV31 positive |

#### Neoplastic nuclear features (neo_nuc_feats)
| Raw column name | Manuscript label |
| --- | --- |
| neoplastic_chrom_area_mean | Mean chromatin area |
| neoplastic_chrom_area_std | Chromatin area variability |
| neoplastic_chrom_mean_intensity_mean | Mean grayscale chromatin intensity |
| neoplastic_chrom_mean_intensity_std | Chromatin grayscale intensity variability |
| neoplastic_chrom_clump_area_nuc_area_ratio_mean | Mean chromatin-to-nuclear area ratio |
| neoplastic_chrom_clump_area_nuc_area_ratio_std | Chromatin-to-nuclear area ratio variability |
| neoplastic_nuc_mean_intensity_mean | Mean nuclear grayscale intensity |
| neoplastic_nuc_mean_intensity_std | Nuclear grayscale intensity variability |
| tumor_cells_area_mean | Mean neoplastic nuclear area |
| tumor_cells_area_std | Neoplastic nuclear area variability |
| tumor_cells_major_axis_len_mean | Mean major axis length |
| tumor_cells_major_axis_len_std | Major axis length variability |
| tumor_cells_eccentricity_mean | Mean eccentricity |
| tumor_cells_eccentricity_std | Eccentricity variability |
| tumor_cells_fractal_dimension_mean | Mean fractal dimension |
| tumor_cells_fractal_dimension_std | Fractal dimension variability |
| tumor_cells_elongation_mean | Mean elongation |
| tumor_cells_elongation_std | Elongation variability |
| tumor_cells_circularity_mean | Mean circularity |
| tumor_cells_circularity_std | Circularity variability |

#### Tissue and lesion features (tissue_feats)
| Raw column name | Manuscript label |
| --- | --- |
| tumor_cell_propotion | Neoplastic cells / all cells |
| gland_cell_propotion | Glandular cells / all cells |
| squam_cell_propotion | Squamous cells / all cells |
| cin_to_biopsy_area_ratio | Lesion / biopsy area |
| cin_to_gland_squam_area_ratio | Lesion / epithelial area |
| cin_to_gland_area_ratio | Lesion / glandular area |
| cin_to_squam_area_ratio | Lesion / squamous area |
| squam_to_biopsy_area_ratio | Squamous / biopsy area |
| gland_to_biopsy_area_ratio | Glandular / biopsy area |
| cin_to_squam_cell_ratio | Neoplastic / squamous cells |
| cin_to_gland_cell_ratio | Neoplastic / glandular cells |
| cin_to_gland_squam_cell_ratio | Neoplastic / epithelial cells |
| area_invaded_gland | Invaded gland area |
| invaded_gland_area_proportion | Invaded gland area proportion |
| tumor_mean_len_breadth_lines | Mean lesion breadth |
| tumor_std_len_breadth_lines | Lesion breadth variability |
| tumor_len_medial_line | Lesion medial line length |
| area_cin | Lesion area |
| area_gland | Glandular area |
| area_squam | Squamous area |
| cin_eccentricity | Lesion eccentricity |
| cin_fractal_dimension | Lesion fractal dimension |
| cin_elongation | Lesion elongation |
| cin_circularity | Lesion circularity |

#### Immune response features (immune_feats)
| Raw column name | Manuscript label |
| --- | --- |
| immune_cell_propotion | Immune cells / all cells |
| TILS_immune_proportion | Total LII / all immune |
| TILS_neoplastic_proportion | Total LII / neoplastic cells |
| TIL_cin_density | Total LII density |
| TILS_tsi_TILS_proportion | Interface LII / total LII |
| TILS_tsi_immune_proportion | Interface LII / all immune |
| TILS_tsi_neoplastic_proportion | Interface LII / neoplastic cells |
| TIL_tsi_cin_density | Interface LII density |
| TILS_cin_TILS_proportion | Intralesional LII / total LII |
| TILS_cin_immune_proportion | Intralesional LII / all immune |
| TILS_cin_TILS_tsi_proportion | Intralesional / interface LII ratio |
| TILS_cin_neoplastic_proportion | Intralesional LII / neoplastic cells |
| TIL_cin_cin_density | Intralesional LII density |
| immune_cluster_proportion | Clustered immune / all immune |
| immune_cluster_stroma_density | Clustered immune density (stroma) |
| n_immune_clusters_total | Number of immune clusters |
| tumor_adjascent_immune_cluster_proportion | Lesion-adjacent clusters / all clusters |
| tumor_adjascent_immune_cluster_immune_proportion | Lesion-adjacent immune / all immune |
| n_tumor_adjascent_immune_clusters | Lesion-adjacent cluster count |
| tumor_adjascent_clustered_immune_cells_stroma_density | Lesion-adjacent clustered immune density |
| squam_adjascent_immune_cluster_proportion | Squamous-adjacent clusters / all clusters |
| squam_adjascent_immune_cluster_immune_proportion | Squamous-adjacent immune / all immune |
| n_squam_adjascent_immune_clusters | Squamous-adjacent cluster count |
| squam_adjascent_clustered_immune_cells_stroma_density | Squamous-adjacent clustered immune density |
| gland_adjascent_immune_cluster_proportion | Gland-adjacent clusters / all clusters |
| gland_adjascent_immune_cluster_immune_proportion | Gland-adjacent immune / all immune |
| n_gland_adjascent_immune_clusters | Gland-adjacent cluster count |
| gland_adjascent_clustered_immune_cells_stroma_density | Gland-adjacent clustered immune density |
| cin_cluster_to_gland_cluster_cellcount_proportion | CIN/gland cluster cell ratio |
| cin_cluster_to_gland_cluster_clustercount_proportion | CIN/gland cluster count ratio |
| mean_immune_cluster_size | Mean immune cluster size |
| std_immune_cluster_size | Immune cluster size variability |

## CLI parameters (high level)
- --input-data: Required path to CSV or Parquet file.
- --output-dir: Output directory for plots and results.
- --feature-themes: Which feature groups to run (default clin_feats and neo_nuc_feats). Use all to run all four themes.
- --apply-feature-mapping / --no-apply-feature-mapping: Rename raw columns to manuscript labels.
- --positive-class / --negative-class: Labels for the binary outcome.
- --n-splits / --n-repeats: CV settings.
- --compute-shap-for-all-folds: If enabled, SHAP is computed for every fold.

## Output
Outputs are saved under the output directory in subfolders per feature theme, including:
- perf_metrics.csv
- ROC and PRC curves
- Confusion matrices
- SHAP beeswarm plots
- A results.pkl file containing all intermediate outputs

## Feature interactions (post-selection)
After running the classifier and feature selection, you can quantify interactions among the selected features.

Example:
```
python -m src.feature_interactions \
  --input-data /path/to/your_dataset.csv \
  --features Colposcopic\ lesion\ size HPV16\ positive \
  --output-dir /path/to/interaction_output
```

Alternatively, provide a file with one feature name per line:
```
python -m src.feature_interactions \
  --input-data /path/to/your_dataset.csv \
  --features-file /path/to/features.txt \
  --output-dir /path/to/interaction_output
```

This script uses FACET. If missing, install:
```
pip install facet-ml sklearndf
```

## Troubleshooting
- Missing columns: Check the required columns and ensure feature mapping is applied correctly.
- Label errors: Ensure your outcome column includes the specified positive and negative labels.
- GPU issues: Use --num-gpus 0 to force CPU execution.
