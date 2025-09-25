import argparse
from pathlib import Path

import pandas as pd

# Define aggregation dictionary for feature aggregation
agg_dict = {
    "Mean Grayscale Intensity Neoplastic Nuclei": "mean",
    "Mean Std Grayscale Intensity Neoplastic Nuclei": "mean",
    "Mean Skewness Grayscale Intensity Neoplastic Nuclei": "mean",
    "Mean Area Of Chromatin Clumps Neoplastic Nuclei": "mean",
    "Std Area Of Chromatin Clumps Neoplastic Nuclei": "mean",
    "Mean Chromatin Clump To Nuclei Area Proportion Neoplastic Nuclei": "mean",
    "Std Chromatin Clump To Nuclei Area Proportion Neoplastic Nuclei": "mean",
    "Mean Number Of Chromatin Clumps Neoplastic Nuclei": "mean",
    "Std Number Of Chromatin Clumps Neoplastic Nuclei": "mean",
    "Mean Chromatin Clump Manders Colocalization Coefficient Neoplastic Nuclei": "mean",
    "Std Chromatin Clump Manders Colocalization Coefficient Neoplastic Nuclei": "mean",
    "Mean Area Neoplastic Nuclei": "mean",
    "Std Area Neoplastic Nuclei": "mean",
    "Mean Major Axis Len Neoplastic Nuclei": "mean",
    "Std Major Axis Len Neoplastic Nuclei": "mean",
    "Mean Minor Axis Len Neoplastic Nuclei": "mean",
    "Std Minor Axis Len Neoplastic Nuclei": "mean",
    "Mean Compactness Neoplastic Nuclei": "mean",
    "Std Compactness Neoplastic Nuclei": "mean",
    "Mean Circularity Neoplastic Nuclei": "mean",
    "Std Circularity Neoplastic Nuclei": "mean",
    "Mean Convexity Neoplastic Nuclei": "mean",
    "Std Convexity Neoplastic Nuclei": "mean",
    "Mean Solidity Neoplastic Nuclei": "mean",
    "Std Solidity Neoplastic Nuclei": "mean",
    "Mean Elongation Neoplastic Nuclei": "mean",
    "Std Elongation Neoplastic Nuclei": "mean",
    "Mean Eccentricity Neoplastic Nuclei": "mean",
    "Std Eccentricity Neoplastic Nuclei": "mean",
    "Mean Fractal Dimension Neoplastic Nuclei": "mean",
    "Std Fractal Dimension Neoplastic Nuclei": "mean",
    "Lesion Medial Line Length": "max",
    "Lesion Mean Depth": "mean",
    "Lesion Median Depth": "mean",
    "Lesion Std Depth": "mean",
    "Lesion Compactness": "mean",
    "Lesion Circularity": "mean",
    "Lesion Convexity": "mean",
    "Lesion Elongation": "mean",
    "Lesion Eccentricity": "mean",
    "Lesion Fractal_Dimension": "mean",
    "Lesion Total Area": "mean",
    "Gland Area": "mean",
    "Squamous Area": "mean",
    "Invaded Gland Area": "mean",
    "Gland to Biopsy Area Proportion": "mean",
    "Squamous to Biopsy Area Proportion": "mean",
    "Lesion to Biopsy Area Proportion": "mean",
    "Lesion to Gland Area Proportion": "mean",
    "Lesion to Squamous Area Proportion": "mean",
    "Lesion to Healthy Epithelial Area Proportion": "mean",
    "Invaded Gland to Healthy Epithelial Area Proportion": "mean",
    "Invaded Gland to All Gland Area Proportion": "mean",
    "Neoplastic Nuclei Proportion": "mean",
    "Squamous Nuclei Proportion": "mean",
    "Glandular Nuclei Proportion": "mean",
    "Neoplastic to Glandular Nuclei Proportion": "mean",
    "Neoplastic to Healthy Epithelial Nuclei Proportion": "mean",
    "Overall Immune Cell Proportion": "mean",
    "Distal Immune to All Immune Proportion": "mean",
    "LIIs to All Immune Proportion": "mean",
    "LIIs to All Neoplastic Proportion": "mean",
    "LIIs within Lesion to All Immune Proportion": "mean",
    "LIIs within Lesion to All Neoplastic Proportion": "mean",
    "LIIs at LSI to All Immune Proportion": "mean",
    "LIIs at LSI to All Neoplastic Proportion": "mean",
    "LIIs within Lesion to All LIIs proportion": "mean",
    "Clustered to All Immune Cells Proportion": "mean",
    "Gland Adjacent to All Clustered Immune Proportion": "mean",
    "Lesion Adjacent to All Clustered Immune Proportion": "mean",
    "Squam Adjacent to All Clustered Immune Proportion": "mean",
    "Number of Immune Cell Clusters": "sum",
    "Average Number of Cells per Immune Cell Cluster": "mean",
    "Std Number of Cells per Immune Cell Cluster": "mean",
    "Number of Cells in Largest Immune Cell Cluster": "max",
    "Number of Lesion Adjacent Immune Cell Clusters": "sum",
    "Number of Gland Adjacent Immune Cell Clusters": "sum",
    "Number of Squam Adjacent Immune Cell Clusters": "sum",
    "Number of Distal Immune Cell Clusters": "sum",
    "Clustered Immune Cell Density": "mean",
    "LII Density": "mean",
    "LIIs within Lesion Density": "mean",
    "LIIs at LSI Density": "mean",
    "Distal Immune Cell Density": "mean",
    "Lesion Adjacent Clustered Immune Density": "mean",
    "Gland Adjacent Clustered Immune Density": "mean",
    "Squam Adjacent Clustered Immune Density": "mean",
    "Lesion Cluster to Gland Cluster Size Proportion": "mean",
}


def main(result_folder: Path, output_csv: Path = None, patient_col: str = "patient_id"):
    """Aggregate sample-level features to patient-level."""
    csv_files = list(result_folder.glob("*.csv"))
    dfs = [pd.read_csv(f) for f in csv_files]
    df_all = pd.concat(dfs, ignore_index=True)

    df_patient = df_all.groupby(patient_col).agg(agg_dict).reset_index()

    if output_csv is None:
        output_csv = result_folder.parent / "extracted_sample_feats_agg.csv"
    df_patient.to_csv(output_csv, index=False)
    print(f"Aggregated features saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate sample-level features to patient-level."
    )
    parser.add_argument(
        "--result_folder",
        type=str,
        required=True,
        help="Folder containing sample-level CSVs",
    )
    parser.add_argument(
        "--output_csv", type=str, default=None, help="Output CSV file path (optional)"
    )
    args = parser.parse_args()
    main(Path(args.result_folder), Path(args.output_csv) if args.output_csv else None)
