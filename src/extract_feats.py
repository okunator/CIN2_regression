import warnings
from pathlib import Path

import pandas as pd
from histolytics.utils.gdf import set_crs
from tqdm import tqdm

from features.epithelial_structure_feats import epithelial_structures_pipeline
from features.immune_feats import immune_feat_pipe
from features.neoplastic_nuclei_feats import neoplastic_nuclei_feat_pipeline
from utils import get_grid_and_translate, read_data


def extract_features(wsi_path: str, tis_path: str, nuc_path: str) -> pd.DataFrame:
    """Extract features from a WSI given paths to the WSI and its annotations.

    Parameters:
        wsi_path (str): Path to the WSI file.
        tis_path (str): Path to the tissue annotation file.
        nuc_path (str): Path to the nuclei annotation file.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted features for one sample.
    """
    reader, tis, nuc = read_data(wsi_path, tis_path, nuc_path)

    # filter to neoplastic tissue and neoplastic nuclei for neoplastic nuclei features
    neo_tis = tis[tis["class_name"] == "area_cin"]
    neo_nuc = nuc[nuc["class_name"] == "neoplastic"]
    grid, neo_tis, neo_nuc = get_grid_and_translate(
        neo_tis, neo_nuc, reader, patch_size=(512, 512)
    )
    grid = set_crs(grid)

    # extract neo nuc features, immune features, epithelial features
    neoplastic_feats = neoplastic_nuclei_feat_pipeline(
        reader,
        neo_nuc,
        grid,
    )
    immune_feats = immune_feat_pipe(tis, nuc)
    epithelial_feats = epithelial_structures_pipeline(tis, nuc)

    df = pd.DataFrame(
        pd.concat([neoplastic_feats, epithelial_feats[0], immune_feats])
    ).T
    df.insert(0, "sample", Path(wsi_path).stem)

    return df


def main(
    wsi_folder: Path, nuc_folder: Path, tis_folder: Path, result_folder: Path
) -> None:
    """Main function to extract features from WSIs in a folder.

    Parameters:
        wsi_folder (Path): Path to folder with WSI files.
        nuc_folder (Path): Path to folder with nuclei annotation files.
        tis_folder (Path): Path to folder with tissue annotation files.
        result_folder (Path): Path to output folder for feature CSVs.
    """
    warnings.filterwarnings("ignore")
    result_folder.mkdir(parents=True, exist_ok=True)

    # Accept any file extension for WSI files
    wsi_files: list[Path] = []
    for ext in ["*.*"]:
        wsi_files.extend(wsi_folder.glob(ext))
    wsi_files = sorted([f for f in wsi_files if f.is_file()])

    failed_samples: list[str] = []

    seg_exts = ["feather", "parquet", "geojson"]

    for wsi_path in tqdm(wsi_files, desc="Extracting features", unit="slide"):
        stem: str = wsi_path.stem
        nuc_candidates: list[Path] = []
        tis_candidates: list[Path] = []
        for ext in seg_exts:
            nuc_candidates.extend(nuc_folder.glob(f"*{stem}*.{ext}"))
            tis_candidates.extend(tis_folder.glob(f"*{stem}*.{ext}"))

        if not nuc_candidates or not tis_candidates:
            tqdm.write(f"Missing files for: {stem}")
            failed_samples.append(stem)
            continue

        nuc_path: Path = nuc_candidates[0]
        tis_path: Path = tis_candidates[0]

        try:
            df: pd.DataFrame = extract_features(
                str(wsi_path), str(tis_path), str(nuc_path)
            )
            out_path: Path = result_folder / f"{stem}_features.csv"
            df.to_csv(out_path, index=False)
        except Exception as e:
            tqdm.write(f"Failed: {stem} - {e}")
            failed_samples.append(stem)

    tqdm.write(f"Failed samples: {failed_samples}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract features from WSI, nuclei, and tissue segmentations."
    )
    parser.add_argument(
        "--wsi_folder",
        type=str,
        required=True,
        help="Path to folder with WSI files (*.mrxs)",
    )
    parser.add_argument(
        "--nuc_folder",
        type=str,
        required=True,
        help="Path to folder with nuclei feather files",
    )
    parser.add_argument(
        "--tis_folder",
        type=str,
        required=True,
        help="Path to folder with tissue feather files",
    )
    parser.add_argument(
        "--result_folder",
        type=str,
        required=True,
        help="Path to output folder for feature CSVs",
    )
    args = parser.parse_args()

    main(
        Path(args.wsi_folder),
        Path(args.nuc_folder),
        Path(args.tis_folder),
        Path(args.result_folder),
    )
