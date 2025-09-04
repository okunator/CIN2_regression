from functools import partial
from typing import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from histolytics.nuc_feats.chromatin import chromatin_feats
from histolytics.nuc_feats.intensity import grayscale_intensity_feats
from histolytics.spatial_agg.grid_agg import grid_aggregate
from histolytics.spatial_geom.shape_metrics import shape_metric
from histolytics.utils.gdf import set_crs
from histolytics.wsi.slide_reader import SlideReader
from histolytics.wsi.wsi_processor import WSIGridProcessor
from pandarallel import pandarallel
from tqdm import tqdm

from norm import get_macenko_stain_matrix, normalize_stains
from utils import get_grid_and_translate, read_data


def neoplastic_nuclei_features(
    wsi_path: str,
    tis_path: str,
    nuc_path: str,
    neo_nuc_cls: str = "neoplastic",
    neo_tis_cls: str = "area_cin",
    norm: bool = False,
) -> pd.Series:
    """Run  morphological feature extraction pipeline for one segmented WSI.

    This pipeline extracts following features for the input WSI:
        - Mean Area Neoplastic Nuclei
        - Std Area Neoplastic Nuclei
        - Mean Circularity Neoplastic Nuclei
        - Std Circularity Neoplastic Nuclei
        - Mean Elongation Neoplastic Nuclei
        - Std Elongation Neoplastic Nuclei
        - Mean Compactness Neoplastic Nuclei
        - Std Compactness Neoplastic Nuclei
        - Mean Eccentricity Neoplastic Nuclei
        - Std Eccentricity Neoplastic Nuclei
        - Mean Fractal Dimension Neoplastic Nuclei
        - Std Fractal Dimension Neoplastic Nuclei
        - Mean Convexity Neoplastic Nuclei
        - Std Convexity Neoplastic Nuclei
        - Mean Solidity Neoplastic Nuclei
        - Std Solidity Neoplastic Nuclei
        - Mean Major Axis Length Neoplastic Nuclei
        - Std Major Axis Length Neoplastic Nuclei
        - Mean Minor Axis Length Neoplastic Nuclei
        - Std Minor Axis Length Neoplastic Nuclei
        - Mean Grayscale Intensity Neoplastic Nuclei
        - Std Grayscale Intensity Neoplastic Nuclei
        - Mean Skewness Grayscale Intensity Neoplastic Nuclei
        - Mean Chromatin Clump to Area Proportion Neoplastic Nuclei
        - Std Chromatin Clump to Area Proportion Neoplastic Nuclei
        - Mean Area of Chromatin Clumps Neoplastic Nuclei
        - Std Area of Chromatin Clumps Neoplastic Nuclei
        - Mean Number of Chromatin Clumps Neoplastic Nuclei
        - Std Number of Chromatin Clumps Neoplastic Nuclei
        - Mean Chromatin Clump Manders Colocalization Coefficient Neoplastic Nuclei
        - Std Chromatin Clump Manders Colocalization Coefficient Neoplastic Nuclei

    Note:
        When computing nuclear-level features at the patch-level, some border adjacent
        nuclei will be split across multiple patches. For every split nuclei, the computed
        features are averaged over the split nuclear parts with weights based on the
        area of each part.

    Parameters:
        wsi_path (str): Path to the whole slide image (WSI) file.
        tis_path (str): Path to the tissue annotation file.
        nuc_path (str): Path to the nuclear annotation file.
        neo_nuc_cls (str): Class name for neoplastic nuclei.
        neo_tis_cls (str): Class name for neoplastic tissue.
        norm (bool): Whether to Macenko normalize the image patches during intensity feature extraction.

    Returns:
        pd.Series: A series containing the extracted neoplastic nuclei features.
    """
    reader, tis, nuc = read_data(
        wsi_path=wsi_path,
        tissue_annot_path=tis_path,
        nuc_annot_path=nuc_path,
    )
    tis = set_crs(tis)
    nuc = set_crs(nuc)

    # fit grid and translate (overlay to WSI)
    grid, _, neo_nuc = get_grid_and_translate(
        tis[tis["class_name"] == neo_tis_cls],
        nuc[nuc["class_name"] == neo_nuc_cls],
        reader,
        patch_size=(256, 256),
        translate=True,
    )
    grid = set_crs(grid)

    # compute nuclear and patch level intensity and chromatin clump features for neoplastic nuclei
    nuc_intensity_feats, nuc_chrom_feats, _, _ = nuclear_intensity_pipeline(
        reader=reader, grid=grid, nuc=neo_nuc, num_processes=8, norm=norm
    )

    # compute nuclear and patch level morphological features for neoplastic nuclei
    nuc_morpho_feats, _ = nuclear_morpho_pipeline(neo_nuc, grid, num_processes=8)

    # Compute summary statistics for all nuclear features
    nuclear_intensity_summary = _compute_nuclear_summary_stats(
        nuc_intensity_feats, compute_std=False
    )
    nuclear_chrom_summary = _compute_nuclear_summary_stats(
        nuc_chrom_feats, compute_std=True
    )
    nuclear_morpho_summary = _compute_nuclear_summary_stats(
        nuc_morpho_feats, compute_std=True
    )

    neoplastic_nuc_feats = pd.concat(
        [nuclear_intensity_summary, nuclear_chrom_summary, nuclear_morpho_summary]
    )

    return neoplastic_nuc_feats


def nuclear_intensity_pipeline(
    reader: SlideReader,
    grid: gpd.GeoDataFrame,
    nuc: gpd.GeoDataFrame,
    norm: bool = False,
    num_processes: int = 8,
    pbar: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute nuclear intensity and chromatin clumping features at nuclear and patch levels.

    The computed features are:
        - Mean Grayscale Intensity
        - Std Grayscale Intensity
        - Skewness Grayscale Intensity
        - Chromatin Clump to Nuclei Area Proportion
        - Area of Chromatin Clumps
        - Number of Chromatin Clumps
        - Chromatin Clump Manders Colocalization Coefficient (how much of the total intensity is in the clumps)
        - (Nuclei Area) - for weighting

    Parameters:
        reader (SlideReader): SlideReader instance for reading WSI.
        grid (gpd.GeoDataFrame): GeoDataFrame containing patch geometries.
        nuc (gpd.GeoDataFrame): GeoDataFrame containing nuclear annotations.
        norm (bool): whether to Macenko normalize the image patches.
        num_processes (int): Number of processes to use for parallel processing.
        pbar (bool): Whether to display a progress bar.

    Returns:
        tuple: A tuple containing the following elements:
        - intensity_feats: DataFrame with intensity features at nuclear level
        - chrom_feats: DataFrame with chromatin features at nuclear level
        - patch_intensity_feats: DataFrame with patch-level intensity features
        - patch_chrom_feats: DataFrame with patch-level chromatin features
    """
    # compute intensity features per nuclei
    patch_intensity_feats, intensity_feats = _run_patch_pipeline(
        reader=reader,
        grid=grid,
        nuc=nuc,
        pipeline=partial(_gray_intensity_pipeline, norm=norm),
        num_processes=num_processes,
        pbar=pbar,
    )

    # get mean and std intensity stats for chrom feats normalization
    intensity_mean = np.nanmean(intensity_feats["Mean Grayscale Intensity"])
    intensity_std = np.nanstd(intensity_feats["Mean Grayscale Intensity"])

    # compute chromatin clumping features
    patch_chrom_feats, chrom_feats = _run_patch_pipeline(
        reader=reader,
        grid=grid,
        nuc=nuc,
        pipeline=partial(
            _chromatin_feats_pipeline,
            mean=intensity_mean,  # slide level standardization
            std=intensity_std,
            norm=norm,
        ),
        num_processes=num_processes,
        pbar=pbar,
    )

    return (
        intensity_feats.drop(columns=["index"]),
        chrom_feats.drop(columns=["index"]),
        patch_intensity_feats,
        patch_chrom_feats,
    )


def nuclear_morpho_pipeline(
    nuc: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, num_processes: int = 8
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Nuclear morphometry pipeline at nuclear and patch levels.

    The computed features are:
        - Nuclei Area
        - Nuclei Circularity
        - Nuclei Elongation
        - Nuclei Compactness
        - Nuclei Eccentricity
        - Nuclei Fractal Dimension
        - Nuclei Convexity
        - Nuclei Solidity
        - Nuclei Major Axis Length
        - Nuclei Minor Axis Length

    Parameters:
        nuc (gpd.GeoDataFrame): GeoDataFrame containing nuclear annotations
        grid (gpd.GeoDataFrame): GeoDataFrame containing patch geometries
        num_processes (int): Number of processes to use for parallel processing

    Returns:
        tuple: A tuple containing the following elements:
        - nuc: GeoDataFrame with nuclear morphometry features
        - patches: GeoDataFrame with patch-level morphometry features
    """
    metrics = [
        "area",
        "major_axis_len",
        "minor_axis_len",
        "compactness",
        "circularity",
        "convexity",
        "solidity",
        "elongation",
        "eccentricity",
        "fractal_dimension",
    ]

    # compute shape morphometrics
    nuc = shape_metric(nuc, metrics=metrics, num_processes=num_processes)

    # Generate column names with proper formatting
    new_col_names = []
    for metric in metrics:
        metric_name = metric.replace("_", " ").title()
        new_col_names.extend([f"Mean {metric_name}", f"Std {metric_name}"])

    # compute patch level stats
    patches = grid_aggregate(
        grid,
        nuc,
        partial(_compute_mean_std_patch_stats, metrics=metrics),
        predicate="intersects",
        new_col_names=new_col_names,
    )

    # Remove the original metric columns if they exist, keep only mean_ and std_ columns
    cols_to_keep = ["geometry"] + new_col_names
    patches = patches[[col for col in cols_to_keep if col in patches.columns]]

    return nuc, patches


def _gray_intensity_pipeline(
    img: np.ndarray, label: np.ndarray, mask: np.ndarray, norm: bool = False
):
    """Compute grayscale intensity features for a given image patch.

    The computed features are:
        - Mean Grayscale Intensity
        - Std Grayscale Intensity
        - Skewness Grayscale Intensity
        - (Nuclei Area) - for weighting

    Parameters:
        img (np.ndarray): RGB image patch. (H, W, C)
        label (np.ndarray): nuclei label mask patch. (H, W)
        mask (np.ndarray): tissue mask patch. (H, W)
        norm (bool): whether to Macenko normalize the image patch.

    Returns:
        pd.DataFrame: Nuclear-level intensity features for this patch
    """
    cols = [
        "Mean Grayscale Intensity",
        "Std Grayscale Intensity",
        "Skewness Grayscale Intensity",
        "Nuclei Area",
    ]

    if label is None or np.max(label) == 0:
        return pd.DataFrame(columns=cols)

    # Try to compute features, but handle the case where we can't compute both mean and std
    try:
        if norm:
            stain_mat = get_macenko_stain_matrix(img)
            img = normalize_stains(img, stain_mat)

        gray_feats: pd.DataFrame = grayscale_intensity_feats(
            img, label, ["mean", "std", "skewness"]
        )

        # get nuclei areas
        lab, areas = np.unique(label, return_counts=True)
        lab = lab[1:]  # skip background (label 0)
        areas = areas[1:]

        # set nuc areas to res df
        valid_indices = np.intersect1d(lab, gray_feats.index)
        lab_to_area = dict(zip(lab, areas))
        areas = [lab_to_area[idx] for idx in valid_indices]
        gray_feats.loc[valid_indices, cols[3]] = areas

        gray_feats = gray_feats.rename(
            columns={
                "mean": cols[0],
                "std": cols[1],
                "skewness": cols[2],
            }
        )
    except ValueError as e:
        if "Shape of passed values" in str(e):
            # Try with just mean first, then std if that fails
            try:
                if norm:
                    stain_mat = get_macenko_stain_matrix(img)
                    img = normalize_stains(img, stain_mat)

                lab, areas = np.unique(label, return_counts=True)
                gray_feats = grayscale_intensity_feats(img, label, ["mean"])
                gray_feats = gray_feats.assign(std=0, skewness=0)

                # get nuclei areas
                lab, areas = np.unique(label, return_counts=True)
                lab = lab[1:]  # skip background (label 0)
                areas = areas[1:]

                # set nuc areas to res df
                valid_indices = np.intersect1d(lab, gray_feats.index)
                lab_to_area = dict(zip(lab, areas))
                areas = [lab_to_area[idx] for idx in valid_indices]
                gray_feats.loc[valid_indices, cols[3]] = areas

                gray_feats = gray_feats.rename(
                    columns={
                        "mean": cols[0],
                        "std": cols[1],
                        "skewness": cols[2],
                    }
                )
            except Exception as e:
                return pd.DataFrame(columns=cols)
        else:
            raise e

    features = gray_feats.fillna(0.0)
    return features


def _chromatin_feats_pipeline(
    img: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray,
    mean: float = 0.0,
    std: float = 1.0,
    norm: bool = False,
) -> pd.DataFrame:
    """Compute chromatin clumping features for a given image patch.

    The computed features are:
        - Chromatin Clump to Nuclei Area Proportion
        - Area of Chromatin Clumps
        - Number of Chromatin Clumps
        - Chromatin Clump Manders Colocalization Coefficient (how much of the total intensity is in the clumps)
        - (Nuclei Area) - for weighting

    Parameters:
        img (np.ndarray): RGB image patch. (H, W, C)
        label (np.ndarray): nuclei label mask patch. (H, W)
        mask (np.ndarray): tissue mask patch. (H, W)
        mean (float): mean grayscale intensity
        std (float): standard deviation of grayscale intensity
        norm (bool): whether to Macenko normalize the image patch.

    Returns:
        pd.DataFrame: Nuclear-level chromatin clumping features for this patch.
    """
    cols = [
        "Chromatin Clump to Nuclei Area Proportion",
        "Area of Chromatin Clumps",
        "Number of Chromatin Clumps",
        "Chromatin Clump Manders Colocalization Coefficient",
        "Nuclei Area",
    ]

    if label is None or np.max(label) == 0:
        return pd.DataFrame(columns=cols)

    try:
        if norm:
            stain_mat = get_macenko_stain_matrix(img)
            img = normalize_stains(img, stain_mat)

        chrom_feats: pd.DataFrame = chromatin_feats(
            img,
            label,
            ["chrom_area", "chrom_nuc_prop", "n_chrom_clumps", "manders_coloc_coeff"],
            mean=mean,
            std=std,
        )

        # get nuclei areas
        lab, areas = np.unique(label, return_counts=True)
        lab = lab[1:]  # skip background (label 0)
        areas = areas[1:]

        # set nuc areas to res df
        valid_indices = np.intersect1d(lab, chrom_feats.index)
        lab_to_area = dict(zip(lab, areas))
        areas = [lab_to_area[idx] for idx in valid_indices]
        chrom_feats.loc[valid_indices, cols[4]] = areas

        chrom_feats = chrom_feats.rename(
            columns={
                "chrom_nuc_prop": cols[0],
                "chrom_area": cols[1],
                "n_chrom_clumps": cols[2],
                "manders_coloc_coeff": cols[3],
            }
        )

        features = chrom_feats.fillna(0.0)
        return features

    except Exception as e:
        raise e


def _run_patch_pipeline(
    reader: SlideReader,
    grid: gpd.GeoDataFrame,
    nuc: gpd.GeoDataFrame,
    pipeline: Callable,
    num_processes: int = 8,
    pbar: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline runner for WSI-image patches."""
    pandarallel.initialize(verbose=1, progress_bar=False, nb_workers=num_processes)

    crop_loader = WSIGridProcessor(
        slide_reader=reader,
        grid=grid,
        nuclei=nuc,
        pipeline_func=pipeline,
        batch_size=num_processes,
        num_workers=num_processes,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )

    # loop the patches
    crop_feats = []
    with crop_loader as loader:
        with tqdm(loader, unit="batch", total=len(loader), disable=not pbar) as pbar:
            for _, batch in enumerate(pbar):
                crop_feats.extend(batch)  # collect the patch level dfs

    # patch level features
    patch_data = []
    for patch_idx, df in crop_feats:
        if len(df) > 0:
            patch_means = df.mean() * np.sqrt(len(df))  # weight based on df size
            patch_means.name = patch_idx
            patch_data.append(patch_means)
        else:
            patch_data.append(
                pd.Series({col: 0.0 for col in df.columns}, name=patch_idx)
            )

    patch_feats = pd.DataFrame(patch_data)

    # compute features at nuclear level. Split nuclei feature values will be weighted
    # by area so that the larger nuclear parts have more influence
    nuc_feats = pd.concat([df for _, df in crop_feats], axis=0)
    nuc_feats.reset_index(drop=False, inplace=True)
    nuc_feats = nuc_feats.groupby("index").parallel_apply(
        _compute_area_weighted_nuc_stats
    )

    return patch_feats, nuc_feats


def _compute_mean_std_patch_stats(
    nuc: gpd.GeoDataFrame, metrics: list[str]
) -> pd.Series:
    """Compute mean and std for each metric column from nuclei within a grid cell."""
    stats = {}
    for col in metrics:
        # Create formatted column names
        metric_name = col.replace("_", " ").title()
        mean_col = f"Mean {metric_name}"
        std_col = f"Std {metric_name}"

        if col in nuc.columns and len(nuc) > 0:
            values = nuc[col].dropna()
            if len(values) > 0:
                stats[mean_col] = values.mean()
                stats[std_col] = values.std() if len(values) > 1 else 0.0
            else:
                stats[mean_col] = 0.0
                stats[std_col] = 0.0
        else:
            stats[mean_col] = 0.0
            stats[std_col] = 0.0
    return pd.Series(stats)


def _compute_area_weighted_nuc_stats(group):
    """Compute area-weighted statistics for split nuclei."""
    total_area = group["Nuclei Area"].sum()
    numeric_cols = group.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != "Nuclei Area"]

    if total_area == 0:
        result = {col: 0.0 for col in feature_cols}
        result["Nuclei Area"] = 0.0
        return pd.Series(result)

    # Compute area-weighted mean for each feature column
    result = {}
    for col in feature_cols:
        weighted_mean = (group[col] * group["Nuclei Area"]).sum() / total_area
        result[col] = weighted_mean

    result["Nuclei Area"] = total_area

    return pd.Series(result)


def _compute_nuclear_summary_stats(nuc_feats, compute_std: bool = True) -> pd.Series:
    """Compute mean and std for each column in nuclear features dataframes.

    Parameters:
        nuc_feats (list of pd.DataFrame): List of dataframes containing nuclear features.
        compute_std (bool): Whether to compute standard deviation. Defaults to True.

    Returns:
        pd.Series: Sample-level feature vector for nuclear features
    """
    all_stats = []

    # Get numeric columns only
    numeric_cols = nuc_feats.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col != " Area":
            clean_name = col.replace("_", " ").title()
            clean_name += " Neoplastic "

            mean_val = nuc_feats[col].mean()
            mean_name = f"Mean {clean_name}" if "Mean" not in clean_name else clean_name

            if compute_std:
                std_val = nuc_feats[col].std()
                std_name = (
                    f"Std {clean_name}" if "Std" not in clean_name else clean_name
                )

            # Add to results
            all_stats.extend(
                [pd.Series({mean_name: mean_val})]
                + ([pd.Series({std_name: std_val})] if compute_std else [])
            )

    # Concatenate all stats into one series
    result = pd.concat(all_stats)
    return result
