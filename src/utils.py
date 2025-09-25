import geopandas as gpd
from histolytics.spatial_ops.rect_grid import rect_grid
from histolytics.utils.gdf import set_crs, set_uid
from histolytics.wsi.slide_reader import SlideReader


def read_data(
    wsi_path: str,
    tissue_annot_path: str,
    nuc_annot_path: str,
    backend: str = "OPENSLIDE",
) -> tuple[SlideReader, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Read WSI and annotations.

    Parameters:
        wsi_path (str): Path to the WSI file.
        tissue_annot_path (str): Path to the tissue annotation file.
        nuc_annot_path (str): Path to the nuclei annotation file.
        backend (str): Backend to use for reading the WSI. Default is "OPENSLIDE".

    Returns:
        tuple[SlideReader, gpd.GeoDataFrame, gpd.GeoDataFrame]:
            The SlideReader object, tissue annotations GeoDataFrame, nuclei annotations GeoDataFrame.
    """
    reader = SlideReader(wsi_path, backend=backend)

    def read_gdf(path: str) -> gpd.GeoDataFrame:
        if path.endswith(".feather"):
            return gpd.read_feather(path, columns=["geometry", "class_name"])
        elif path.endswith(".parquet"):
            return gpd.read_parquet(path, columns=["geometry", "class_name"])
        elif path.endswith(".geojson"):
            return gpd.read_file(path)[["geometry", "class_name"]]
        else:
            raise ValueError(f"Unsupported annotation file format: {path}")

    tis = read_gdf(tissue_annot_path)
    nuc = read_gdf(nuc_annot_path)

    tis = tis.loc[~tis["class_name"].isna()]
    nuc = nuc.loc[~nuc["class_name"].isna()]

    # Explode multipolygons if present
    if "MultiPolygon" in tis.geometry.geom_type.unique():
        tis = tis.explode(index_parts=False, ignore_index=True)
    if "MultiPolygon" in nuc.geometry.geom_type.unique():
        nuc = nuc.explode(index_parts=False, ignore_index=True)

    return reader, set_uid(set_crs(tis)), set_uid(set_crs(nuc))


def get_grid_and_translate(
    tis: gpd.GeoDataFrame,
    nuc: gpd.GeoDataFrame,
    reader: SlideReader = None,
    patch_size: tuple = (256, 256),
    translate: bool = True,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create a grid of patches from the tissue annotations.

    Note:
        The grid can be translated to the WSI coordinates if a SlideReader is provided.
    """
    grid = rect_grid(tis, patch_size)

    if translate and reader:
        xmin, ymin, _, _ = reader.data_bounds
        grid["geometry"] = grid.translate(xoff=xmin, yoff=ymin)
        tis = tis.assign(geometry=tis.geometry.translate(xoff=xmin, yoff=ymin))
        nuc = nuc.assign(geometry=nuc.geometry.translate(xoff=xmin, yoff=ymin))

    return grid, tis, nuc
