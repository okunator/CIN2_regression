import geopandas as gpd
from histolytics.spatial_ops.rect_grid import rect_grid
from histolytics.wsi.slide_reader import SlideReader


def read_data(
    wsi_path: str,
    tissue_annot_path: str,
    nuc_annot_path: str,
) -> tuple[SlideReader, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Read WSI and annotations. Return"""
    reader = SlideReader(wsi_path, backend="OPENSLIDE")
    tis = gpd.read_feather(
        tissue_annot_path,
        columns=["geometry", "class_name"],
    )
    nuc = gpd.read_feather(
        nuc_annot_path,
        columns=["geometry", "class_name"],
    )

    return reader, tis, nuc


def get_grid_and_translate(
    tis: gpd.GeoDataFrame,
    nuc: gpd.GeoDataFrame,
    reader: SlideReader = None,
    patch_size: tuple = (256, 256),
    translate: bool = True,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create a grid of patches from the tissue annotations."""
    grid = rect_grid(tis, patch_size)

    if translate and reader:
        xmin, ymin, _, _ = reader.data_bounds
        grid["geometry"] = grid.translate(xoff=xmin, yoff=ymin)
        tis = tis.assign(geometry=tis.geometry.translate(xoff=xmin, yoff=ymin))
        nuc = nuc.assign(geometry=nuc.geometry.translate(xoff=xmin, yoff=ymin))

    return grid, tis, nuc
