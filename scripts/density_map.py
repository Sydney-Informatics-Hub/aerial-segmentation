# -*- coding: utf-8 -*-
import logging

import geopandas as gpd

# import rasterio as rio
import numpy as np
from shapely.geometry import Polygon

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def storey_averager(annotation, storey_column="storeys"):
    """This function will get the average number of storeys of buildings in the
    annotation geodataframe.

    Args:
        annotation (geodataframe): A geodataframe of annotations.

    Returns:
        average_storeys (int): The average number of storeys of buildings in the annotation geodataframe.
    """

    for i, row in annotation.iterrows():
        try:
            if (
                row[storey_column] == "None"
                or row[storey_column] == "0"
                or row[storey_column] == 0
            ):
                annotation.drop(i, inplace=True)
        except KeyError:
            pass
    try:
        average_storeys = annotation[storey_column].mean()
    except KeyError:
        average_storeys = 1
        logger.warning(
            "No storeys column found in the annotation geodataframe. Will assume that all buildings have 1 storey."
        )

    return average_storeys


def density_estimate_combined_area(
    annotation,
    crs=None,
    average_storeys=None,
    footprint_ratio: float = 0.5,
    storey_column: str = "storeys",
) -> float:
    """This function will get the area of annotation geodataframe, and also get
    the number of geometries in the annotation geodataframe, and return a
    number that is the aera of the annotation geodataframe divided by the
    number of geometries in the annotation geodataframe.

    Args:
        annotation (geodataframe): A geodataframe of annotations.
        crs (str): The crs of the annotation geodataframe to calculate the area. If none is given, will rely on UTM crs. If fails, will use 'EPSG:3857' as fallback. Defaults to None.
        average_storeys (int): The average number of storeys of buildings in the annotation geodataframe. If None, will not calculate the average number of storeys using the meta data. Defaults to None.
        footprint_ratio (float): The ratio of the footprint-area-based density to number-based density calculations. It should be a number between 0 and 1. 0 means the footprint area density won't be considered and 1 means number density won't be considered. Defaults to 0.5.

    Returns:
        density (float): The area of the annotation geodataframe divided by the number of geometries in the annotation geodataframe.
    """

    assert (
        footprint_ratio >= 0 and footprint_ratio <= 1
    ), "footprint_ratio must be between 0 and 1"

    density_area = density_estimate_area_area(
        annotation, crs, average_storeys, storey_column
    )

    density_number = density_estimate_number_area(
        annotation, crs, average_storeys, storey_column
    )

    density = (
        density_area * footprint_ratio + density_number * (1 - footprint_ratio)
    ) / 2

    return density


def density_estimate_number_area(
    annotation, crs=None, average_storeys=None, storey_column: str = "storeys"
) -> float:
    """This function will get the area of annotation geodataframe, and also get
    the number of geometries in the annotation geodataframe, and return a
    number that is the aera of the annotation geodataframe divided by the
    number of geometries in the annotation geodataframe.

    Args:
        annotation (geodataframe): A geodataframe of annotations.
        crs (str): The crs of the annotation geodataframe to calculate the area. If none is given, will rely on UTM crs. If fails, will use 'EPSG:3857' as fallback. Defaults to None.
        average_storeys (int): The average number of storeys of buildings in the annotation geodataframe. If None, will not calculate the average number of storeys using the meta data. Defaults to None.

    Returns:
        density (float): The area of the annotation geodataframe divided by the number of geometries in the annotation geodataframe.
    """

    if crs is None:
        try:
            crs = annotation.estimalte_utm_crs()
        except Exception as e:
            crs = "EPSG:3857"
            logger.warning(
                f"Could not estimate the UTM crs of the annotation geodataframe. Will use {crs} as fallback."
            )
            print(e)

    if annotation.crs != crs:
        annotation = annotation.to_crs(crs)

    if average_storeys is None:
        average_storeys = storey_averager(annotation, storey_column)
        logger.info(
            f"Average storeys is {average_storeys} for the given extent {annotation.bounds}"
        )
    elif average_storeys == 0:
        average_storeys = 1
        logger.warning(
            "Average storeys cannot be 0. Will assume that all buildings have 1 storey."
        )
    else:
        average_storeys = int(average_storeys)
        logger.info(
            f"Average storeys is {average_storeys} for the given extent {annotation.bounds}"
        )

    area = annotation.area
    number = annotation.shape[0]

    density = (number * average_storeys) / area

    return density


def density_estimate_area_area(
    annotation, crs=None, average_storeys=None, storey_column: str = "storeys"
) -> float:
    """This function will get the area of annotation geodataframe, and also get
    the number of geometries in the annotation geodataframe, and return a
    number that is the aera of the annotation geodataframe divided by the
    number of geometries in the annotation geodataframe.

    Args:
        annotation (geodataframe): A geodataframe of annotations.
        crs (str): The crs of the annotation geodataframe to calculate the area. If none is given, will rely on UTM crs. If fails, will use 'EPSG:3857' as fallback. Defaults to None.
        average_storeys (int): The average number of storeys of buildings in the annotation geodataframe. If None, will not calculate the average number of storeys using the meta data. Defaults to None.

    Returns:
        density (float): The area of the annotation geodataframe divided by the number of geometries in the annotation geodataframe.
    """
    if crs is None:
        try:
            crs = annotation.estimalte_utm_crs()
        except Exception as e:
            crs = "EPSG:3857"
            logger.warning(
                f"Could not estimate the UTM crs of the annotation geodataframe. Will use {crs} as fallback."
            )
            print(e)

    if annotation.crs != crs:
        annotation = annotation.to_crs(crs)

    if average_storeys is None:
        average_storeys = storey_averager(annotation, storey_column)
        logger.info(
            f"Average storeys is {average_storeys} for the given extent {annotation.bounds}"
        )
    elif average_storeys == 0:
        average_storeys = 1
        logger.warning(
            "Average storeys cannot be 0. Will assume that all buildings have 1 storey."
        )
    else:
        average_storeys = int(average_storeys)
        logger.info(
            f"Average storeys is {average_storeys} for the given extent {annotation.bounds}"
        )

    area = annotation.area

    footprint_area = 0
    for _, row in annotation.iterrows():
        footprint_area += row["geometry"].area

    density = (footprint_area * average_storeys) / area

    return density


def density_map_maker(
    gdf,
    average_storeys: int = None,
    footprint_ratio: float = 0.5,
    tile_size: int = 100,
    size_unit: str = None,
    area_unit: str = "utm",
):
    """This function will use the density_estimate_combined_area function to
    create a density map, by tiling the geojson.

    Args:
        gdf (geodataframe): A geodataframe of annotations.
        average_storeys (int): The average number of storeys of buildings in the annotation geodataframe. If None, will not calculate the average number of storeys using the meta data. Defaults to None.
        footprint_ratio (float): The ratio of the footprint-area-based density to number-based density calculations. It should be a number between 0 and 1. 0 means the footprint area density won't be considered and 1 means number density won't be considered. Defaults to 0.5.
        tile_size (int): The size of the tile. Defaults to 10.
        size_unit (str): The unit of the tile size. If is None, will use the unit of the crs of the gdf or from 'area_unit'. If set to 'percent', will use the percentage of the width of the gdf bounds. Defaults to percent. Overall, this can be ignored as long as percentage of width is the preferred window size.
        area_unit (str): The unit of the area. Defaults to "utm".
    """

    # Prepare the gdf
    if area_unit == "meter":
        crs = 3857
        gdf = gdf.to_crs(epsg=crs)
    elif area_unit == "utm":
        crs = gdf.estimate_utm_crs()
        gdf = gdf.to_crs(crs)
    elif area_unit is None:
        crs = None
    else:
        logger.warning("area_unit must be 'meter', 'utm', or None.")

    # Get the bounds of the gdf
    bounds = gdf.total_bounds
    width = abs(bounds[2] - bounds[0])
    height = abs(bounds[3] - bounds[1])
    x_min, y_min, x_max, y_max = bounds

    # Get the tile size
    if size_unit == "percent":
        tile_size = width * tile_size / 100
    elif size_unit is None:
        pass
    else:
        logger.warning("size_unit must be 'percent' or None. Will assume None.")

    assert tile_size > 0, "tile_size must be greater than 0."

    x_coords = np.arange(x_min, x_max, tile_size)
    y_coords = np.arange(y_min, y_max, tile_size)
    polygons = []
    for x in x_coords:
        for y in y_coords:
            polygons.append(
                Polygon(
                    [
                        (x, y),
                        (x + tile_size, y),
                        (x + tile_size, y + tile_size),
                        (x, y + tile_size),
                    ]
                )
            )
    grid = gpd.GeoDataFrame({"geometry": polygons}, crs=gdf.crs)
    intersected = gpd.overlay(gdf, grid, how="intersection")

    # TODO for each window, calculate the density_estimate_combined_area

    for extent in intersected.geometry:
        raster_extent = gpd.GeoDataFrame({"id": 1, "geometry": [extent]}, crs=gdf.crs)
        tile_polygons = gdf.clip(raster_extent)

        # Split multipolygon
        tile_polygons = tile_polygons.explode(index_parts=False)
        tile_polygons = tile_polygons.reset_index(drop=True)

        intersected
        tile_polygons
        average_storeys
        footprint_ratio
        tile_size
        size_unit
        height

    # TODO return a density map
