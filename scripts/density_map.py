# -*- coding: utf-8 -*-
import logging

# import geopandas as gpd
# import rasterio as rio

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
    crs="EPSG:3857",
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
        crs (str): The crs of the annotation geodataframe to calculate the area. For meters, use 'EPSG:3857'. For degrees, use 'EPSG:4326'.
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
    annotation, crs="EPSG:3857", average_storeys=None, storey_column: str = "storeys"
) -> float:
    """This function will get the area of annotation geodataframe, and also get
    the number of geometries in the annotation geodataframe, and return a
    number that is the aera of the annotation geodataframe divided by the
    number of geometries in the annotation geodataframe.

    Args:
        annotation (geodataframe): A geodataframe of annotations.
        crs (str): The crs of the annotation geodataframe to calculate the area. For meters, use 'EPSG:3857'. For degrees, use 'EPSG:4326'.
        average_storeys (int): The average number of storeys of buildings in the annotation geodataframe. If None, will not calculate the average number of storeys using the meta data. Defaults to None.

    Returns:
        density (float): The area of the annotation geodataframe divided by the number of geometries in the annotation geodataframe.
    """

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
    annotation, crs="EPSG:3857", average_storeys=None, storey_column: str = "storeys"
) -> float:
    """This function will get the area of annotation geodataframe, and also get
    the number of geometries in the annotation geodataframe, and return a
    number that is the aera of the annotation geodataframe divided by the
    number of geometries in the annotation geodataframe.

    Args:
        annotation (geodataframe): A geodataframe of annotations.
        crs (str): The crs of the annotation geodataframe to calculate the area. For meters, use 'EPSG:3857'. For degrees, use 'EPSG:4326'.
        average_storeys (int): The average number of storeys of buildings in the annotation geodataframe. If None, will not calculate the average number of storeys using the meta data. Defaults to None.

    Returns:
        density (float): The area of the annotation geodataframe divided by the number of geometries in the annotation geodataframe.
    """

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


def density_map_maker(geojson, average_storeys, footprint_ratio, tile_size, map_units):
    """This function will use the density_estimate_combined_area function to
    create a density map, by tiling the geojson."""

    # make a sliding window on the geojson, with the size of tile_size
    # for each window, calculate the density_estimate_combined_area
    # return a density map

    pass
