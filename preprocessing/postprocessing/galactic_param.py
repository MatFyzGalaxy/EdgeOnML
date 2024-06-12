#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:54:55 2024

@author: zc
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.ned import Ned
from astroquery.simbad import Simbad

# turn off warnings from astroquery
import warnings

warnings.filterwarnings("ignore")

FIT_PATH = r"data\fits"
YOLO_OUTPUT_PATH = (
    "yolo\labels"
)


@dataclass
class YoloOutput:
    x_center_norm: float
    y_center_norm: float
    confidence: float


@dataclass
class FitOutput:
    wcs: WCS
    fit_width: int
    fit_height: int


@dataclass
class GalaxyData:
    fit_name: str
    RA: float
    DEC: float
    Galaxy_Name_Simbad: str
    Galaxy_Name_NED: str
    Confidence: float


def extract_data_from_yolo(yolo_output_file_path: str, line_number: int) -> YoloOutput:
    with open(yolo_output_file_path, "r") as yolo_output_txt:
        yolo_output_list = yolo_output_txt.readlines()

    _, x_center_norm, y_center_norm, _, _, confidence = (
        yolo_output_list[line_number].strip().split()
    )
    return YoloOutput(
        x_center_norm=float(x_center_norm),
        y_center_norm=float(y_center_norm),
        confidence=float(confidence),
    )


def extract_data_from_fit(fit_file_path: str) -> FitOutput:
    with fits.open(fit_file_path) as hdu_list:
        image_shape = hdu_list[0].data.shape
        return FitOutput(
            wcs=WCS(hdu_list[0].header),
            fit_width=image_shape[1],
            fit_height=image_shape[0],
        )


def get_skycoord(fit_output: FitOutput, yolo_output: YoloOutput) -> SkyCoord:
    """
    premena pixelov - len centralny, zakomentovana cast je pre cely obrazok
    """
    x_center = round(yolo_output.x_center_norm * fit_output.fit_width)
    y_center = round(
        fit_output.fit_height - (yolo_output.y_center_norm * fit_output.fit_height)
    )
    print(f"Galaxy center on the fit: x: {x_center}; y: {y_center}")

    central_pixel = np.array([[x_center, y_center]])
    world_coords = fit_output.wcs.pixel_to_world(
        central_pixel[:, 0], central_pixel[:, 1]
    )
    ra_values = world_coords.ra.deg[0]
    dec_values = world_coords.dec.deg[0]

    return SkyCoord(ra=ra_values, dec=dec_values, unit="deg", frame="icrs")


def get_galaxy_name_simbad(ra_deg, dec_deg) -> str:
    # first way of getting parameters from database SIMBAD
    coords = SkyCoord(
        ra=ra_deg, dec=dec_deg, unit="deg", frame="icrs", equinox="J2000.0"
    )
    result_table = Simbad.query_region(coords, radius="0d0m6s")

    if result_table is None or len(result_table) <= 0:
        return "Galaxy not found"

    return result_table["MAIN_ID"][0]


def get_galaxy_name_ned(sky_coord: SkyCoord) -> str:
    # second way of getting parameters from database NED
    result_table = Ned.query_region(
        sky_coord, radius=0.001 * units.deg, equinox="J2000.0"
    )
    if result_table is None or len(result_table) <= 0:
        return "Galaxy not found"

    return result_table["Object Name"][0]


def process_fit(
    fit_file_path: str, yolo_output_file_path: str, line_number: int
) -> GalaxyData:
    yolo_output = extract_data_from_yolo(
        yolo_output_file_path=yolo_output_file_path, line_number=line_number
    )
    fit_output = extract_data_from_fit(fit_file_path=fit_file_path)

    central_coords = get_skycoord(fit_output=fit_output, yolo_output=yolo_output)
    print("Central Coordinates (RA2000, DEC2000):", central_coords)

    ra = np.round(central_coords.ra.deg, 3)
    dec = np.round(central_coords.dec.deg, 3)
    galaxy_name_1 = get_galaxy_name_simbad(ra, dec)
    print("Galaxy Name:", galaxy_name_1)

    galaxy_name_2 = get_galaxy_name_ned(central_coords)
    print("Galaxy Name:", galaxy_name_2)

    return GalaxyData(
        fit_name=fit_file_path.split("\\")[-1],
        RA=ra,
        DEC=dec,
        Galaxy_Name_Simbad=galaxy_name_1,
        Galaxy_Name_NED=galaxy_name_2,
        Confidence=yolo_output.confidence,
    )


def process_all_fits(fit_dir_path: str) -> pd.DataFrame:
    df_galaxy_data = pd.DataFrame(
        columns=[
            "fit_name",
            "RA",
            "DEC",
            "Galaxy_Name_Simbad",
            "Galaxy_Name_NED",
            "Confidence",
        ]
    )

    for fit_filename in os.listdir(fit_dir_path):
        fit_filepath = os.path.join(fit_dir_path, fit_filename)
        label_filepath = os.path.join(
            YOLO_OUTPUT_PATH, f"{fit_filename.split()[0]}.txt"
        )
        line_number = int(fit_filename.split()[-1].split(".")[0])
        try:
            galaxy_data = process_fit(
                fit_file_path=fit_filepath,
                yolo_output_file_path=label_filepath,
                line_number=line_number,
            )
        except Exception as e:
            print(f"Error processing {fit_filename}: {e}")
            continue
        df_galaxy_data = df_galaxy_data.append(galaxy_data.__dict__, ignore_index=True)

    return df_galaxy_data


result_df = process_all_fits(FIT_PATH)
print(f"Galaxies processed: {len(result_df)}")
result_df.to_excel("fp_galaxy_data2.xlsx", index=False)
