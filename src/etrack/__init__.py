# -*- coding: utf-8 -*-
""" etrack package for easier reading and handling of efish tracking data.

Copyright © 2024, Jan Grewe

Redistribution and use in source and binary forms, with or without modification, are permitted under the terms of the BSD License. See LICENSE file in the root of the Project.
"""
import pathlib
import logging

from .image_marker import ImageMarker, MarkerTask
from .tracking_result import TrackingResult, coordinate_transformation
from .arena import Arena, Region
from .tracking_data import TrackingData
from .io.ioclasses import DLCReader
from .io.ioclasses import NixtrackReader
from .util import RegionShape, AnalysisType, Orientation, FileType, YAxis


def read_dataset(filename: str, filetype:FileType, crop_origin: tuple[int, int]=(0, 0),
                 yorientation: YAxis = YAxis.Upright):
    """Open a file that contains tracking data. Supported file types are currently 
    nixtrack files e.g. written with the nix output of SLEAP (https://sleap.ai) or
    hdf *.h5 files written by DeepLabCut (https://github.com/deeplabcut).

    If the video data was cropped before tracking and the tracked positions are with respect to the cropped images, we may want to correct for this to convert the data back to absolute positions in the video frame.

    yorientation argument defines whether the origin is the top-left (YAxis.Inverted) or bottom-left (YAxis.Upright) corner.

    Parameters
    ----------
    filename : str
        the file name
    filetype : FileType
        Either etrack.FileType.Deeplabcut or etrack.FileType.Sleap
    crop_origin: 2-tuple
        tuple of (xoffset, yoffset), defaults to (0, 0) i.e. no cropping
    yorientation: etrack.YAxis
        Either YAxis.Upright (default) or YAxis.inverted

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        if the file does not exist or cannot be read.
    """
    p = pathlib.Path(filename)
    logging.debug(f"Open file {filename} with filetype {filetype.__str__()}")
    if not pathlib.Path.exists(p):
        logging.error(f"File {filename} does not exist!")
        raise ValueError(f"File {filename} does not exist!")
    if filetype == FileType.Deeplabcut:
        return DLCReader(filename, crop_origin, yorientation=yorientation)
    elif filetype == FileType.Sleap:
        return NixtrackData(filename, crop_origin=crop_origin, yorientation=yorientation)
    
    pass