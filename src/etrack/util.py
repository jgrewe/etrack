"""
Module containing utility functions and enum classes.
"""
import numpy as np

from enum import Enum

class Illumination(Enum):
    Backlight = 0
    Incident = 1


class FileType(Enum):
    Deeplabcut = 0
    Sleap = 1
    Auto = 2

    def __str__(self) -> str:
        return self.name

class RegionShape(Enum):
    """
    Enumeration representing the shape of a region.

    Attributes:
        Circular: Represents a circular region.
        Rectangular: Represents a rectangular region.
    """

    Circular = 0
    Rectangular = 1

    def __str__(self) -> str:
        return self.name


class AnalysisType(Enum):
    """
    Enumeration representing different types of analysis used when analyzing whether
    positions fall into a given region.
    
    Possible types:
        AnalysisType.Full: considers both, the x- and the y-coordinates
        AnalysisType.CollapseX: consider only the x-coordinates
        AnalysisType.CollapseY: consider only the y-coordinates
    """
    Full = 0
    CollapseX = 1
    CollapseY = 2

    def __str__(self) -> str:
        """
        Returns the string representation of the analysis type.
        
        Returns:
            str: The name of the analysis type.
        """
        return self.name

class PositionType(Enum):
    Absolute = 0
    Cropped = 1


class Orientation(Enum):
    Radians = 0
    Compass = 1


class YAxis(Enum):
    Upright = 0
    Inverted = 1


def randianstocompass(direction):
    """
    Convert angles in radians to compass degrees.

    Parameters:
    direction (numpy.ndarray): Array of angles in radians.

    Returns:
    numpy.ndarray: Array of compass directions in degrees.

    """
    degrees = direction * 180 / np.pi
    degrees = 90 - degrees
    degrees[degrees < 0] = degrees[degrees < 0] + 360
    return degrees


def align_detections(node1_pos, node1_times, node2_pos, node2_times):
    """
    Aligns the detections from two nodes based on their timestamps. Use this in case that the 
    two nodes were not detected at all the same times (common problem).

    Parameters:
    -----------
    node1_pos : numpy.ndarray
        The positions of the detections from node 1.
    node1_times : numpy.ndarray
        The timestamps of the detections from node 1.
    node2_pos : numpy.ndarray
        The positions of the detections from node 2.
    node2_times : numpy.ndarray
        The timestamps of the detections from node 2.

    Returns:
    --------
    t : numpy.ndarray
        The common timestamps between the two nodes.
    n1pos : numpy.ndarray
        The positions of the detections from node 1 within the common timestamps.
    n2pos : numpy.ndarray
        The positions of the detections from node 2 within the common timestamps.

    Raises:
    -------
    ValueError
        If the front node and back node times do not have the same sampling rate.
    """
    if np.median(np.diff(node1_times)) != np.median(np.diff(node2_times)):
        raise ValueError("Front node- and back-node times do not have the same sampling rate!")

    common_time = set(node1_times).intersection(set(node2_times))
    min_time = min(common_time)
    max_time = max(common_time)

    t = node1_times[(node1_times >= min_time) & (node1_times <= max_time)]
    n1pos = node1_pos[(node1_times >= min_time) & (node1_times <= max_time), :]
    n2pos = node2_pos[(node2_times >= min_time) & (node2_times <= max_time), :]

    return t, n1pos, n2pos


def body_orientation(frontnode_positions, backnode_positions, orientation=Orientation.Compass):
    """
    Calculate the body orientation angle between frontnode_positions and backnode_positions. E.g. the snout 
    and tail positions.

    Parameters:
    -----------
    frontnode_positions : numpy.ndarray
        Array of shape (n, 2) representing the positions of the front nodes.
    backnode_positions : numpy.ndarray
        Array of shape (n, 2) representing the positions of the back nodes.
    orientation : Orientation, optional
        The orientation type to use for the angle calculation. Default is Orientation.Compass.

    Returns:
    --------
    numpy.ndarray
        Array of shape (n,) containing the body orientation angles.

    """
    # get the angle between them
    o = np.arctan2(frontnode_positions[:, 1] - backnode_positions[:, 1],
                   frontnode_positions[:, 0] - backnode_positions[:, 0])

    if orientation == Orientation.Compass:
        o = randianstocompass(o)

    return o
