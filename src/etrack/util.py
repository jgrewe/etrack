"""
Module containing utility functions and enum classes.
"""
from enum import Enum

class Illumination(Enum):
    Backlight = 0
    Incident = 1


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