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
    Full = 0
    CollapseX = 1
    CollapseY = 2

    def __str__(self) -> str:
        return self.name

class PositionType(Enum):
    Absolute = 0
    Cropped = 1