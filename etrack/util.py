from enum import Enum

class Illumination(Enum):
    Backlight = 0
    Incident = 1


class RegionShape(Enum):
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