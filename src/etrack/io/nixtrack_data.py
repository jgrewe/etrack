import os
import numpy as np
import pandas as pd 
import numbers as nb
import nixtrack as nt

from ..tracking_data import TrackingData
from IPython import embed


class NixtrackData(object):
    
    def __init__(self, filename, crop=(0, 0)) -> None:
        """
        If the video data was cropped before tracking and the tracked positions are with respect to the cropped images, we may want to correct for this to convert the data back to absolute positions in the video frame.

        Parameters
        ----------
        filename : str
            full filename
        crop : 2-tuple
            tuple of (xoffset, yoffset)

        Raises
        ------
        ValueError if crop value is not a 2-tuple
        """
        if not os.path.exists(filename):
            raise ValueError("File %s does not exist!" % filename)
        if not isinstance(crop, tuple) or len(crop) < 2:
            raise ValueError("Cropping info must be a 2-tuple of (x, y)")
        self._file_name = filename
        self._crop = crop
        self._dataset = nt.Dataset(self._file_name)
        if not self._dataset.is_open:
            raise ValueError(f"An error occurred opening file {self._file_name}! File is not open!")

    @property
    def filename(self):
        return self._file_name

    @property
    def bodyparts(self):
        return self._dataset.nodes
    
    def _correct_cropping(self, orgx, orgy):
        x = orgx + self._crop[0]
        y = orgy + self._crop[1]
        return x, y
    
    @property
    def tracks(self):
        return self._dataset.tracks

    def track(self, bodypart=0, fps=None):
        if isinstance(bodypart, nb.Number):
            bp = self.bodyparts[bodypart]
        elif isinstance(bodypart, (str)) and bodypart in self.bodyparts:
            bp = bodypart
        else:
            raise ValueError(f"Body part {bodypart} is not a tracked node!")
        if fps is None:
            fps = self._dataset.fps

        positions, time, _, nscore = self._dataset.positions(node=bp, axis_type=nt.AxisType.Time)
        valid = ~np.isnan(positions[:, 0])
        positions = positions[valid,:]
        time = time[valid]
        score = nscore[valid]

        return TrackingData(positions[:, 0], positions[:, 1], time, score, bp, fps=fps)
