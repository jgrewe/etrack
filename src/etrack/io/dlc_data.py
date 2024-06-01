import os
import numpy as np
import pandas as pd 
import numbers as nb

from ..tracking_data import TrackingData


class DLCReader(object):
    """Class that represents the tracking data stored in a DeepLabCut hdf5 file."""
    def __init__(self, results_file, crop=(0, 0)) -> None:
        """
        If the video data was cropped before tracking and the tracked positions are with respect to the cropped images, we may want to correct for this to convert the data back to absolute positions in the video frame.

        Parameters
        ----------
        crop : 2-tuple
            tuple of (xoffset, yoffset)

        Raises
        ------
        ValueError if crop value is not a 2-tuple
        """
        if not os.path.exists(results_file):
            raise ValueError("File %s does not exist!" % results_file)
        if not isinstance(crop, tuple) or len(crop) < 2:
            raise ValueError("Cropping info must be a 2-tuple of (x, y)")
        self._file_name = results_file
        self._crop = crop
        self._data_frame = pd.read_hdf(results_file)
        self._level_shape = self._data_frame.columns.levshape
        self._scorer = self._data_frame.columns.levels[0].values
        self._bodyparts = self._data_frame.columns.levels[1].values if self._level_shape[1] > 0 else []
        self._positions = self._data_frame.columns.levels[2].values if self._level_shape[2] > 0 else []

    @property
    def filename(self):
        return self._file_name

    @property
    def dataframe(self):
        return self._data_frame

    @property
    def scorer(self):
        return self._scorer

    @property
    def bodyparts(self):
        return self._bodyparts
    
    def _correct_cropping(self, orgx, orgy):
        x = orgx + self._crop[0]
        y = orgy + self._crop[1]
        return x, y

    def track(self, scorer=0, bodypart=0, framerate=30):
        if isinstance(scorer, nb.Number):
            sc = self._scorer[scorer]
        elif isinstance(scorer, str) and scorer in self._scorer:
            sc = scorer
        else:
            raise ValueError(f"Scorer {scorer} is not in dataframe!")
        if  isinstance(bodypart, nb.Number):
            bp = self._bodyparts[bodypart]
        elif isinstance(bodypart, str) and bodypart in self._bodyparts:
            bp = bodypart
        else:
            raise ValueError(f"Body part {bodypart} is not in dataframe!")

        x = np.asarray(self._data_frame[sc][bp]["x"] if "x" in self._positions else [])
        y = np.asarray(self._data_frame[sc][bp]["y"] if "y" in self._positions else [])
        x, y = self._correct_cropping(x, y)
        l = np.asarray(self._data_frame[sc][bp]["likelihood"] if "likelihood" in self._positions else [])

        time = np.arange(len(x))/framerate

        return TrackingData(x, y, time, l, bp, fps=framerate)