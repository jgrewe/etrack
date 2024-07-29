import pathlib
import numpy as np
import pandas as pd 
import numbers as nb
import nixtrack as nt
import logging as log

from abc import ABC, abstractmethod

from ..util import Orientation, YAxis, randianstocompass
from ..tracking_data import TrackingData

class DataSource(ABC):

    def __init__(self, filename:str, crop_origin: tuple[int, int]=(0, 0),
                 yorientation: YAxis = YAxis.Upright) -> None:
        super().__init__()
        p = pathlib.Path(filename)
        if not p.exists():
            raise FileExistsError(f"File {filename} does not exist!")
        if not isinstance(crop_origin, tuple) or len(crop_origin) < 2:
            raise ValueError("Cropping info must be a 2-tuple of (x, y)")
        self._filename = filename
        self._crop = crop_origin
        self._yorientation = yorientation
        self._scorer = None
        self._bodyparts = None
        self._positions = None

    @abstractmethod
    def track_data(self, bodypart=0, fps=30, scorer=0, track=-1):
        """
        Retrieve tracking data for a specific body part and track.

        Parameters
        ----------
            bodypart : int or str
            Index or name of the body part to retrieve tracking data for.
            fps : float
            Frames per second of the tracking data. If not provided, it will be retrieved from the dataset (only NixtrackData).
            scorer: int
            Index of the scorer, defaults to 0, only supported for DLC data.
            track : int or str
            Index of the track to retrieve tracking data for. Defaults to -1, only supported for NicTrack data

        Returns
        -------
            TrackingData: An object containing the x and y positions, time, score, body part name, and frames per second.

        Raises
        ------
            ValueError: If the body part or track is not valid.
        """
        raise NotImplementedError("Needs to be implemented by subclass")

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def orientation(self) -> YAxis:
        return self._yorientation

    @property
    def scorer(self):
        return self._scorer

    @property
    def bodyparts(self):
        return self._bodyparts

    @property
    def fps(self):
        """Property that holds frames per second of the original video.
        Returns
        -------
        int : the frames of second
        """
        raise NotImplementedError("Needs to be implemented by subclass")

    def _correct_cropping(self, orgx, orgy):
        """
        Corrects the coordinates based on the cropping values, If it cropping was done during tracking.

        Args:
            orgx (int): The original x-coordinate.
            orgy (int): The original y-coordinate.

        Returns:
            tuple: A tuple containing the corrected x and y coordinates.
        """
        x = orgx + self._crop[0]
        y = orgy + self._crop[1]
        return x, y

    

class DLCReader(DataSource):
    """Class that represents the tracking data stored in a DeepLabCut hdf5 file."""

    def __init__(self, filename, crop_origin, yorientation) -> None:
        """
        If the video data was cropped before tracking and the tracked positions are with respect to the cropped images, we may want to correct for this to convert the data back to absolute positions in the video frame.

        Parameters
        ----------
        crop_origin : 2-tuple
            tuple of (xoffset, yoffset)

        Raises
        ------
        ValueError if crop_origin value is not a 2-tuple
        """
        super().__init__(filename, crop_origin, yorientation)
        self._load()

    def _load(self):
        self._data_frame = pd.read_hdf(self.filename)
        self._level_shape = self._data_frame.columns.levshape
        self._scorer = self._data_frame.columns.levels[0].values
        self._bodyparts = self._data_frame.columns.levels[1].values if self._level_shape[1] > 0 else []
        self._positions = self._data_frame.columns.levels[2].values if self._level_shape[2] > 0 else []

    @property
    def dataframe(self):
        return self._data_frame

    @property
    def fps(self):
        Warning("DLC does not know about the framerate of the video recording.")

        return 0

    def track_data(self, bodypart=0, fps=30, scorer=0, track=-1):
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

        time = np.arange(len(x))/fps

        return TrackingData(x, y, time, l, bp, fps=fps)


class NixtrackReader(DataSource):
    """Wrapper around a nix data file that has been written according to the nixtrack model (https://github.com/bendalab/nixtrack)
    """
    def __init__(self, filename, crop_origin=(0, 0), yorientation=YAxis.Upright) -> None:
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
        super().__init__(filename, crop_origin, yorientation)
        self._load()

    def _load(self):
        self._dataset = nt.Dataset(self._filename)
        if not self._dataset.is_open:
            raise ValueError(f"An error occurred opening file {self._filename}! File is not open!")

    @property
    def bodyparts(self):
        """
        Returns the bodyparts of the dataset.

        Returns:
            list: A list of bodyparts.
        """
        return self._dataset.nodes

    @property
    def fps(self):
        return self._dataset.fps

    @property
    def tracks(self):
        """
        Returns a list of tracks from the dataset.

        Returns:
            list: A list of tracks.
        """
        return [t[0] for t in self._dataset.tracks]

    def track_data(self, bodypart=0, fps=None, scorer=None, track=-1):
        if isinstance(bodypart, nb.Number):
            bp = self.bodyparts[bodypart]
        elif isinstance(bodypart, (str)) and bodypart in self.bodyparts:
            bp = bodypart
        else:
            raise ValueError(f"Body part {bodypart} is not a tracked node!")

        if track not in self.tracks:
            raise ValueError(f"Track {track} is not a valid track name!")
        if not isinstance(track, (list, tuple)):
            track = [track]
        elif isinstance(track, int):
            track = [self.tracks[track]]

        if fps is None:
            fps = self._dataset.fps

        positions, time, _, nscore = self._dataset.positions(node=bp, axis_type=nt.AxisType.Time)
        if self._yorientation == YAxis.Upright:
            ymax = self._dataset.frame_height
            if len(positions.shape) == 3:
                positions[:, 1, :] = ymax - positions[:, 1, :]
            else:
                positions[:, 1] = ymax - positions[:, 1]
        valid = ~np.isnan(positions[:, 0])
        positions = positions[valid,:]
        time = time[valid]
        score = nscore[valid]

        return TrackingData(positions[:, 0], positions[:, 1], time, score, bp, fps=fps)
