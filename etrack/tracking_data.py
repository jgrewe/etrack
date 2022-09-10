import numpy as np


class TrackingData(object):
    """Class that represents tracking data, i.e. positions of an agent tracked in an environment.
    These data are the x, and y-positions, the time at which the agent was detected, and the quality associated with the position estimation.
    TrackingData contains these data and offers a few functions to work with it.
    Using the 'quality_threshold', 'temporal_limits', or the 'position_limits' data can be filtered (see filter_tracks function).
    The 'interpolate' function allows to fill up the gaps that be result from filtering with linearly interpolated data points.

    More may follow... 
    """
    def __init__(self, x, y, time, quality, node="", fps=None, quality_threshold=None, temporal_limits=None, position_limits=None) -> None:
        self._orgx = x
        self._orgy = y
        self._orgtime = time
        self._orgquality = quality
        self._x = x
        self._y = y
        self._time = time
        self._quality = quality
        self._node = node
        self._threshold = quality_threshold
        self._position_limits = position_limits
        self._time_limits = temporal_limits

    @property
    def original_positions(self):
        return self._orgx, self._orgy

    @property
    def original_quality(self):
        return self._orgquality

    def interpolate(self, store=True, min_count=10):
        if len(self._x) < min_count:
            print(f"{self._node} data has less than {min_count} data points with sufficient quality!")
            return None
        x = np.interp(self._orgtime, self._time, self._x)
        y = np.interp(self._orgtime, self._time, self._y)
        if store:
            self._x = x
            self._y = y
            self._time = self._orgtime.copy()

    @property
    def quality_threshold(self):
        return self._threshold

    @quality_threshold.setter
    def quality_threshold(self, new_threshold):
        """Setter of the quality threshold that should be applied when filterin the data. Setting this to None removes the quality filter.

        Parameters
        ----------
        new_threshold : float
            
        """
        self._threshold = new_threshold

    @property
    def position_limits(self):
        return self._position_limits

    @position_limits.setter
    def position_limits(self, new_limits):
        """Sets the limits for the position filter. 'new_limits' must be a 4-tuple of the form (x0, y0, width, height). If None, the limits will be removed.

        Parameters
        ----------
        new_limits: 4-tuple
            tuple of x-position, y-position, the width and the height. Passing None removes the filter

        Raises
        ------
        ValueError, if new_value is not a 4-tuple
        """
        if new_limits is not None and not (isinstance(new_limits, (tuple, list)) and len(new_limits) == 4):
            raise ValueError(f"The new_limits vector must be a 4-tuple of the form (x, y, width, height)")
        self._position_limits = new_limits

    @property
    def temporal_limits(self):
        return self._time_limits

    @temporal_limits.setter
    def temporal_limits(self, new_limits):
        """Limits for temporal filter. The limits must be a 2-tuple of start and end time. Setting this to None removes the filter.

        Parameters
        ----------
        new_limits : 2-tuple
            The new limits in the form (start, end) given in seconds.
        """
        if new_limits is not None and not (isinstance(new_limits, (tuple, list)) and len(new_limits) == 2):
            raise ValueError(f"The new_limits vector must be a 2-tuple of the form (start, end). ")
        self._time_limits = new_limits

    def filter_tracks(self):
        """Applies the filters to the tracking data. All filters will be applied squentially, i.e. an AND connection.
        To change the filter settings use the setters for 'quality_threshold', 'temporal_limits', 'position_limits'. Setting them to None disables the respective filter discarding the setting.
        """
        self._x = self._orgx.copy()
        self._y = self._orgy.copy()
        self._time = self._orgtime.copy()
        self._quality = self.original_quality.copy()

        if self.position_limits is not None:
            x_max = self.position_limits[0] + self.position_limits[2]
            y_max = self.position_limits[1] + self.position_limits[3]
            indices = np.where((self._x >= self.position_limits[0]) & (self._x < x_max) &
                               (self._y >= self.position_limits[1]) & (self._y < y_max))
            self._x = self._x[indices]
            self._y = self._y[indices]
            self._time = self._time[indices]
            self._quality = self._quality[indices]
        
        if self.temporal_limits is not None:
            indices = np.where((self._time >= self.temporal_limits[0]) &
                               (self._time < self.temporal_limits[1]))
            self._x = self._x[indices]
            self._y = self._y[indices]
            self._time = self._time[indices]
            self._quality = self._quality[indices]

        if self.quality_threshold is not None:
            indices = np.where((self._quality >= self.quality_threshold))
            self._x = self._x[indices]
            self._y = self._y[indices]
            self._time = self._time[indices]
            self._quality = self._quality[indices]

    def positions(self):
        """Returns the filtered data (if filters have been applied). 

        Returns
        -------
        np.ndarray
            The x-positions
        np.ndarray
            The y-positions
        np.ndarray
            The time vector
        np.ndarray
            The detection quality
        """
        return self._x, self._y, self._time, self._quality

    def speed(self):
        """ Returns the agent's speed as a function of time and position. The speed estimation is associated to the time/position between two sample points.

        Returns
        -------
        np.ndarray:
            The time vector.
        np.ndarray:
            The speed.
        tuple of np.ndarray
            The position
        """
        speed = np.sqrt(np.diff(self._x)**2 + np.diff(self._y)**2) / np.diff(self._time)
        t = self._time[:-1] + np.diff(self._time) / 2
        x = self._x[:-1] + np.diff(self._x) / 2
        y = self._y[:-1] + np.diff(self._y) / 2

        return t, speed, (x, y)

    def __repr__(self) -> str:
        s = f"Tracking data of node '{self._node}'!"
        return s