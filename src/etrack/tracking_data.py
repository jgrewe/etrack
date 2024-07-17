"""
Module that defines the TrackingData class that wraps the position data for a given node/bodypart that has been tracked.
"""

import numpy as np

from .util import randianstocompass, Orientation

class TrackingData(object):
    """Class that represents tracking data, i.e. positions of an agent tracked in an environment.
    These data are the x, and y-positions, the time at which the agent was detected, and the quality associated with the position estimation.
    TrackingData contains these data and offers a few functions to work with it.
    Using the 'quality_threshold', 'temporal_limits', or the 'position_limits' data can be filtered (see filter_tracks function).
    The 'interpolate' function allows to fill up the gaps that may result from filtering with linearly interpolated data points.

    More may follow...
    """

    def __init__(self, x, y, time, quality, node="", fps=None, 
                 quality_threshold=None, temporal_limits=None, position_limits=None) -> None:
        """
        Initialize a TrackingData object.

        Parameters
        ----------
        x : float
            The x-coordinates of the tracking data.
        y : float
            The y-coordinates of the tracking data.
        time : float
            The time vector associated with the x-, and y-coordinates.
        quality : float
            The quality score associated with the position estimates.
        node : str, optional
            The node name associated with the data. Default is an empty string.
        fps : float, optional
            The frames per second of the tracking data. Default is None.
        quality_threshold : float, optional
            The quality threshold for the tracking data. Default is None.
        temporal_limits : tuple, optional
            The temporal limits for the tracking data. Default is None.
        position_limits : tuple, optional
            The position limits for the tracking data. Default is None.
        """
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
        self._fps = fps

    @property
    def original_positions(self):
        return self._orgx, self._orgy

    @property
    def original_quality(self):
        return self._orgquality

    def interpolate(self, start_time=None, end_time=None, min_count=5):
        """
        Interpolates the tracking data to fill gaps in the tracking.

        Parameters
        ----------
            start_time : float, optional
                The start time for interpolation. If not provided, the first time value in the data will be used.
            end_time : float, optional
                The end time for interpolation. If not provided, the last time value in the data will be used.
            min_count : int, optional
                The minimum number of data points required for interpolation. If the number of data points is less than this value, interpolation will not be performed.

        Returns
        -------
            x : ndarray
                The interpolated x-coordinates.
            y : ndarray
                The interpolated y-coordinates.
            time : ndarray
                The interpolated time values.
            interpolated : ndarray
                An array indicating whether each returned position is original (0) or interpolated (1).

        Note
        ----
            This method uses numpy's interpolation function to fill in missing values in the tracking data. It generates a new set of time values based on the specified start and end times, and then interpolates the x and y coordinates at those time points. The interpolated array is used to indicate whether each interpolated value is valid or not.

        """
        if len(self._x) < min_count:
            print(
                f"TrackingData.interpolate: {self._node} data has less than {min_count} data points with sufficient quality ({len(self._x)})!"
            )
            return None, None, None, None
        start = self._time[0] if start_time is None else start_time
        end = self._time[-1] if end_time is None else end_time
        time = np.round(np.arange(start, end + 1.0/self._fps, 1.0 / self._fps), 4)
        x = np.interp(time, self._time, self._x)
        y = np.interp(time, self._time, self._y)
        interpolated = np.ones_like(time, dtype=int)
        tt = np.round(self._time, 4)
        for i, t in enumerate(time):
            if t in tt:
                interpolated[i] = 0
        if x.shape != y.shape != time.shape != interpolated.shape:
            from IPython import embed
            print("x or y or time or interpolated are not the same size")
            embed()
            exit()
        return x, y, time, interpolated

    @property
    def quality_threshold(self):
        """Property that holds the quality filter setting.
        
        Returns
        -------
            float : the quality threshold
        """
        return self._threshold

    @quality_threshold.setter
    def quality_threshold(self, new_threshold):
        """Setter of the quality threshold that should be applied when filtering the data. Setting this to None removes the quality filter.
        
        Data points that have a quality score below the given threshold are discarded.

        Parameters
        ----------
        new_threshold : float

        """
        self._threshold = new_threshold

    @property
    def position_limits(self):
        """
        Get the position limits of the tracking data.

        Returns:
            tuple: A 4-tuple containing the start x, and y positions, width and height limits.
        """
        return self._position_limits

    @position_limits.setter
    def position_limits(self, new_limits):
        """Sets the limits for the position filter. 'new_limits' must be a 4-tuple of the form (x0, y0, width, height). If None, the limits will be removed.
        Data points outside the position limits are discarded.

        Parameters
        ----------
        new_limits: 4-tuple
            tuple of x-position, y-position, the width and the height. Passing None removes the filter

        Raises
        ------
        ValueError, if new_value is not a 4-tuple
        """
        if new_limits is not None and not (
            isinstance(new_limits, (tuple, list)) and len(new_limits) == 4
        ):
            raise ValueError(
                f"The new_limits vector must be a 4-tuple of the form (x, y, width, height)"
            )
        self._position_limits = new_limits

    @property
    def temporal_limits(self):
        """
        Get the temporal limits of the tracking data.

        Returns
        -------
            tuple: A tuple containing the start and end time of the tracking data.
        """
        return self._time_limits

    @temporal_limits.setter
    def temporal_limits(self, new_limits):
        """Limits for temporal filter. The limits must be a 2-tuple of start and end time. Setting this to None removes the filter.
        Data points the are associated with times outside the limits are discarded.
        
        Parameters
        ----------
        new_limits : 2-tuple
            The new limits in the form (start, end) given in seconds.

        Returns
        -------
        None
        
        Raises
        ------
            ValueError if the limits are not valid.
        """
        if new_limits is not None and not (
            isinstance(new_limits, (tuple, list)) and len(new_limits) == 2
        ):
            raise ValueError(
                f"The new_limits vector must be a 2-tuple of the form (start, end). "
            )
        self._time_limits = new_limits

    def filter_tracks(self, align_time=True):
        """Applies the filters to the tracking data. All filters will be applied sequentially, i.e. an AND connection.
        To change the filter settings use the setters for 'quality_threshold', 'temporal_limits', 'position_limits'. Setting them to None disables the respective filter discarding the setting.

        Parameters
        ----------
        align_time: bool
            Controls whether the time vector is aligned to the first time point at which the agent is within the positional_limits. Default = True
        """
        self._x = self._orgx.copy()
        self._y = self._orgy.copy()
        self._time = self._orgtime.copy()
        self._quality = self.original_quality.copy()

        if self.position_limits is not None:
            x_max = self.position_limits[0] + self.position_limits[2]
            y_max = self.position_limits[1] + self.position_limits[3]
            indices = np.where(
                (self._x >= self.position_limits[0])
                & (self._x < x_max)
                & (self._y >= self.position_limits[1])
                & (self._y < y_max)
            )
            self._x = self._x[indices]
            self._y = self._y[indices]
            self._time = self._time[indices] - self._time[0] if align_time else 0.0
            self._quality = self._quality[indices]

        if self.temporal_limits is not None:
            indices = np.where(
                (self._time >= self.temporal_limits[0])
                & (self._time < self.temporal_limits[1])
            )
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
        """Returns the filtered data (if filters have been applied, otherwise the original data).

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

    def speed(self, x=None, y=None, t=None):
        """ Returns the agent's speed as a function of time and position. The speed estimation is associated to the time/position between two sample points. If any of the arguments is not provided, the function will use the x,y coordinates that are stored within the object, otherwise, if all are provided, the user-provided values will be used.

        Since the velocities are estimated from the difference between two sample points the returned velocities and positions are assigned to positions and times between the respective sampled positions/times. 

        Parameters
        ----------
        x: np.ndarray
            The x-coordinates, defaults to None
        y: np.ndarray
            The y-coordinates, defaults to None
        t: np.ndarray
            The time vector, defaults to None
        Returns
        -------
        np.ndarray:
            The time vector.
        np.ndarray:
            The speed.
        tuple of np.ndarray
            The position
        """
        if x is None or y is None or t is None:
            x = self._x.copy()
            y = self._y.copy()
            t = self._time.copy()
        dt = np.diff(t)
        speed = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt
        t = t[:-1] + dt / 2
        x = x[:-1] + np.diff(x) / 2
        y = y[:-1] + np.diff(y) / 2

        return t, speed, (x, y)

    def movement_direction(self, x=None, y=None, orientation=Orientation.Compass):
        """
        Calculate the movement direction based on the given x and y coordinates.

        Parameters
        ----------
            x : ndarray, optional
                The x-coordinates. If not provided, the instance variable _x will be used.
            y : ndarray, optional
                The y-coordinates. If not provided, the instance variable _y will be used.
            orientation : Orientation, optional
                Whether to return the direction in compass degrees (0-360) with zero heading north. 
                If False, the direction will be returned in radians 0 headning right. Defaults to Orientation.Compass.

        Returns
        -------
            ndarray: 
                The movement direction in degrees or radians, depending on the value of orientation.

        """
        if x is None:
            x = self._x
        if y is None:
            y = self._y
        dx = np.diff(x)
        dy = np.diff(y)
        direction = np.arctan2(dy, dx)
        if orientation == Orientation.Compass:
            direction = randianstocompass(direction)

        return direction

    def __repr__(self) -> str:
        s = f"Tracking data of node '{self._node}'."
        return s
