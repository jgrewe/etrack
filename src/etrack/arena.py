import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.draw import disk
from .util import RegionShape, AnalysisType, Illumination


class Region(object):
    """
    Class representing a region (of interest). Regions can be either circular or rectangular. 
    A Region can have a parent, i.e. it is contained inside a parent region. It can also have children.

    Coordinates are given in absolute coordinates. The extent is treated depending on the shape. In case of a circular
    shape, it is the radius and the origin is the center of the circle. Otherwise the origin is the bottom, or top-left corner, depending on the y-axis orientation, if inverted, then it is top-left. FIXME: check this
    
    """
    def __init__(self, origin, extent, inverted_y=True, name="", region_shape=RegionShape.Rectangular, parent=None) -> None:
        """Region constructor.
        Parameters
        ----------
        origin : 2-tuple
            x, and y coordinates
        extent : scalar or 2-tuple, scalar only allowed to circular regions, 2-tuple for rectangular.
        inverted_y : bool, optional
            _description_, by default True
        name : str, optional
            _description_, by default ""
        region_shape : _type_, optional
            _description_, by default RegionShape.Rectangular
        parent : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            Raises Value error when origin or extent are invalid
        """
        logging.debug(
            f"etrack.Region: Create {str(region_shape)} region {name} with props origin {origin}, extent {extent} and parent {parent}"
        )
        if len(origin) != 2:
            raise ValueError("Region: origin must be 2-tuple!")
        self._parent = parent
        self._name = name
        self._shape_type = region_shape
        self._origin = origin
        self._check_extent(extent)
        self._extent = extent
        self._inverted_y = inverted_y

    @staticmethod
    def circular_mask(width, height, center, radius):
        assert center[1] + radius < width and center[1] - radius > 0
        assert center[0] + radius < height and center[0] - radius > 0

        mask = np.zeros((height, width), dtype=np.uint8)
        rr, cc = disk(reversed(center), radius)
        mask[rr, cc] = 1

        return mask

    @property
    def name(self):
        return self._name

    @property
    def inverted_y(self):
        return self._inverted_y

    @property
    def _max_extent(self):
        if self._shape_type == RegionShape.Rectangular:
            max_extent = (
                self._origin[0] + self._extent[0],
                self._origin[1] + self._extent[1],
            )
        else:
            max_extent = (
                self._origin[0] + self._extent,
                self._origin[1] + self._extent,
            )
        return np.asarray(max_extent)

    @property
    def _min_extent(self):
        if self._shape_type == RegionShape.Rectangular:
            min_extent = self._origin
        else:
            min_extent = (
                self._origin[0] - self._extent,
                self._origin[1] - self._extent,
            )
        return np.asarray(min_extent)

    @property
    def xmax(self):
        return self._max_extent[0]

    @property
    def xmin(self):
        return self._min_extent[0]

    @property
    def ymin(self):
        return self._min_extent[1]

    @property
    def ymax(self):
        return self._max_extent[1]

    @property
    def position(self):
        """
        Get the position of the arena.

        Returns
        -------
        tuple
            A tuple containing the x-coordinate, y-coordinate, width, and height of the arena.
        """
        x = self._min_extent[0]
        y = self._min_extent[1]
        width = self._max_extent[0] - self._min_extent[0]
        height = self._max_extent[1] - self._min_extent[1]
        return x, y, width, height

    def _check_extent(self, ext):
        """Checks whether the extent matches the shape. i.e. if the shape is Rectangular, extent must be a length 2 list, tuple, otherwise, if the region is circular, extent must be a single numerical value.

        Parameters
        ----------
        ext : tuple, or numeric scalar
        """
        if self._shape_type == RegionShape.Rectangular:
            if not isinstance(ext, (list, tuple, np.ndarray)) and len(ext) != 2:
                raise ValueError(
                    "Extent must be a length 2 list or tuple for rectangular regions!"
                )
        elif self._shape_type == RegionShape.Circular:
            if not isinstance(ext, (int, float)):
                raise ValueError(
                    "Extent must be a numerical scalar for circular regions!"
                )
        else:
            raise ValueError(f"Invalid ShapeType, {self._shape_type}!")

    def fits(self, other) -> bool:
        """
        Checks if the given region fits into the current region.

        Args:
            other (Region): The region to check if it fits.

        Returns:
            bool: True if the given region fits into the current region, False otherwise.
        """
        assert isinstance(other, Region)
        does_fit = all(
            (
                other._min_extent[0] >= self._min_extent[0],
                other._min_extent[1] >= self._min_extent[1],
                other._max_extent[0] <= self._max_extent[0],
                other._max_extent[1] <= self._max_extent[1],
            )
        )
        if not does_fit:
            m = (
                f"Region {other.name} does not fit into {self.name}. "
                f"min x: {other._min_extent[0] >= self._min_extent[0]},",
                f"min y: {other._min_extent[1] >= self._min_extent[1]},",
                f"max x: {other._max_extent[0] <= self._max_extent[0]},",
                f"max y: {other._max_extent[1] <= self._max_extent[1]}",
            )
            logging.debug(m)
        return does_fit

    @property
    def is_child(self):
        """
        Check if the current instance is a child.

        Returns:
            bool: True if the instance has a parent, False otherwise.
        """
        return self._parent is not None

    def points_in_region(self, x, y, analysis_type=AnalysisType.Full):
        """Returns the indices of the points specified by 'x' and 'y' that fall into this region.

        Parameters
        ----------
        x : np.ndarray
            the x positions
        y : np.ndarray
            the y positions
        analysis_type : AnalysisType, optional
            defines how the positions are evaluated, by default AnalysisType.Full
            FIXME: some of this can probably be solved using linear algebra, what with multiple exact same points?
        """
        if self._shape_type == RegionShape.Rectangular or (
            self._shape_type == RegionShape.Circular
            and analysis_type != AnalysisType.Full
        ):
            if analysis_type == AnalysisType.Full:
                indices = np.where(
                    ((y >= self._min_extent[1]) & (y <= self._max_extent[1]))
                    & ((x >= self._min_extent[0]) & (x <= self._max_extent[0]))
                )[0]
                indices = np.array(indices, dtype=int)
            elif analysis_type == AnalysisType.CollapseX:
                x_indices = np.where(
                    (x >= self._min_extent[0]) & (x <= self._max_extent[0])
                )[0]
                indices = np.asarray(x_indices, dtype=int)
            else:
                y_indices = np.where(
                    (y >= self._min_extent[1]) & (y <= self._max_extent[1])
                )[0]
                indices = np.asarray(y_indices, dtype=int)
        else:
            if self.is_child:
                mask = self.circular_mask(
                    self._parent.position[2],
                    self._parent.position[3],
                    self._origin,
                    self._extent,
                )
            else:
                mask = self.circular_mask(
                    self.position[2], self.position[3], self._origin, self._extent
                )
            img = np.zeros_like(mask)
            img[np.asarray(y, dtype=int), np.asarray(x, dtype=int)] = 1
            temp = np.where(img & mask)
            indices = []
            for i, j in zip(list(temp[1]), list(temp[0])):
                matches = np.where((x == i) & (y == j))
                if len(matches[0]) == 0:
                    continue
                indices.append(matches[0][0])
            indices = np.array(indices)
        return indices

    def time_in_region(self, x, y, time, analysis_type=AnalysisType.Full):
        """Returns the entering and leaving times at which the animal entered
        and left a region. In case the animal was not observed after entering
        this region (for example when hidden in a tube) the leaving time is
        the maximum time entry.
        Whether the full position, or only the x- or y-position should be considered 
        is controlled with the analysis_type parameter.

        Parameters
        ----------
        x : np.ndarray
        The animal's x-positions
        y : np.ndarray
            the animal's y-positions
        time : np.ndarray
            the time array
        analysis_type : AnalysisType, optional
            The type of analysis, by default AnalysisType.Full

        Returns
        -------
        np.ndarray
            The entering times
        np.ndarray
            The leaving times
        """
        indices = self.points_in_region(x, y, analysis_type)
        if len(indices) == 0:
            return np.array([]), np.array([])

        diffs = np.diff(indices)
        if len(diffs) == sum(diffs):
            entering = [time[indices[0]]]
            leaving = [time[indices[-1]]]
        else:
            entering = []
            leaving = []
            jumps = np.where(diffs > 1)[0]
            start = time[indices[0]]
            for i in range(len(jumps)):
                end = time[indices[jumps[i]]]
                entering.append(start)
                leaving.append(end)
                start = time[indices[jumps[i] + 1]]

            end = time[indices[-1]]
            entering.append(start)
            leaving.append(end)
        return np.array(entering), np.array(leaving)

    def patch(self, **kwargs):
        """
        Create and return a matplotlib patch object based on the shape type of the arena.

        Parameters:
        - kwargs: Additional keyword arguments to customize the patch object.

        Returns:
        - A matplotlib patch object representing the arena shape.

        If the 'fc' (facecolor) keyword argument is not provided, it will default to None.
        If the 'fill' keyword argument is not provided, it will default to False.

        For rectangular arenas, the patch object will be a Rectangle with width and height
        based on the arena's position.
        For circular arenas, the patch object will be a Circle with radius based on the
        arena's extent.

        Example usage:
        ```
        arena = Arena()
        patch = arena.patch(fc='blue', fill=True)
        ax.add_patch(patch)
        ```
        """
        if "fc" not in kwargs:
            kwargs["fc"] = None
            kwargs["fill"] = False
        if self._shape_type == RegionShape.Rectangular:
            w = self.position[2]
            h = self.position[3]
            return patches.Rectangle(self._origin, w, h, **kwargs)
        else:
            return patches.Circle(self._origin, self._extent, **kwargs)

    def __repr__(self):
        return f"Region: '{self._name}' of {self._shape_type} shape."


class Arena(Region):
    """
    Class to represent the experimental arena. Arena is derived from Region and can be either rectangular or circular. 
    An arena can not have a parent.
    See Region for more details.
    """
    def __init__(self, origin, extent, inverted_y=True, name="", arena_shape=RegionShape.Rectangular, 
                 illumination=Illumination.Backlight) -> None:
        """ Construct a new Area with a given origin and extent.


        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        super().__init__(origin, extent, inverted_y, name, arena_shape)
        self._illumination = illumination
        self.regions = {}

    def add_region(
        self, name, origin, extent, shape_type=RegionShape.Rectangular, region=None
    ):
        if name is None or name in self.regions.keys():
            raise ValueError(
                "Region name '{name}' is invalid. The name must not be None and must be unique among the regions."
            )
        if region is None:
            region = Region(
                origin, extent, name=name, region_shape=shape_type, parent=self
            )
        else:
            region._parent = self
        doesfit = self.fits(region)
        if not doesfit:
            logging.warn(
                f"Warning! Region {region.name} with size {region.position} does fit into {self.name} with size {self.position}!"
            )
        self.regions[name] = region

    def remove_region(self, name):
        """
        Remove a region from the arena.

        Parameter:
            name : str
            The name of the region to remove.

        Returns:
            None
        """
        if name in self.regions:
            self.regions.pop(name)

    def __repr__(self):
        return f"Arena: '{self._name}' of {self._shape_type} shape."

    def plot(self, axis=None):
        """
        Plots the arena on the given axis.

        Parameters
        ----------
        - axis (matplotlib.axes.Axes, optional): The axis on which to plot the arena. If not provided, a new figure and axis will be created.

        Returns
        -------
        - matplotlib.axes.Axes: The axis on which the arena is plotted.
        """
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        axis.add_patch(self.patch())
        axis.set_xlim([self._origin[0], self._max_extent[0]])

        if self.inverted_y:
            axis.set_ylim([self._max_extent[1], self._origin[1]])
        else:
            axis.set_ylim([self._origin[1], self._max_extent[1]])
        for r in self.regions:
            axis.add_patch(self.regions[r].patch())
        return axis

    def region_vector(self, x, y):
        """Returns a vector that contains the region names within which the agent was found.
            FIXME: This does not work well with overlapping regions!@!
        Parameters
        ----------
        x : np.array
            the x-positions
        y : np.ndarray
            the y-positions

        Returns
        -------
        np.array
            vector of the same size as x and y. Each entry is the region to which the position is assigned to. If the point is not assigned to a region, the entry will be empty.
        """
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        rv = np.empty(x.shape, dtype=str)
        for r in self.regions:
            indices = self.regions[r].points_in_region(x, y)
            rv[indices] = r
        return rv

    def in_region(self, x, y):
        """
        Determines if the given coordinates (x, y) are within any of the defined regions in the arena.

        Parameters
        ----------
        x : float
        The x-coordinate of the point to check.
        y : float
        The y-coordinate of the point to check.

        Returns
        -------
        dict: 
        A dictionary containing the region names as keys and a list of indices of points within each region as values.
        """
        tmp = {}
        for r in self.regions:
            print(r)
            indices = self.regions[r].points_in_region(x, y)
            tmp[r] = indices
        return tmp

    def __getitem__(self, key):
        if isinstance(key, (str)):
            return self.regions[key]
        else:
            return self.regions[self.regions.keys()[key]]


if __name__ == "__main__":
    a = Arena((0, 0), (1024, 768), name="arena", arena_shape=RegionShape.Rectangular)
    a.add_region("small rect1", (0, 0), (100, 300))
    a.add_region("small rect2", (150, 0), (100, 300))
    a.add_region("small rect3", (300, 0), (100, 300))
    a.add_region("circ", (600, 400), 150, shape_type=RegionShape.Circular)
    axis = a.plot()
    x = np.linspace(a.position[0], a.position[0] + a.position[2] - 1, 100, dtype=int)
    y = np.asarray(
        (np.sin(x * 0.01) + 1) * a.position[3] / 2 + a.position[1] - 1, dtype=int
    )
    # y = np.linspace(a.position[1], a.position[1] + a.position[3] - 1, 100, dtype=int)
    axis.scatter(x, y, c="k", s=2)

    ind = a.regions[3].points_in_region(x, y)
    if len(ind) > 0:
        axis.scatter(x[ind], y[ind], label="circ full")

    ind = a.regions[3].points_in_region(x, y, AnalysisType.CollapseX)
    if len(ind) > 0:
        axis.scatter(x[ind], y[ind] - 10, label="circ collapseX")

    ind = a.regions[3].points_in_region(x, y, AnalysisType.CollapseY)
    if len(ind) > 0:
        axis.scatter(x[ind], y[ind] + 10, label="circ collapseY")

    ind = a.regions[0].points_in_region(x, y, AnalysisType.CollapseX)
    if len(ind) > 0:
        axis.scatter(x[ind], y[ind] - 10, label="rect collapseX")

    ind = a.regions[1].points_in_region(x, y, AnalysisType.CollapseY)
    if len(ind) > 0:
        axis.scatter(x[ind], y[ind] + 10, label="rect collapseY")

    ind = a.regions[2].points_in_region(x, y, AnalysisType.Full)
    if len(ind) > 0:
        axis.scatter(x[ind], y[ind] + 20, label="rect full")
    axis.legend()
    plt.show()

    a.plot()
    plt.show()
