import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.draw import disk

from .util import RegionShape, AnalysisType, Illumination


class Region(object):

    def __init__(self, origin, extent, inverted_y=True, name="", region_shape=RegionShape.Rectangular, parent=None) -> None:
        assert len(origin) == 2
        self._origin = origin
        self._extent = extent
        self._inverted_y = inverted_y
        self._name = name
        self._shape_type = region_shape
        self._check_extent(extent)
        self._parent = parent

    @staticmethod
    def circular_mask(width, height, center, radius):
        assert center[1] + radius < width and center[1] - radius > 0
        assert center[0] + radius < height and center[0] - radius > 0

        mask = np.zeros((height, width), dtype=np.uint8)
        rr, cc = disk(reversed(center), radius)
        mask[rr, cc] = 1

        return mask

    @property
    def _max_extent(self):
        if self._shape_type == RegionShape.Rectangular:
            max_extent = (self._origin[0] + self._extent[0], self._origin[1] + self._extent[1])
        else:
            max_extent = (self._origin[0] + self._extent, self._origin[1] + self._extent)
        return max_extent

    @property
    def _min_extent(self):
        if self._shape_type == RegionShape.Rectangular:
            min_extent = self._origin
        else:
            min_extent = (self._origin[0] - self._extent, self._origin[1] - self._extent)
        return min_extent

    @property
    def position(self):
        """Returns the position and extent of the region as 4-tuple, (x, y, width, height)
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
                raise ValueError("Extent must be a length 2 list or tuple for rectangular regions!")
        elif self._shape_type == RegionShape.Circular:
            if not isinstance(ext, (int, float)):
                raise ValueError("Extent must be a numerical scalar for circular regions!")
        else:
            raise ValueError(f"Invalid ShapeType, {self._shape_type}!")

    def fits(self, other) -> bool:
        """
            Returns true if the other region fits inside this region!
        """
        assert isinstance(other, Region)
        does_fit = all((other._min_extent[0] >= self._min_extent[0], other._min_extent[1] >= self._min_extent[1], 
                        other._max_extent[0] <= self._max_extent[0], other._max_extent[1] <= self._max_extent[1]))
        return does_fit

    @property
    def is_child(self):
        return self._parent is not None

    def points_in_region(self, x, y, analysis_type=AnalysisType.Full):
        """returns the indices of the points specified by 'x' and 'y' that fall into this region.

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
        if self._shape_type == RegionShape.Rectangular or (self._shape_type == RegionShape.Circular and analysis_type != AnalysisType.Full):
            if analysis_type == AnalysisType.Full:
                indices = np.where(((y >= self._min_extent[1]) & (y <= self._max_extent[1])) & 
                                   ((x >= self._min_extent[0]) & (x <= self._max_extent[0])))[0]
                indices = np.array(indices, dtype=int)
            elif analysis_type == AnalysisType.CollapseX:
                x_indices = np.where((x >= self._min_extent[0]) & (x <= self._max_extent[0] ))[0]
                indices = np.asarray(x_indices, dtype=int)
            else:
                y_indices = np.where((y >= self._min_extent[1]) & (y <= self._max_extent[1] ))[0]
                indices = np.asarray(y_indices, dtype=int)
        else:
            if self.is_child:
                mask = self.circular_mask(self._parent.position[2], self._parent.position[3], self._origin, self._extent)
            else:
                mask = self.circular_mask(self.position[2], self.position[3], self._origin, self._extent)
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
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        y : _type_
            _description_
        time : _type_
            _description_
        analysis_type : _type_, optional
            _description_, by default AnalysisType.Full

        Returns
        -------
        _type_
            _description_
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
            jumps  = np.where(diffs > 1)[0]
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

    def __init__(self, origin, extent, inverted_y=True, name="", arena_shape=RegionShape.Rectangular,
                 illumination=Illumination.Backlight) -> None:
        super().__init__(origin, extent, inverted_y, name, arena_shape)
        self._illumination = illumination
        self.regions = {}

    def add_region(self, name, origin, extent, shape_type=RegionShape.Rectangular, region=None):
        if name is None or name in self.regions.keys():
            raise ValueError("Region name '{name}' is invalid. The name must not be None and must be unique among the regions.") 
        if region is None:
            region = Region(origin, extent, name=name, region_shape=shape_type, parent=self)
        else:
            region._parent = self
        if self.fits(region):
            self.regions[name] = region
        else:
            Warning(f"Region {region} fits not! Not added to the list of regions!")

    def remove_region(self, name):
        if name in self.regions:
            self.regions.pop(name)

    def __repr__(self):
        return f"Arena: '{self._name}' of {self._shape_type} shape."

    def plot(self, axis=None):
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        axis.add_patch(self.patch())
        axis.set_xlim([self._origin[0], self._max_extent[0]])
        axis.set_ylim([self._origin[1], self._max_extent[1]])
        for r in self.regions:
            axis.add_patch(r.patch())
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
            vector of the same size as x and y. Each entry is the region to which the position is assinged to. If the point is not assigned to a region, the entry will be empty.
        """
        rv = np.empty(x.shape, dtype=str)
        for r in self.regions:
            indices = self.regions[r].points_in_region(x, y)
            rv[indices] = r
        return rv

    def in_region(self, x, y):
        tmp = {}
        for r in self.regions:
            indices = self.regions[r].points_in_region(x, y)
            tmp[r] = indices
        return tmp

if __name__ == "__main__":
    a = Arena((0, 0), (1024, 768), name="arena", arena_shape=RegionShape.Rectangular)
    a.add_region("small rect1", (0, 0), (100, 300))
    a.add_region("small rect2", (150, 0), (100, 300))
    a.add_region("small rect3", (300, 0), (100, 300))
    a.add_region("circ", (600, 400), 150, shape_type=RegionShape.Circular)
    axis = a.plot()
    x = np.linspace(a.position[0], a.position[0] + a.position[2] - 1, 100, dtype=int)
    y = np.asarray((np.sin(x*0.01) + 1) * a.position[3] / 2 + a.position[1] -1, dtype=int)
    #y = np.linspace(a.position[1], a.position[1] + a.position[3] - 1, 100, dtype=int)
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
        axis.scatter(x[ind], y[ind]-10, label="rect collapseX")

    ind = a.regions[1].points_in_region(x, y, AnalysisType.CollapseY)
    if len(ind) > 0:
        axis.scatter(x[ind], y[ind] + 10, label="rect collapseY")

    ind = a.regions[2].points_in_region(x, y, AnalysisType.Full)
    if len(ind) > 0:
        axis.scatter(x[ind], y[ind]+20, label="rect full")
    axis.legend()
    plt.show()

    a.plot()
    plt.show()