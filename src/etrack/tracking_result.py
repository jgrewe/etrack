import os 
import pathlib
import numpy as np
import pandas as pd 
import numbers as nb
import matplotlib.pyplot as plt

from .util import FileType
"""
x_0 = 0
width = 1230
y_0 = 0 
height = 1100
x_factor = 0.81/width  # Einheit m/px 
y_factor = 0.81/height  # Einheit m/px 
center = (np.round(x_0 + width/2), np.round(y_0 + height/2))
center_meter = ((center[0] - x_0) * x_factor, (center[1] - y_0) * y_factor)
"""
def coordinate_transformation(position,x_0, y_0, x_factor, y_factor):
        x = (position[0] - x_0) * x_factor
        y = (position[1] - y_0) * y_factor
        return (x, y) #in m 

class TrackingResult(object):

    def __init__(self, results_file, x_0=0, y_0= 0, width_pixel=1230, height_pixel=1100, width_meter=0.81, height_meter=0.81, filetype=FileType.Auto) -> None:
        super().__init__()
        if not pathlib.Path.exists(results_file):
            raise ValueError("File %s does not exist!" % results_file)
        self._file_name = results_file
        self.x_0 = x_0
        self.y_0 = y_0
        self.width_pix = width_pixel
        self.width_m = width_meter
        self.height_pix = height_pixel
        self.height_m = height_meter
        self.x_factor = self.width_m / self.width_pix # m/pix
        self.y_factor = self.height_m / self.height_pix # m/pix

        self.center = (np.round(self.x_0 + self.width_pix/2), np.round(self.y_0 + self.height_pix/2))
        self.center_meter = ((self.center[0] - self.x_0) * self.x_factor, (self.center[1] - self.y_0) * self.y_factor)

        if filetype == FileType.Auto:
            p = pathlib.PosixPath(self._file_name)
                           
        self._data_frame = pd.read_hdf(results_file)
        self._level_shape = self._data_frame.columns.levshape
        self._scorer = self._data_frame.columns.levels[0].values
        self._bodyparts = self._data_frame.columns.levels[1].values if self._level_shape[1] > 0 else []
        self._positions = self._data_frame.columns.levels[2].values if self._level_shape[2] > 0 else []

    def angle_to_center(self, bodypart=0, twopi=True, origin="topleft", min_likelihood=0.95):
        if  isinstance(bodypart, nb.Number):
            bp = self._bodyparts[bodypart]
        elif isinstance(bodypart, str) and bodypart in self._bodyparts:
            bp = bodypart
        else:
            raise ValueError("Bodypart %s is not in dataframe!" % bodypart)
        _, x, y, _, _ = self.position_values(bodypart=bp, min_likelihood=min_likelihood)
        if x is None:
            print("Error: no valid angles for %s" % self._file_name)
            return []
        x_meter = x - self.center_meter[0]
        y_meter = y - self.center_meter[1]
        if origin.lower() == "topleft":
            y_meter *= -1
        phi = np.arctan2(y_meter, x_meter) * 180 / np.pi
        if twopi:
            phi[phi < 0] = 360 + phi[phi < 0]
        return phi

    def coordinate_transformation(self, position):
        x = (position[0] - self.x_0) * self.x_factor
        y = (position[1] - self.y_0) * self.y_factor
        return (x, y) #in m 

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

    @property
    def positions(self):
        return self._positions

    def position_values(self, scorer=0, bodypart=0, framerate=30, interpolate=True, min_likelihood=0.95):
        """returns the x and y positions in m and the likelihood of the positions.

        Args:
            scorer (int, optional): [description]. Defaults to 0.
            bodypart (int, optional): [description]. Defaults to 0.
            framerate (int, optional): [description]. Defaults to 30.

        Raises:
            ValueError: [description]
            ValueError: [description]

        Returns:
            time [np.array]: the time axis
            x [np.array]: the x-position in m 
            y [np.array]: the y-position in m
            l [np.array]: the likelihood of the position estimation
            bp string: the body part
            [type]: [description]
        """
        time, x, y, l, bp = self.pixel_positions(scorer, bodypart, framerate, interpolate, min_likelihood)
        x, y = self._to_meter(x, y)
        return time, x, y, l, bp

    def pixel_positions(self, scorer=0, bodypart=0, framerate=30, interpolate=True, min_likelihood=0.95):
        if  isinstance(scorer, nb.Number):
            sc = self._scorer[scorer]
        elif isinstance(scorer, str) and scorer in self._scorer:
            sc = scorer
        else:
            raise ValueError("Scorer %s is not in dataframe!" % scorer)
        if  isinstance(bodypart, nb.Number):
            bp = self._bodyparts[bodypart]
        elif isinstance(bodypart, str) and bodypart in self._bodyparts:
            bp = bodypart
        else:
            raise ValueError("Bodypart %s is not in dataframe!" % bodypart)

        x = np.asarray(self._data_frame[sc][bp]["x"] if "x" in self._positions else [])
        y = np.asarray(self._data_frame[sc][bp]["y"] if "y" in self._positions else [])
        l = np.asarray(self._data_frame[sc][bp]["likelihood"] if "likelihood" in self._positions else [])

        time = np.arange(len(x))/framerate
        if interpolate:
            x, y = self.interpolate(time, x, y, l, min_likelihood)
        return time, x, y, l, bp

    def _to_meter(self, x, y):
        new_x = (np.asarray(x) - self.x_0) * self.x_factor
        new_y = (np.asarray(y) - self.y_0) * self.y_factor
        return new_x, new_y

    def _speed(self, t, x, y):
        speed = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / np.diff(t)
        return speed

    def interpolate(self, t, x, y, l, min_likelihood=0.9):
        time2 = t[l > min_likelihood]
        if len(l[l > min_likelihood]) < 10:
            print("%s has less than 10 datapoints with likelihood larger than %.2f" % (self._file_name, min_likelihood) )
            return None, None
        x2 = x[l > min_likelihood]
        y2 = y[l > min_likelihood]
        x3 = np.interp(t, time2, x2)
        y3 = np.interp(t, time2, y2)
        return x3, y3

    def plot(self, scorer=0, bodypart=0, threshold=0.9, framerate=30):
        t, x, y, l, name  = self.position_values(scorer=scorer, bodypart=bodypart, framerate=framerate, min_likelihood=threshold)
        plt.scatter(x[l > threshold], y[l > threshold], c=t[l > threshold], label=name)
        plt.scatter(self.center_meter[0], self.center_meter[1], marker="*")
        plt.plot(x[l > threshold], y[l > threshold])
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.gca().invert_yaxis()
        bar = plt.colorbar()
        bar.set_label("time [s]")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    from IPython import embed
    filename = "2020.12.04_lepto48DLC_resnet50_boldnessDec11shuffle1_200000.h5"
    path = "/mnt/movies/merle_verena/boldness/labeled_videos/day_4/"
    tr = TrackingResult(path+filename)
    time, x, y, l, bp = tr.position_values(bodypart=2)


    thresh = 0.95
    time2 = time[l>thresh]
    x2 = x[l>thresh]
    y2 = y[l>thresh]
    x3 = np.interp(time, time2, x2)
    y3 = np.interp(time, time2, y2)

    fig, axes = plt.subplots(3,1, sharex=True)
    axes[0].plot(time, x)
    axes[0].plot(time, x3)
    axes[1].plot(time, y)
    axes[1].plot(time, y3)

    axes[2].plot(time, l)
    plt.show()

    embed()