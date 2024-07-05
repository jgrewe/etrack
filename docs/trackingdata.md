# TrackingData

Class that represents the position data associated with one node/bodypart.
The class encapsulates the data and offers several hepler functions for filtering, estimating the traveling speed, the movent direction etc. Refer to the API documentation for more information.

```python
import etrack as et

dataset = "/test/some_tracking_data.nix"
ntd = et.NixtrackData(dataset)
td = ntd.track_data(bodypart="snout")
```

The variable ``td`` is an instance of the ``TrackingData`` class that holds the tracked positions, the respective time vector, and the quality score associated with the node for each instance, it has been found. Depending on the way the data is stored, there may be empty (0, 0) coordinates when the node was not found in the respective frame.

## Simple plot of the tracking data

To create a simple plot of the tracking results you can

```python
import matplotlib.pyplot as plt

x, y, time, score = td.positions()

plot.scatter(x, y, c=score)
plt.show()
```

![scatter of tracking data](intro_scatter.png)

You may notice that there are gaps in the detection and a dot at ``(0,0)`` that should not be there. The color encodes the score, i.e. some detections are of lower quality than others.

## Filtering of the data

``TrackingData``offers the opportunity to filter the data. There are three predefined filter functions according to:

* temporal limits
* position limits
* quality threshold

```python
td.quality_threshold = 0.8
td.position_limits = (400, 50, 3100, 450)
td.filter_tracks()

plt.scatter(x, y, c=time)
plt.xlabel("x-position")
plt.ylabel("y-position")
plt.show()
```

![scatter of tracking data after filtering](intro_scatter2.png)

Several data points have been kicked out mainly due to the quality filter which in this case is just a scalar value ``0.8``. To set the position filter you have to provide a 4-tuple with entries for ``(x, y, width, height)`` of the region of interest.

To apply the filter(s) call the ``filter_tracks()`` function. To unset a filter, set its value to ``None`` and re-apply by calling ``filter_tracks()`` again.

## Interpolation

