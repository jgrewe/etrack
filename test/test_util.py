import pytest
import numpy as np
import etrack as et
from IPython import embed

dataset = "test/sleap_testfile.nix"


@pytest.fixture
def nixtrack_data():
    # Create a NixTrackData object with some test data
    return et.NixtrackData(dataset)


def test_radianstocompass():
    radians = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
    compass = et.util.randianstocompass(radians)
    assert np.round(compass[0]) == 90
    assert np.round(compass[1]) == 0
    assert np.round(compass[2]) == 270
    assert np.round(compass[3]) == 180

def test_aligndetections():
    # test for cases in which the sampling is not the same
    # test for cases in which there is no temporal overlap
    # test for the normal, well behaving case with temporal overlap
    pass


def test_bodyorientation(nixtrack_data):
    front = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]).T
    back = np.array([[0, 0, 1, 0], [0, 0, 0, 1]]).T
    goal_degrees = np.array([90, 0, 270, 180])
    goal_radians = np.array([0, np.pi/2, np.pi, -np.pi/2])
    degrees = et.util.body_orientation(front, back, orientation=et.Orientation.Compass)
    radians = et.util.body_orientation(front, back, orientation=et.Orientation.Radians)
    for d, r, gd,gr in zip(degrees, radians, goal_degrees, goal_radians):
        assert d == gd
        assert r == gr

    spatial_limits = (350, 35, 3600-350, 450)
    quality_threshold = 0.15

    snout = nixtrack_data.track_data(bodypart="snout")
    snout.position_limits = spatial_limits
    snout.quality_threshold = quality_threshold
    snout.filter_tracks()
    fx, fy, ft, _ = snout.interpolate()
    fpos = np.vstack((fx, fy)).T

    center = nixtrack_data.track_data(bodypart="center")
    center.position_limits = spatial_limits
    center.quality_threshold = quality_threshold
    center.filter_tracks()
    bx, by, bt, _ = center.interpolate()
    bpos = np.vstack((bx, by)).T

    common_t, common_fpos, common_bpos = et.util.align_detections(fpos, ft, bpos, bt)
    orientation = et.util.body_orientation(common_fpos, common_bpos, orientation=et.Orientation.Compass)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.scatter(common_fpos[:, 0], common_fpos[:, 1], c="g")
    ax1.scatter(common_bpos[:, 0], common_bpos[:, 1], c="r")
    for i in range(len(common_bpos)):
        ax1.plot([common_fpos[i, 0], common_bpos[i, 0]], [common_fpos[i, 1], common_bpos[i, 1]], lw=0.5)
    ax2.plot(common_t, np.unwrap(orientation, period=360))
    plt.show()


if __name__ == "__main__":
    pytest.main()