import pytest
import numpy as np
import etrack as et

dataset = "test/sleap_testfile.nix"

@pytest.fixture
def td():
    # Create a TrackingData object with some test data
    ntd = et.NixtrackData(dataset)
    data = ntd.track_data(bodypart="snout")
    return data


def test_interpolate(td):
    td.position_limits = (350, 35, 3600-350, 450)
    td.quality_threshold = 0.15
    td.filter_tracks()
    xi, yi, ti, i = td.interpolate()

    assert ti[0] == np.round(td._time[0], 4)
    assert ti[-1] == np.round(td._time[-1], 4)

    assert xi[0] == td._x[0]
    assert yi[0] == td._y[0]

    assert xi[-1] == td._x[-1]
    assert yi[-1] == td._y[-1]

    assert len(i) >= len(td._time)
    assert len(i) == len(xi) == len(yi) == len(ti)

    assert sum(i) == len(i) - len(td._x)


def test_movementdirection(td):
    direction = td.movement_direction(orientation=et.Orientation.Radians)
    assert len(direction) == len(td._x) - 1 == len(td._y) -1
    assert np.min(direction) >= -np.pi
    assert np.max(direction) <= np.pi

    direction2 = td.movement_direction(orientation=et.Orientation.Compass)
    assert len(direction2) == len(td._x) - 1 == len(td._y) -1
    assert np.min(direction2) >= 0
    assert np.max(direction2) <= 360

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(211, projection="polar")
    ax2 = fig.add_subplot(212, projection="polar")
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)

    ax.hist(direction)
    ax2.hist(direction2 / 360*2*np.pi)
    plt.show()


if __name__ == "__main__":
    pytest.main()