import pytest
import numpy as np
import etrack as et


dataset = "test/2022lepto01_converted_2024.03.27_0.mp4.nix"

@pytest.fixture
def td():
    # Create a TrackingData object with some test data
    ntd = et.NixtrackData(dataset)
    td = ntd.track_data(bodypart="snout")
    return td


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
    assert (len(i)) == len(xi) == len(yi)

    assert sum(i) == len(i) - len(td._x)

if __name__ == "__main__":
    pytest.main()