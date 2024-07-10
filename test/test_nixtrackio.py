import pytest
import numpy as np
import etrack as et
from IPython import embed

dataset = "test/2022lepto01_converted_2024.03.27_0.mp4.nix"


@pytest.fixture
def nixtrack_data():
    # Create a NixTrackData object with some test data
    return et.NixtrackData(dataset)


def test_basics(nixtrack_data):
    assert nixtrack_data.filename == dataset
    assert len(nixtrack_data.bodyparts) == 5
    assert len(nixtrack_data.tracks) == 2
    assert nixtrack_data.fps == 25


def test_trackingdata(nixtrack_data):
    with pytest.raises(ValueError):
        nixtrack_data.track_data(bodypart="test")
        nixtrack_data.track_data(track="fish")

    assert nixtrack_data.track_data("center") is not None
    assert nixtrack_data.track_data("center", "none") is not None


if __name__ == "__main__":
    pytest.main()