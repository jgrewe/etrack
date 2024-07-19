import pytest
import numpy as np
import etrack as et

dlcdataset = "test/dlc_testfile.h5"
slpdataset = "test/sleap_testfile.nix"


@pytest.fixture
def dlc_data():
    # Create a NixTrackData object with some test data
    return et.TrackingResult(dlcdataset)


@pytest.fixture
def slp_data():
    # Create a NixTrackData object with some test data
    return et.TrackingResult(slpdataset)

def test_readdata(dlc_data, slp_data):
    with pytest.raises:
        et.TrackingResult("invalid_file.nix")
    
    # tr = et.TrackingResult(dlcdataset, filetype=et.FileType.Auto)
