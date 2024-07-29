import pytest
import etrack as et

dlcdataset = "test/dlc_testfile.h5"
slpdataset = "test/sleap_testfile.nix"

@pytest.fixture
def dlc_data():
    # Create a NixTrackData object with some test data
    return et.read_dataset(dlcdataset, et.FileType.Deeplabcut)


@pytest.fixture
def slp_data():
    # Create a NixTrackData object with some test data
    return et.read_dataset(slpdataset, et.FileType.Sleap)


def test_readdata(dlc_data, slp_data):
    with pytest.raises(FileExistsError):
        et.read_dataset("invalid_file.nix", et.FileType.Deeplabcut)

    assert isinstance(dlc_data, et.DLCReader)
    assert isinstance(slp_data, et.NixtrackReader)
    
    # from IPython import embed
    # embed()