import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp

from etrack import Arena, Region, RegionShape


def test_region():
    # Create a parent region
    parent_region = Region((0, 0), (100, 100), name="parent", region_shape=RegionShape.Rectangular)

    # Create a child region
    child_region = Region((10, 10), (50, 50), name="child", region_shape=RegionShape.Rectangular, parent=parent_region)

    # Test properties
    assert child_region.name == "child"
    assert child_region.inverted_y == True
    assert (child_region._max_extent == np.array((60, 60))).all()
    assert (child_region._min_extent == np.array((10, 10))).all()
    assert child_region.xmax == 60
    assert child_region.xmin == 10
    assert child_region.ymin == 10
    assert child_region.ymax == 60
    assert child_region.position == (10, 10, 50, 50)
    assert child_region.is_child == True

    # Test fits method
    assert parent_region.fits(child_region) == True

    # Test points_in_region method
    x = [15, 20, 25, 30, 35, 5]
    y = [15, 20, 25, 30, 35, 5]
    assert (child_region.points_in_region(x, y) == np.array([0, 1, 2, 3, 4])).all()

    # Test time_in_region method
    x = [5, 15, 20, 25, 30, 35, 35]
    y = [5, 15, 20, 25, 30, 35, 65]
    time = np.arange(0, len(x), 1)
    enter, leave = child_region.time_in_region(x, y, time)
    assert enter[0] == 1
    assert leave[0] == 5

    # Test patch method
    patch = child_region.patch(color='red')
    assert isinstance(patch, mp.Patch)

    # Test __repr__ method
    assert repr(child_region) == "Region: 'child' of Rectangular shape."


def test_arena():
    # Create an arena
    arena = Arena((0, 0), (100, 100), name="arena", arena_shape=RegionShape.Rectangular)
    # Test add_region method
    arena.add_region("small rect1", (0, 0), (50, 50))
    assert len(arena.regions) == 1
    assert arena.regions["small rect1"].name == "small rect1"
    # Test remove_region method
    arena.remove_region("small rect1")
    assert len(arena.regions) == 0
    # Test plot method
    axis = arena.plot()
    assert isinstance(axis, plt.Axes)
    # Test region_vector method
    x = [10, 20, 30]
    y = [10, 20, 30]
    assert (arena.region_vector(x, y) == "").all()

    # Test in_region method
    # assert len(arena.in_region(10, 10)) > 0
    # print(arena.in_region(10, 10))

    # print(arena.in_region(110, 110))
    # assert arena.in_region(110, 110) == False
    # Test __getitem__ method
    arena.add_region("small rect2", (0, 0), (50, 50))
    assert arena["small rect2"].name == "small rect2"


if __name__ == "__main__":
    pytest.main()