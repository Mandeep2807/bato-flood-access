"""AOI coverage fraction (area-weighted grid)."""

from __future__ import annotations

import numpy as np
import pytest

from bato.acquisition.coverage import AOICoverageGrid, points_in_ring
from bato.contracts.common import PolygonGeometry


def poly(*points: tuple[float, float]) -> PolygonGeometry:
    return PolygonGeometry(coordinates=((*points, points[0]),))


AOI_SQUARE = poly((85.20, 28.00), (85.40, 28.00), (85.40, 28.20), (85.20, 28.20))


@pytest.fixture(scope="module")
def grid() -> AOICoverageGrid:
    return AOICoverageGrid(AOI_SQUARE, 200)


def test_full_partial_and_no_coverage(grid):
    assert grid.fraction([poly((84, 27), (87, 27), (87, 30), (84, 30))]) == 1.0
    assert grid.fraction([poly((85.30, 27), (87, 27), (87, 30), (85.30, 30))]) == pytest.approx(0.5, abs=0.005)
    assert grid.fraction([poly((86, 27), (87, 27), (87, 30), (86, 30))]) == 0.0


def test_union_of_overlapping_slices_is_not_double_counted(grid):
    south = poly((84, 27), (87, 27), (87, 28.12), (84, 28.12))
    north = poly((84, 28.08), (87, 28.08), (87, 30), (84, 30))
    assert grid.fraction([south]) + grid.fraction([north]) > 1.0
    assert grid.fraction([south, north]) == 1.0


def test_footprint_inside_aoi_gives_area_ratio(grid):
    quarter = poly((85.20, 28.00), (85.30, 28.00), (85.30, 28.10), (85.20, 28.10))
    assert grid.fraction([quarter]) == pytest.approx(0.25, abs=0.005)


def test_non_convex_aoi():
    l_shape = poly((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2))   # area 3 square degrees
    g = AOICoverageGrid(l_shape, 300)
    top_left = poly((-1, 1), (1, 1), (1, 3), (-1, 3))               # covers the 1x1 upper arm
    assert g.fraction([top_left]) == pytest.approx(1 / 3, abs=0.01)


def test_area_weighting_by_latitude():
    tall = poly((10, 0), (11, 0), (11, 60), (10, 60))
    g = AOICoverageGrid(tall, 400)
    lower_half = poly((9, -1), (12, -1), (12, 30), (9, 30))
    # Equal latitude spans but unequal areas: sin(30)/sin(60) of the strip lies below 30 N.
    expected = np.sin(np.radians(30)) / np.sin(np.radians(60))
    assert g.fraction([lower_half]) == pytest.approx(expected, abs=0.01)


def test_points_in_ring_vectorised():
    ring = ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))
    inside = points_in_ring(np.array([0.5, 1.5, 0.25]), np.array([0.5, 0.5, 0.75]), ring)
    assert inside.tolist() == [True, False, True]


def test_invalid_grid_size():
    with pytest.raises(ValueError):
        AOICoverageGrid(AOI_SQUARE, 1)
