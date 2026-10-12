"""Road topology, segmentation and three-state road impact (synthetic geometry, TEST FIXTURE)."""

from __future__ import annotations

import numpy as np
import pytest

from bato.contracts.baseline import BaselineState
from bato.contracts.common import BBox
from bato.contracts.impact import RoadState
from bato.impact.roads import (
    OUTSIDE_RASTER,
    SAMPLE_FLOOD,
    SAMPLE_OPEN,
    SAMPLE_UNOBSERVABLE,
    RoadWay,
    StateSampler,
    assess_network,
    build_network,
    dilate,
    haversine_m,
    is_bridge,
    parse_roads,
    road_buffer_radius_px,
)

BOX = BBox(min_lon=85.0, min_lat=28.0, max_lon=85.1, max_lat=28.1)


def way(osm_id: int, coords, **tags) -> RoadWay:
    return RoadWay(osm_id=osm_id, tags=tags, parts=(tuple(coords),))


def test_shared_vertices_become_junctions_and_crossings_without_vertex_do_not():
    a = way(1, [(85.01, 28.05), (85.05, 28.05), (85.09, 28.05)])
    b = way(2, [(85.05, 28.05), (85.05, 28.09)])                    # meets a at a shared vertex
    c = way(3, [(85.03, 28.02), (85.03, 28.08)], bridge="yes")      # crosses a, no shared vertex (bridge)
    net = build_network([a, b, c], 50.0, BOX)
    assert len(net.edges) == 4                                      # a split in two at the junction, b, c
    assert net.junction_nodes == 6                                  # 5 ends + 1 junction ... and c's 2 ends
    deg: dict[int, int] = {}
    for s in net.segments:
        deg[s.u] = deg.get(s.u, 0) + 1
        deg[s.v] = deg.get(s.v, 0) + 1
    assert max(deg.values()) == 3                                   # only the shared vertex
    assert all(s.length <= 50.0 + 1e-6 for s in net.segments)
    assert abs(sum(s.length for s in net.segments) - sum(e.length for e in net.edges)) < 1e-6


def test_exit_nodes_are_dead_ends_on_the_analysis_box_edge():
    a = way(1, [(85.0, 28.05), (85.05, 28.05)])                    # starts on the west edge
    b = way(2, [(85.05, 28.05), (85.06, 28.06)])                   # dead end inside the box: not an exit
    net = build_network([a, b], 50.0, BOX)
    assert len(net.exit_nodes) == 1
    lon, lat = net.node_lonlat[net.exit_nodes[0]]
    assert lon == pytest.approx(85.0) and lat == pytest.approx(28.05)


def test_segments_have_deterministic_ids_and_equal_lengths():
    net = build_network([way(7, [(85.01, 28.01), (85.02, 28.01)])], 50.0, BOX)
    total = haversine_m(np.array([85.01]), np.array([28.01]), np.array([85.02]), np.array([28.01]))[0]
    assert len(net.segments) == int(np.ceil(total / 50.0))
    assert [s.segment_id for s in net.segments[:2]] == ["seg-7-0-0-0", "seg-7-0-0-1"]
    assert np.allclose([s.length for s in net.segments], total / len(net.segments))


def test_parse_roads_handles_multilinestring_and_skips_nodes():
    feats = [
        {"type": "Feature", "geometry": {"type": "MultiLineString", "coordinates": [[[85.0, 28.0], [85.01, 28.0]],
                                                                                     [[85.02, 28.0], [85.03, 28.0]]]},
         "properties": {"@osmId": "way/5", "highway": "track"}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [85.0, 28.0]},
         "properties": {"@osmId": "node/1", "highway": "bus_stop"}},
    ]
    ways = parse_roads(feats)
    assert len(ways) == 1 and len(ways[0].parts) == 2 and ways[0].tags == {"highway": "track"}


def test_dilate_square_window():
    m = np.zeros((5, 5), dtype=bool)
    m[2, 2] = True
    assert dilate(m, 1).sum() == 9 and dilate(m, 0).sum() == 1


def test_road_buffer_radius_in_pixels():
    assert road_buffer_radius_px(10.0, 10.0, 28.0) == 1          # 10 m / 8.83 m ground pixel
    assert road_buffer_radius_px(0.0, 10.0, 28.0) == 0


class LineSampler:
    """Stand-in sampler: classes by position along a straight west-east line (index of the sample)."""

    def __init__(self, classes: np.ndarray, reasons: np.ndarray) -> None:
        self.classes, self.reasons = classes, reasons

    def classify(self, lonlat: np.ndarray):
        lon = lonlat[:, 0]
        i = np.clip(((lon - 85.01) / 0.001 * len(self.classes)).astype(int), 0, len(self.classes) - 1)
        return self.classes[i], self.reasons[i]


def _assess(pattern: list[tuple[int, int, int]], min_block: float = 25.0, min_unobs: float = 25.0):
    """pattern: (class, reason, length in 1 m cells) along a ~98 m road."""
    classes = np.concatenate([np.full(n, c) for c, _r, n in pattern]).astype(np.int8)
    reasons = np.concatenate([np.full(n, r) for _c, r, n in pattern]).astype(np.int64)
    net = build_network([way(1, [(85.01, 28.01), (85.011, 28.01)])], 1000.0, BOX)   # one segment
    return assess_network(net, LineSampler(classes, reasons), 1.0, min_block, min_unobs)[0]


def test_flood_run_long_enough_blocks():
    a = _assess([(SAMPLE_OPEN, 0, 30), (SAMPLE_FLOOD, 1, 30), (SAMPLE_OPEN, 0, 38)])
    assert a.state is RoadState.BLOCKED and a.affected_length_m == pytest.approx(30, abs=2)


def test_short_flood_run_is_recorded_but_does_not_block():
    a = _assess([(SAMPLE_OPEN, 0, 40), (SAMPLE_FLOOD, 1, 15), (SAMPLE_OPEN, 0, 43)])
    assert a.state is RoadState.OPEN and a.affected_length_m > 0


def test_unobservable_run_that_could_hide_a_block_is_unobservable():
    a = _assess([(SAMPLE_OPEN, 0, 40), (SAMPLE_UNOBSERVABLE, int(BaselineState.NOT_ASSESSED_STEEP), 30),
                 (SAMPLE_OPEN, 0, 28)])
    assert a.state is RoadState.UNOBSERVABLE
    assert set(a.unobservable_reasons) == {"NOT_ASSESSED_STEEP"} and a.unobservable_length_m == pytest.approx(30, abs=2)


def test_short_flood_next_to_unobservable_is_not_called_open():
    # 15 m flood + 15 m unobservable: no observed blocking run, but a 30 m run could be blocked
    a = _assess([(SAMPLE_OPEN, 0, 40), (SAMPLE_FLOOD, 1, 15), (SAMPLE_UNOBSERVABLE, OUTSIDE_RASTER, 15),
                 (SAMPLE_OPEN, 0, 28)])
    assert a.state is RoadState.UNOBSERVABLE
    assert "OUTSIDE_SAR_RASTER" in a.unobservable_reasons


def test_short_unobservable_gap_between_open_stays_open():
    a = _assess([(SAMPLE_OPEN, 0, 50), (SAMPLE_UNOBSERVABLE, int(BaselineState.NOT_ASSESSED_SHADOW), 10),
                 (SAMPLE_OPEN, 0, 38)])
    assert a.state is RoadState.OPEN and a.unobservable_length_m > 0


def test_blocked_wins_over_unobservable():
    a = _assess([(SAMPLE_UNOBSERVABLE, int(BaselineState.NOT_ASSESSED_STEEP), 40), (SAMPLE_FLOOD, 1, 30),
                 (SAMPLE_OPEN, 0, 28)])
    assert a.state is RoadState.BLOCKED


def test_state_sampler_classes_and_window():
    state = np.zeros((10, 10), dtype=np.uint8)
    state[5, 5] = int(BaselineState.FLOOD)
    state[0, 0] = int(BaselineState.NOT_ASSESSED_LAYOVER)
    # pixel (r, c) centre in EPSG:3857 = (c + .5) * 10, -(r + .5) * 10 with origin (0, 0)
    sampler = StateSampler(state=state, transform=(10.0, 0.0, 0.0, 0.0, -10.0, 0.0), radius_px=1)
    from bato.inference.diagnostics import lonlat_from_3857
    xs = np.array([45.0, 65.0, 5.0, 95.0, 500.0])
    ys = np.array([-55.0, -55.0, -5.0, -95.0, -5.0])
    lon, lat = lonlat_from_3857(xs, ys)
    cls, reason = sampler.classify(np.stack([lon, lat], axis=1))
    assert cls.tolist() == [SAMPLE_FLOOD, SAMPLE_FLOOD, SAMPLE_UNOBSERVABLE, SAMPLE_OPEN, SAMPLE_UNOBSERVABLE]
    assert reason[2] == int(BaselineState.NOT_ASSESSED_LAYOVER) and reason[4] == OUTSIDE_RASTER


def test_bridge_tag():
    assert is_bridge({"bridge": "yes"}) and is_bridge({"bridge": "viaduct"})
    assert not is_bridge({"bridge": "no"}) and not is_bridge({})


def test_flood_run_is_joined_across_way_splits():
    """A 40 m flood centred on the vertex where OSM splits a road into two ways must still block."""
    def classes(lonlat):
        lon = lonlat[:, 0]
        x = (lon - 85.0105) * 98_300                                  # metres from the split vertex
        cls = np.where(np.abs(x) <= 20, SAMPLE_FLOOD, SAMPLE_OPEN).astype(np.int8)
        return cls, np.zeros(len(lon), dtype=np.int64)

    sampler = type("S", (), {"classify": staticmethod(classes)})()
    one = build_network([way(1, [(85.01, 28.01), (85.0105, 28.01), (85.011, 28.01)])], 1000.0, BOX)
    split = build_network([way(1, [(85.01, 28.01), (85.0105, 28.01)]), way(2, [(85.0105, 28.01), (85.011, 28.01)])],
                          1000.0, BOX)
    a1 = assess_network(one, sampler, 1.0, 25.0, 25.0)
    a2 = assess_network(split, sampler, 1.0, 25.0, 25.0)
    assert all(a.state is RoadState.BLOCKED for a in a1)
    assert len(a2) == 2 and all(a.state is RoadState.BLOCKED for a in a2)
    assert all(a.affected_length_m < 25.0 for a in a2)              # each piece alone is below the threshold


def test_highway_areas_are_not_roads():
    feats = [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[85.0, 28.0], [85.01, 28.0],
                                                                                   [85.01, 28.01], [85.0, 28.0]]]},
              "properties": {"@osmId": "way/9", "highway": "pedestrian", "area": "yes"}}]
    assert parse_roads(feats) == []


# ---------------------------------------------------------------------------
# Road rule v2 (Step 4C FLOOD + disturbance raster; only geometry makes a road unobservable)
# ---------------------------------------------------------------------------

def _v2_world(step4c_code: int, debris_code: int):
    """A ~147 m west-east road at 28.01 N on a 10 m EPSG:3857 grid; cols 15-21 on the road row get the codes."""
    from bato.impact.roads import lonlat_to_3857
    x0, y0 = lonlat_to_3857(np.array([85.009]), np.array([28.011]))
    transform = (10.0, 0.0, float(x0[0]), 0.0, -10.0, float(y0[0]))
    state = np.zeros((30, 40), dtype=np.uint8)
    debris = np.zeros((30, 40), dtype=np.uint8)
    _x, y = lonlat_to_3857(np.array([85.01]), np.array([28.01]))
    row = int((float(y[0]) - transform[5]) // -10.0)
    state[row - 2:row + 3, 15:22] = step4c_code
    debris[row - 2:row + 3, 15:22] = debris_code
    net = build_network([way(1, [(85.01, 28.01), (85.0115, 28.01)])], 1000.0, BOX)
    return net, state, debris, transform


def _one(net, sampler):
    return assess_network(net, sampler, 1.0, 25.0, 25.0)[0]


def test_v2_steep_terrain_is_observed_not_unobservable():
    net, state, debris, tf = _v2_world(int(BaselineState.NOT_ASSESSED_STEEP), 0)
    v1 = _one(net, StateSampler(state=state, transform=tf, radius_px=1))
    v2_sampler = StateSampler(state=state, transform=tf, radius_px=1, debris=debris)
    v2 = _one(net, v2_sampler)
    assert v1.state is RoadState.UNOBSERVABLE and set(v1.unobservable_reasons) == {"NOT_ASSESSED_STEEP"}
    assert v2.state is RoadState.OPEN and v2.unobservable_length_m == 0 and v2.disturbance_samples == 0
    assert v2_sampler.rule == "v2"


def test_v2_disturbance_within_the_buffer_blocks_the_road():
    net, state, debris, tf = _v2_world(int(BaselineState.NOT_ASSESSED_STEEP), 1)
    a = _one(net, StateSampler(state=state, transform=tf, radius_px=1, debris=debris))
    assert a.state is RoadState.BLOCKED
    assert a.disturbance_samples == a.flood_samples > 0 and a.affected_length_m >= 25.0


def test_v2_geometry_gates_still_make_the_road_unobservable():
    for code, name in ((int(BaselineState.NOT_ASSESSED_LAYOVER), "NOT_ASSESSED_LAYOVER"),
                       (int(BaselineState.NOT_ASSESSED_SHADOW), "NOT_ASSESSED_SHADOW")):
        net, state, debris, tf = _v2_world(code, code)
        a = _one(net, StateSampler(state=state, transform=tf, radius_px=1, debris=debris))
        assert a.state is RoadState.UNOBSERVABLE and set(a.unobservable_reasons) == {name}


def test_v2_flood_still_blocks_and_outside_raster_is_unobservable():
    net, state, debris, tf = _v2_world(int(BaselineState.FLOOD), 0)
    a = _one(net, StateSampler(state=state, transform=tf, radius_px=1, debris=debris))
    assert a.state is RoadState.BLOCKED and a.disturbance_samples == 0
    sampler = StateSampler(state=state, transform=tf, radius_px=1, debris=debris)
    from bato.inference.diagnostics import lonlat_from_3857
    lon, lat = lonlat_from_3857(np.array([tf[2] - 500.0]), np.array([tf[5] - 50.0]))
    cls, reason = sampler.classify(np.stack([lon, lat], axis=1))
    assert cls.tolist() == [SAMPLE_UNOBSERVABLE] and reason.tolist() == [OUTSIDE_RASTER]


def test_v2_debris_grid_must_match():
    state = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(Exception, match="does not match"):
        StateSampler(state=state, transform=(10.0, 0.0, 0.0, 0.0, -10.0, 0.0), radius_px=1,
                     debris=np.zeros((9, 10), dtype=np.uint8))
