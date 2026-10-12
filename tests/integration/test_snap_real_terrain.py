"""OPT-IN: the Step 3.4 terrain-correction graph against the REAL installed SNAP (no data read or written).

Skipped unless BATO_RUN_SNAP_TESTS=1. Confirms with `gpt <Operator> -h` that SNAP knows every
operator/parameter of graphs/s1_terrain.xml and accepts every value that has a SNAP value set
(resampling methods, incidence-angle choices, auxFile). Takes ~30-60 s.

    $env:BATO_RUN_SNAP_TESTS="1"; python -m pytest tests/integration/test_snap_real_terrain.py -v -s
"""

from __future__ import annotations

import os

import pytest

from bato.preprocessing import snap

pytestmark = pytest.mark.skipif(os.environ.get("BATO_RUN_SNAP_TESTS") != "1",
                                reason="real SNAP tests are opt-in: set BATO_RUN_SNAP_TESTS=1")


def test_installed_snap_accepts_the_terrain_graph(cfg):
    gpt = snap.resolve_gpt(cfg.settings.snap)
    diag = snap.snap_diagnostics(gpt.path, cfg.settings.snap.diag_timeout_s)
    t = cfg.settings.preprocessing.s1_terrain
    rendered = snap.render_graph(snap.graph_path(t.graph).read_text(encoding="utf-8"), {
        "input_file": "C:/placeholder/in.dim", "source_bands": "Sigma0_VV,Sigma0_VH",
        "dem_file": "C:/placeholder/dem.tif", "dem_nodata": t.dem_nodata, "dem_apply_egm": t.dem_apply_egm,
        "output_file": "C:/placeholder/out.dim"})
    checked = snap.check_operator_parameters(gpt.path, rendered, cfg.settings.snap.diag_timeout_s)
    report = snap.check_parameter_values(gpt.path, rendered, ["Terrain-Correction"], cfg.settings.snap.diag_timeout_s)
    print(f"\ngpt {gpt.path} ({gpt.source}); SNAP {diag.version}")
    print(f"Terrain-Correction value sets confirmed: {sorted(report['Terrain-Correction']['checked'])}")
    print(f"Terrain-Correction parameters without a listed value set: {report['Terrain-Correction']['not_listed']}")
    assert set(checked) == {"Read", "Terrain-Correction", "Write"}
    expected = {"demResamplingMethod", "imgResamplingMethod"}
    unverified = expected & set(report["Terrain-Correction"]["not_listed"])
    assert not unverified, f"SNAP help lists no value set for {unverified}: check `gpt Terrain-Correction -h` by hand"
