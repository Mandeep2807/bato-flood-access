"""OPT-IN: checks against the REAL installed SNAP (no data is read or written).

Skipped unless BATO_RUN_SNAP_TESTS=1. Uses the gpt found by the normal resolution
(snap.gpt_path -> BATO_SNAP_GPT -> default install -> PATH). Takes ~1-2 minutes
(each gpt call starts a JVM).

    set BATO_RUN_SNAP_TESTS=1          (Windows cmd)   |  $env:BATO_RUN_SNAP_TESTS="1"  (PowerShell)
    python -m pytest tests/integration/test_snap_real.py -v
"""

from __future__ import annotations

import os

import pytest

from bato.preprocessing import snap

pytestmark = pytest.mark.skipif(os.environ.get("BATO_RUN_SNAP_TESTS") != "1",
                                reason="real SNAP tests are opt-in: set BATO_RUN_SNAP_TESTS=1")


def test_installed_snap_accepts_the_step_3_2_graph(cfg):
    gpt = snap.resolve_gpt(cfg.settings.snap)
    diag = snap.snap_diagnostics(gpt.path, cfg.settings.snap.diag_timeout_s)
    print(f"\ngpt {gpt.path} ({gpt.source}); SNAP version {diag.version}")
    r = cfg.settings.preprocessing.s1_radiometric
    rendered = snap.render_graph(snap.graph_path(r.graph).read_text(encoding="utf-8"), {
        "input_file": "C:/placeholder/input.SAFE.zip", "orbit_type": r.orbit_type,
        "orbit_poly_degree": r.orbit_poly_degree, "orbit_continue_on_fail": r.orbit_continue_on_fail,
        "geo_region": "POLYGON ((85 27, 86 27, 86 28, 85 28, 85 27))", "polarisations": "VV,VH",
        "remove_thermal_noise": r.remove_thermal_noise, "reintroduce_thermal_noise": r.reintroduce_thermal_noise,
        "border_limit": r.border_limit, "trim_threshold": r.trim_threshold, "output_in_db": r.output_in_db,
        "output_sigma0": r.output_sigma0, "output_file": "C:/placeholder/out.dim"})
    checked = snap.check_operator_parameters(gpt.path, rendered, cfg.settings.snap.diag_timeout_s)
    assert len(checked) == 7
