"""OPT-IN: the Step 3.3 Lee Sigma graph against the REAL installed SNAP (no data is read or written).

Skipped unless BATO_RUN_SNAP_TESTS=1. Confirms with `gpt <Operator> -h` that SNAP knows every
operator/parameter of graphs/s1_speckle.xml AND accepts the Lee Sigma values (filter 'Lee Sigma',
windowSize, targetWindowSizeStr, sigmaStr, numLooksStr). Takes ~30-60 s.

    $env:BATO_RUN_SNAP_TESTS="1"; python -m pytest tests/integration/test_snap_real_speckle.py -v -s
"""

from __future__ import annotations

import os

import pytest

from bato.preprocessing import snap

pytestmark = pytest.mark.skipif(os.environ.get("BATO_RUN_SNAP_TESTS") != "1",
                                reason="real SNAP tests are opt-in: set BATO_RUN_SNAP_TESTS=1")


def test_installed_snap_accepts_the_lee_sigma_graph(cfg):
    gpt = snap.resolve_gpt(cfg.settings.snap)
    diag = snap.snap_diagnostics(gpt.path, cfg.settings.snap.diag_timeout_s)
    s = cfg.settings.preprocessing.s1_speckle
    rendered = snap.render_graph(snap.graph_path(s.graph).read_text(encoding="utf-8"), {
        "input_file": "C:/placeholder/in.dim", "source_bands": "Sigma0_VV,Sigma0_VH", "filter": s.filter,
        "num_looks": s.num_looks, "window_size": s.window_size, "target_window_size": s.target_window_size,
        "sigma": s.sigma, "output_file": "C:/placeholder/out.dim"})
    checked = snap.check_operator_parameters(gpt.path, rendered, cfg.settings.snap.diag_timeout_s)
    report = snap.check_parameter_values(gpt.path, rendered, ["Speckle-Filter"], cfg.settings.snap.diag_timeout_s)
    print(f"\ngpt {gpt.path} ({gpt.source}); SNAP {diag.version}")
    print(f"Speckle-Filter value sets confirmed: {report['Speckle-Filter']['checked']}")
    print(f"Speckle-Filter parameters without a listed value set: {report['Speckle-Filter']['not_listed']}")
    assert set(checked) == {"Read", "Speckle-Filter", "Write"}
    lee_sigma = {"filter", "numLooksStr", "windowSize", "targetWindowSizeStr", "sigmaStr"}
    unverified = lee_sigma & set(report["Speckle-Filter"]["not_listed"])
    assert not unverified, f"SNAP help lists no value set for {unverified}: check `gpt Speckle-Filter -h` by hand"
