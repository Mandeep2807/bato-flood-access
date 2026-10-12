"""Flood-water detection.

Implemented:

* Step 4C - :mod:`bato.detection.baseline` (+ :mod:`.components`): deterministic SAR change baseline on the
  Step 3.5 stack - new open-water darkening vs both pre-event images in VV and VH, terrain/geometry gating
  to NOT_ASSESSED (never "no flood"), 8-connected cluster filter, provenance and diagnostics. Not validated.

NOT IMPLEMENTED YET: calibrated model-based detection (the FloodViT layer of Step 4B is preliminary and
unvalidated), fusion into the ImpactClass raster, debris/disturbance change statistics.
"""
