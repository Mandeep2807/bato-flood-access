"""SAR preprocessing matched to the Kuro Siwo training data.

Implemented:

* Step 3.1 - :mod:`bato.preprocessing.safe_reader`: read-only SAFE zip inspection.
* Step 3.2 - :mod:`bato.preprocessing.radiometric` (+ :mod:`.snap`, :mod:`.dimap`,
  :mod:`.sigma0_validation`): one Sentinel-1 GRD pass -> linear sigma0 VV/VH in
  radar geometry with ESA SNAP gpt (orbit, AOI subset, thermal and border noise
  removal, calibration), validated with rasterio, with provenance.
* Step 3.3 - :mod:`bato.preprocessing.speckle` (+ :mod:`.speckle_validation`): Lee Sigma
  speckle filter (Kuro Siwo parameters) on the published Step 3.2 output, still radar geometry.
* Step 3.4 - :mod:`bato.preprocessing.terrain` (+ :mod:`.dem_mosaic`, :mod:`.terrain_validation`):
  Range-Doppler terrain correction of the published Step 3.3 output with BATO's Step 2C
  Copernicus DEM tiles, EPSG:3857 10 m (Kuro Siwo node), linear sigma0.
* Step 3.5 - :mod:`bato.preprocessing.stack`: PRE-1/PRE-2/POST temporal stack of the Step 3.4
  outputs on one common grid (whole-pixel copy, no resampling), linear sigma0, validity mask.

* Step 4A - :mod:`bato.preprocessing.model_input`: the Kuro Siwo-compatible model-input contract
  (channel order POST / sec1 = PRE-2 / sec2 = PRE-1, clamp + z-score, no DEM, 224 x 224 tiles,
  validity/ignore masks), a manifest of the stack's tiles and a tile reader. No training.

NOT IMPLEMENTED YET: slice assembly and Sentinel-2 cloud masking.
"""
