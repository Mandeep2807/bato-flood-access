"""Model inference on BATO stacks.

* Step 4B - :mod:`bato.inference.floodvit` (torch-free core: tiling, Step 4A input path, overlap
  averaging, invalid -> ignore, statistics, provenance) and :mod:`bato.inference.floodvit_model`
  (the Kuro Siwo FloodViT network; needs torch + einops and a Kuro Siwo checkout).
"""
