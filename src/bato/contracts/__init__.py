"""Typed data contracts passed between BATO pipeline stages.

A  aoi.py          AOIRequest -> AOI
B  acquisition.py  Acquisition
C  pairing.py      PrePostPair (same relative orbit only)
D  detection.py    DetectionRaster, ImpactClass
E  impact.py       RoadSegmentImpact
F  settlement.py   SettlementStatus, classify_settlement
G  evidence.py     EvidenceRecord
   osm.py          OSMSnapshot, select_snapshot_date (pre-event rule)
   safe.py         SafeProduct, SafeBand (Step 3.1 SAFE inspection)
   preprocessing.py PreprocessingProvenance, RadiometricValidation (Step 3.2)

See docs/data_contracts.md for field-level documentation.
"""
