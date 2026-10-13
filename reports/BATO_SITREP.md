# BATO SITREP 01 — Trishuli Flood, Nepal

**Issued** 2026-10-13 18:43 UTC · **Observed** 2026-08-28 12:21 UTC (POST pass) · **NOT VALIDATED**

**Incident:** flood, Bhote Koshi–Trishuli corridor (Rasuwa/Nuwakot), Nepal · **Event:** 2026-08-26 · **Area:** 85.17–85.42° E, 27.97–28.28° N (roads within a 15 km buffer) · **Data:** Sentinel-1D ascending, orbit 85 — PRE-1 2026-06-29, PRE-2 2026-08-16, **POST 2026-08-28 12:21:41 UTC**; OSM pre-event snapshot 2026-08-25. *State at the POST pass, issued 46 days later: not an assessment of current conditions; use only to plan field verification.*

### 1. Operational summary

Radar shows flood evidence on **30 road segments in 9 stretches (1.227 km)**. **0 settlements are confirmed cut off**; 1 is connected on observed roads; **163 settlements have access unverified** (routes cross road the radar could not assess). **91.2 % of scene pixels were not assessed** (steep slopes, layover, shadow), so most roads are unobservable.

### 2. Roads and settlement access

| Road state | Length | Settlement access (265) | Count |
|---|---|---|---|
| Blocked (30 segments, 9 stretches) | **1.227 km** | Connected | 1 |
| Observed open | 463.714 km | Access unverified | **163** |
| Unobservable | 2,942.161 km | Confirmed cutoff | **0** |
| | | No pre-event road access | 101 |

**Observed open** = no flood signal detected, passability not guaranteed. **Unobservable** = unknown, not open. **No pre-event road access** = no mapped road within 500 m (OSM gap, not flood).

### 3. Priorities for field verification

1. **Verify the 9 blocked stretches** (dashboard evidence drawer: location, radar chips, dates).
2. **Survey the 163 access-unverified settlements**, starting with segments shared by several (settlement drawer).
3. **Reconnoitre before use:** observed-open and unobservable roads are unconfirmed.
4. **101 settlements with no mapped road:** assess access by other means; check OSM completeness.

### 4. Key uncertainty and limitations

- **Step 4C:** provisional thresholds, **not validated**; open-water darkening only (misses debris, landslides, structural damage).
- **Single pass:** no observation after 2026-08-28 12:21 UTC; conditions may have changed since.
- **EMSR927:** **evaluation only**; no flood class, so **flood precision, recall, F1 and IoU are not calculable**. BATO missed most EMSR927-damaged road: the blocked list is incomplete.
- **Not estimated:** population, casualties and forecasts.

### 5. Sources and attribution

Sentinel-1 GRD (Copernicus Data Space Ecosystem), Copernicus DEM GLO-30, OpenStreetMap (ohsome-api-v2).<br>
Contains modified Copernicus Sentinel data 2026.<br>
Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.<br>
© OpenStreetMap contributors.<br>
Evaluation reference only: European Union, Copernicus Emergency Management Service data (EMSR927).<br>
<sub>Prepared by 404 Competition Not Found · bato-sitrep-v1 · run aoi-e893707c2beb/ts-80841ae62417 (rule v1) · deterministic, not AI-generated</sub>
