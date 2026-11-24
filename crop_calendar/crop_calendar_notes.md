# Crop Calendar Sourcing and Agronomic Notes (Issue #43)

## 1. Overview
The QWeatherRiskF project models agricultural yield anomalies and financial portfolio risks across the 13 districts of Andhra Pradesh (2015–2019). Weather indices generated in Issue 3 are temporally aggregated into six seasonal windows (`annual_cal`, `annual_crop`, `kharif`, `ne`, `rabi_next`, `rabi_prev`).

To prevent severe temporal misalignment and data leakage (e.g., associating July monsoon rainfall with Rabi chickpea yields), this crop calendar establishes the verified physical and agronomic in-ground windows for all 17 eligible crops cultivated in Andhra Pradesh.

---

## 2. Public Agronomic Sourcing Policy
Per project governance standards:
- **No silent imputation:** Missing is not zero.
- **No inferred windows without source:** If a crop does not have commercial cultivation in Andhra Pradesh or lacks an official package of practices, its operational modeling status is explicitly marked as `unavailable`.
- **Authoritative sources used:**
  1. **ANGRAU (Acharya N.G. Ranga Agricultural University):**
     - *Vyavasaya Panchangam* (Agricultural Almanac)
     - *Package of Practices for Kharif and Rabi Crops*
  2. **ICAR - CRIDA (Central Research Institute for Dryland Agriculture):**
     - *District Agriculture Contingency Plans for Andhra Pradesh* (published for all 13 districts: Anantapuramu, Chittoor, East Godavari, Guntur, Krishna, Kurnool, Prakasam, SPSR Nellore, Srikakulam, Visakhapatnam, Vizianagaram, West Godavari, YSR Kadapa).
  3. **Directorate of Economics & Statistics (DES), Government of Andhra Pradesh:**
     - *Season and Crop Reports of Andhra Pradesh* (2015–2019).

---

## 3. The 17 Eligible Crops Agronomic Profiles

### 1. Rice (`rice`)
- **Kharif (Sarva):**
  - Sowing/Nursery: June – July; Transplanting: July – August.
  - Harvesting: November – December.
  - Duration: 130–150 days.
  - Weather Window: `kharif`.
  - **Vulnerability:** Maturation and harvest in October–November overlap with the North-East monsoon, making coastal deltas (East Godavari, West Godavari, Krishna) vulnerable to cyclone landfall, storm surges, and crop lodging.
- **Rabi (Dalwa):**
  - Sowing/Nursery: November – December; Transplanting: December – January.
  - Harvesting: March – April.
  - Duration: 120–135 days.
  - Weather Window: `rabi_next`.
  - **Vulnerability:** Irrigated crop dependent on reservoir/canal releases; sensitive to terminal heat stress in March–April.

### 2. Groundnut (`groundnut`)
- **Kharif (Rainfed):**
  - Sowing: June – July (with onset of South-West monsoon).
  - Harvesting: October – November.
  - Duration: 105–120 days.
  - Weather Window: `kharif`.
  - **Vulnerability:** Dominant rainfed crop in Rayalaseema (Anantapuramu, Chittoor, Kurnool). Highly sensitive to dry spells during pegging and pod development (July–August).
- **Rabi (Irrigated):**
  - Sowing: October – November.
  - Harvesting: February – March.
  - Duration: 110–125 days.
  - Weather Window: `rabi_next`.

### 3. Maize (`maize`)
- **Kharif:**
  - Sowing: June – July.
  - Harvesting: September – October.
  - Duration: 95–110 days.
  - Weather Window: `kharif`.
- **Rabi:**
  - Sowing: October – November.
  - Harvesting: February – March.
  - Duration: 115–130 days.
  - Weather Window: `rabi_next`.
  - High-yielding zero-till rice-fallow maize in Guntur, Krishna, and West Godavari.

### 4. Sugarcane (`sugarcane`)
- **Annual Crop:**
  - Planting: January – March (Spring) or October – November (Autumn).
  - Harvesting: December – March (10–12 months duration).
  - Weather Window: `annual_crop` (also spans `annual_cal`).
  - Vulnerable to prolonged waterlogging and lodging during October–November cyclones.

### 5. Pigeonpea / Red Gram (`pigeonpea`)
- **Kharif (Long-Duration Pulse):**
  - Sowing: June – July.
  - Harvesting: December – January.
  - Duration: 160–180 days.
  - Weather Window: `kharif` (vegetative stage in SW monsoon, pod formation/harvesting in NE monsoon).
  - Intercropped with groundnut in Rayalaseema as a drought risk mitigation hedge.

### 6. Sesamum / Til (`sesamum`)
- **Kharif:** Sown June–July, harvested September–October. Window: `kharif`.
- **Rabi / Rice Fallow:** Sown January–February, harvested April–May. Window: `rabi_next`.

### 7. Chickpea / Bengal Gram (`chickpea`)
- **Rabi (Strictly Winter Pulse):**
  - Sowing: October – November.
  - Harvesting: January – February.
  - Duration: 90–105 days.
  - Weather Window: `rabi_next`.
  - Grown on deep black cotton soils (Kurnool, Anantapuramu, Prakasam, YSR Kadapa) relying on residual soil moisture from the retreating monsoon.

### 8. Sunflower (`sunflower`)
- **Kharif:** Sown July–August, harvested October–November. Window: `kharif`.
- **Rabi:** Sown October–November, harvested January–February. Window: `rabi_next`.

### 9. Finger Millet / Ragi (`finger_millet`)
- **Kharif:** Sown June–July, harvested October–November. Window: `kharif`. Staple crop in North Coastal agency tracts (Visakhapatnam, Vizianagaram, Srikakulam).
- **Rabi:** Sown November–December, harvested February–March. Window: `rabi_next`.

### 10. Pearl Millet / Bajra (`pearl_millet`)
- **Kharif:** Sown June–July, harvested September–October. Duration: 80–90 days. Window: `kharif`. Drought hardy staple in Prakasam and Rayalaseema.

### 11. Cotton (`cotton`)
- **Kharif (Semi-Annual Cash Crop):**
  - Sowing: June – July.
  - Harvesting (multiple pickings): November – February.
  - Duration: 150–180 days.
  - Weather Window: `kharif` (extends into `ne` and `rabi_next`).
  - Major commercial hub in Guntur and Kurnool. High sensitivity to excessive moisture and boll rot from October cyclonic depressions.

### 12. Castor (`castor`)
- **Kharif:** Sown July–August, harvested December–February. Duration: 150–180 days. Window: `kharif`.

### 13. Kharif Sorghum (`kharif_sorghum`)
- **Kharif:** Sown June–July, harvested October–November. Duration: 100–115 days. Window: `kharif`.

### 14. Rabi Sorghum (`rabi_sorghum`)
- **Rabi (Maghi Jowar):** Sown September–October, harvested January–February. Duration: 105–120 days. Window: `rabi_next`. Grown on deep vertisols with stored moisture in Kurnool and Anantapuramu.

### 15. Soyabean (`soyabean`)
- **Kharif:** Sown June–July, harvested September–October. Window: `kharif`.

### 16. Rapeseed and Mustard (`rapeseed_and_mustard`)
- **Rabi:** Sown October–November, harvested January–February. Window: `rabi_next`. Minor localized winter cultivation.

### 17. Safflower (`safflower`)
- **Rabi:** Sown October–November, harvested February–March. Window: `rabi_next`. Vertisol residual moisture crop in Rayalaseema.

### Non-Commercial / Excluded AP Crops:
- **Wheat (`wheat`):** Temperate crop. Cultivation in tropical AP is commercially non-viable (<0.01% of gross cropped area), restricted to small experimental/agency patches in Visakhapatnam hills. Status: `unavailable` for operational prediction modeling.
- **Linseed (`linseed`):** Negligible acreage in AP, lacking an official commercial production package in ANGRAU guidelines. Status: `unavailable` for operational prediction modeling.

---

## 4. Alignment to Issue #3 Weather Windows

| Calendar Season | Sowing Window | Harvest Window | Aligned Issue 3 Weather Window | Overlap / Boundary Notes |
|---|---|---|---|---|
| `kharif` | June – July | September – November | `kharif` | SW Monsoon active window (Jun 1 – Sep 30). Late Kharif overlaps with NE monsoon onset in Oct. |
| `rabi` | October – December | January – April | `rabi_next` | Winter / post-monsoon window (Oct 1 – Mar 31). |
| `annual` | January – March | December – March | `annual_crop` | Full agricultural crop year (Jun 1 – May 31) / `annual_cal` (Jan 1 – Dec 31). |
