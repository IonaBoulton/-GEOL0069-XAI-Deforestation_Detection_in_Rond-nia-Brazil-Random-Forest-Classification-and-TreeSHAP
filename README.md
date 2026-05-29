# GEOL0069-XAI-Deforestation_Detection_in_Rond-nia-Brazil-Random-Forest-Classification-and-TreeSHAP
Explainable Forest Loss Detection in Rondônia, Brazil: A Dual-Label Experiment Comparing Hansen GFC and Spectral Threshold Training Strategies for Sentinel-2 Random Forest Classification with TreeSHAP Analysis (2019–2022)

<p align="center">
  <img src="https://github.com/user-attachments/assets/1b7d8d83-687d-47c6-a088-8d71b777d9ef" alt="Amazon-deforestation-679x419" width="679" height="419">
  <br>
  <i>Amazon Rainforest Deforestation and logging </i>
</p>

---

<p align="center">

<details>
<summary><b>🌿 2019 — Before: Sentinel-2 true-colour composite (click to reveal)</b></summary>
<br>
<img src="https://raw.githubusercontent.com/IonaBoulton/-GEOL0069-XAI-Deforestation_Detection_in_Rond-nia-Brazil-Random-Forest-Classification-and-TreeSHAP/main/Figures/Fig1_RawDataInspection.png" width="85%"/>
</details>

<details>
<summary><b>🟡 2022 — ΔNDVI change signal (click to reveal)</b></summary>
<br>
<img src="https://raw.githubusercontent.com/IonaBoulton/-GEOL0069-XAI-Deforestation_Detection_in_Rond-nia-Brazil-Random-Forest-Classification-and-TreeSHAP/main/Figures/Fig4_DeltaNDVI_CoreSignal.png" width="85%"/>
</details>

<details>
<summary><b>🔴 Random Forest prediction map — forest loss 2019–2022 (click to reveal)</b></summary>
<br>
<img src="https://raw.githubusercontent.com/IonaBoulton/-GEOL0069-XAI-Deforestation_Detection_in_Rond-nia-Brazil-Random-Forest-Classification-and-TreeSHAP/main/Figures/Fig10_PredictionMap.png" width="85%"/>
<br><br>
<i>Green = stable forest · Red = deforested 2019–2022 · Grey = no data · 1,391 km² lost · ≈20.87 Mt CO₂ at risk</i>
</details>

</p>

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Background & Context](#2-background--context)
3. [Motivation](#3-motivation)
4. [Notebook Overview & Layout](#4-notebook-overview--layout)
5. [Data Sourcing & Preprocessing](#5-data-sourcing--preprocessing)
6. [Methodological Framework](#6-methodological-framework)
7. [Experimental Design: Experiment A vs B](#7-experimental-design-experiment-a-vs-b)
8. [Results](#8-results)
9. [Explainability: TreeSHAP Analysis](#9-explainability-treeshap-analysis)
10. [Environmental Assessment](#10-environmental-assessment)
11. [Discussion, Limitations & Future Work](#11-discussion-limitations--future-work)
12. [Video Walkthrough](#12-video-walkthrough)
13. [Acknowledgements](#13-acknowledgements)
14. [References](#14-references)
15. [Contact](#15-contact)

---

## 1. Introduction

This repository contains the final independent project for the **GEOL0069 — AI for Earth 
Observations** module at University College London.

The project maps and explains rapid tropical deforestation in **Rondônia, Brazil** — one of 
the most actively deforested regions on Earth — over a four-year window from **2019 to 2022**, 
using a pair of dry-season **Sentinel-2 Level-2A** acquisitions.

At its core, the pipeline trains a **Random Forest classifier** on engineered spectral change 
features (NDVI, NBR, NDWI and their temporal deltas) and explains every prediction using 
**TreeSHAP** — game-theoretic feature attribution that reveals *why* the model flags a pixel 
as deforested, not just *that* it does.

A central methodological contribution is a **dual-label experiment**:

- **Experiment A** trains on external labels from the **Hansen Global Forest Change** dataset 
  (UMD, 30 m) — an independent, peer-reviewed ground truth that introduces a known 
  resolution mismatch with Sentinel-2's 10 m grid.
- **Experiment B** trains on **spectral threshold labels** derived directly from ΔNDVI — 
  achieving native 10 m spatial alignment at the cost of label independence.

Comparing the two experiments quantifies the real-world accuracy cost of cross-sensor label 
misalignment and produces an honest methodological discussion rarely seen in student projects. 
The pipeline also tracks its own carbon footprint via **CodeCarbon**, comparing research 
emissions against the carbon stored in the forests being monitored.

---

## 2. Background & Context

<p align="center">
  <img src="Figures/map-Rondonia-Brazil.jpg.webp" 
       alt="Map of South America with Rondônia highlighted" 
       width="54%"/>
  &nbsp;&nbsp;
  <img src="Figures/unnamed.gif" 
       alt="Detailed map of Rondônia state showing cities, rivers and borders" 
       width="38%"/>
</p>

<p align="center">
  <b>Figure 1.</b> Location of Rondônia state within Brazil and South America (left), 
  with state-level detail showing major cities, rivers and borders (right). 
  Rondônia is situated in the south-western Brazilian Amazon, bordering Bolivia 
  to the south-west. Its capital, Porto Velho, lies along the Madeira River.<br>
  <sub>
    Sources: 
    <a href="https://www.britannica.com/place/Rondonia">Encyclopædia Britannica (2024)</a> · 
    <a href="http://www.v-brazil.com/tourism/rondonia/map-rondonia.html">V-Brazil Tourism (2024)</a>
  </sub>
</p>

---

### 2.1 · Study Area: Rondônia, Brazil

Rondônia is a state in the **south-western Brazilian Amazon**, bordering Bolivia to the
south-west and covering approximately 237,576 km². Its capital is **Porto Velho**, situated
along the Madeira River — a major tributary of the Amazon. Though largely tropical rainforest,
Rondônia today represents one of the most heavily altered landscapes in the entire Amazon basin.

### 2.2 · The Arc of Deforestation

Rondônia sits at the heart of what scientists call the **"Arc of Deforestation"** — a
crescent-shaped belt of active forest clearance that follows the southern and eastern margins
of the Amazon basin from the state of Mato Grosso in the east through Rondônia and into
Pará in the north (NASA Earth Observatory, 2006). This arc marks the active agricultural
frontier where forest is converted to cattle pasture and cropland at a faster rate than
anywhere else on Earth.

---

<p align="center">
  <img src="Figures/maaproject.org-maap-164-amazon-tipping-point-where-are-we-Map2-Total-Deforestation-AmzBiog-200dpi-Eng.jpg" 
       alt="Map of total deforestation across the Amazon biome showing concentration in Rondônia" 
       width="75%"/>
</p>

<p align="center">
  <b>Figure 2.</b> Cumulative deforestation across the Amazon biome, 
  highlighting the intense concentration of forest loss in Rondônia and the 
  broader southern arc of deforestation. Rondônia consistently ranks among 
  the most heavily deforested Amazonian states.<br>
  <sub>
    Source: 
    <a href="https://www.maaproject.org/2022/amazon-tipping-point/">
    MAAP (Monitoring of the Andean Amazon Project), 2022 — MAAP #164</a>
  </sub>
</p>

---

### 2.3 · History of Deforestation in Rondônia

Rondônia was almost entirely forested as recently as the 1960s. The transformation began
with the construction and paving of the **BR-364 highway**, which connected the state to
Brazil's Atlantic coast and opened the interior to large-scale migration (Fearnside & Salati,
1985). Government-sponsored colonisation programmes in the 1970s and 1980s encouraged
settlers from Brazil's south and south-east to relocate to the region, offering cheap land
parcels along a grid of secondary roads branching perpendicularly from the BR-364 at
approximately 4 km intervals. From satellite imagery, this pattern of clearing — forest
removed in strips extending outward from a central road spine — is immediately recognisable
as the iconic **"fishbone" deforestation pattern** (Roberts et al., 2002; NASA SVS, 2013).

Colonisation and forest clearance in Rondônia began systematically in the early 1970s,
and by 1986 the total cleared area had grown from just 230 km² in 1980 to over 3,390 km²,
with road networks expanding from 110 km to more than 4,660 km over the same period. 
Between 1986 and 2020, natural vegetation cover in Rondônia declined from 90.9% to 62.7%,
with fragmentation increasing dramatically to produce tens of thousands of isolated forest patches. 

### 2.4 · Remote Sensing of Deforestation in Rondônia

Rondônia has been one of the most intensively studied regions in the remote sensing literature,
precisely because the scale and speed of its forest loss make it an ideal test case for
satellite-based monitoring techniques.

Early studies used **Landsat Thematic Mapper** data to document clearance rates through the
1980s and 1990s (Stone et al., 1991; Skole & Tucker, 1993). Roberts et al. (2002) applied
multitemporal spectral mixture analysis across 80,000 km² of central Rondônia, classifying
land cover into primary forest, pasture, and secondary growth and demonstrating how road
infrastructure drives spatial patterns of clearance. Huang et al. (2009) later extended this
work using time-series Landsat data and the Normalised Degradation Fraction Index (NDFI) to
distinguish between outright deforestation and subtler forest degradation — showing that as
gross deforestation rates declined after 2004, degradation rates actually increased.

The **Hansen Global Forest Change** dataset (Hansen et al., 2013) — one of the two label
sources used in this project — emerged from this tradition, providing annual 30 m
Landsat-derived loss-year maps covering the entire humid tropics from 2000 onwards. It
remains the most widely used global deforestation reference product and has been applied
extensively in Rondônia to quantify forest loss rates and assess the effectiveness of
conservation units (Pedlowski et al., 2005).

More recent work has shifted toward **higher-resolution sensors**, with Sentinel-2's 10 m
multispectral bands enabling detection of finer-scale clearance events, secondary road
incursion, and edge degradation that 30 m Landsat data routinely misses. This resolution
gap — and its consequences for label quality in supervised classification — is a central
motivation for the dual-label experiment conducted in this project.

## 3. Motivation

Tropical deforestation is one of the most consequential environmental crises of the 21st 
century. The Amazon stores an estimated 150–200 billion tonnes of carbon and absorbs 
approximately 29% of annual anthropogenic CO₂ emissions (Gatti et al., 2021). 
Deforestation converts this sink into a source, disrupts the regional water cycle, 
devastates biodiversity, and risks triggering an irreversible ecological tipping point — 
estimated to occur when forest loss exceeds 20–25% of the biome (Lovejoy & Nobre, 2018). 
Parts of the Amazon may already have crossed this threshold (RAISG, 2022).

Monitoring deforestation at the speed and scale it occurs is beyond the capacity of 
traditional field surveys. Satellite remote sensing combined with machine learning offers 
the only practical solution, with Random Forest classifiers consistently achieving strong 
performance in forest change detection tasks (Maxwell et al., 2018). However, accuracy 
alone is insufficient for operational environmental monitoring — decision-makers need to 
understand *why* a model flags a pixel as deforested, not just *that* it does. This 
motivates the integration of **TreeSHAP** for pixel-level explainability, and **CodeCarbon** 
to ensure the carbon cost of the research itself is transparent and accountable.

## 4. Notebook Overview & Layout

The notebook is structured as a single, linear pipeline that can be run top-to-bottom 
in Google Colab. It is divided into 10 sections, each clearly headed and self-contained:

| Section | Content |
|---|---|
| 0 | Environment setup, dependency installation & CodeCarbon tracking |
| 1 | Data loading & visual inspection of raw Sentinel-2 scenes |
| 2 | Preprocessing — cloud masking & band stacking |
| 3 | Feature engineering — spectral indices & temporal delta features |
| 4A / 4B | Label preparation — **Experiment A** (Hansen GFC) & **Experiment B** (Spectral threshold) |
| 5A / 5B | Random Forest training & evaluation for each experiment |
| 6 | Side-by-side experiment comparison |
| 7 | Full-scene prediction map (Experiment B model) |
| 8 | TreeSHAP explainability analysis |
| 9 | Area statistics & environmental assessment |
| 10 | Critical discussion & limitations |

### Why two experiments?

A central methodological challenge in supervised deforestation mapping is the choice of 
training labels. Labels that are spatially or temporally misaligned with the imagery will 
degrade model performance regardless of classifier quality. Rather than simply adopting 
one labelling strategy, this notebook explicitly tests two:

- **Experiment A** uses external labels from the **Hansen Global Forest Change** dataset — 
  an independent, peer-reviewed ground truth at 30 m resolution. This is the principled 
  approach, but introduces a known spatial mismatch with Sentinel-2's native 10 m grid.
- **Experiment B** derives labels directly from **ΔNDVI thresholding** at native 10 m 
  resolution — achieving perfect spatial alignment at the cost of label independence.

Comparing the two experiments quantifies the real-world accuracy cost of cross-sensor 
label misalignment and produces an honest, critical methodological discussion. The full 
scene prediction map and TreeSHAP analysis in Sections 7–8 use the Experiment B model, 
with this choice explicitly justified.

> **Note:** All figures are saved automatically to the `Figures/` folder in your 
> Google Drive project directory on each run.

## 5. Data Sourcing & Preprocessing

### 5.1 · Environment Setup (Section 0)

The notebook runs entirely in **Google Colab** with all dependencies installed at runtime.
The following packages are installed and imported:

| Package | Purpose |
|---|---|
| `rasterio` | Reading, writing and resampling GeoTIFF rasters |
| `numpy` / `pandas` | Array operations and tabular data handling |
| `scikit-learn` | Random Forest classifier, train/test split, evaluation metrics |
| `shap` | TreeSHAP explainability — Shapley value computation |
| `codecarbon` | Carbon emissions tracking across the full notebook runtime |
| `matplotlib` / `seaborn` | All figures and visualisations |
| `geopandas` | Geospatial vector operations |
| `joblib` | Model serialisation |
| `psutil` | RAM monitoring during tiled full-scene prediction |

**CodeCarbon** is initialised silently at the start of Section 0 and runs in the background
throughout the entire pipeline, logging energy consumption and CO₂ equivalent emissions
to `emissions.csv` in the project Google Drive directory. Results are reported in Section 9.

---

### 5.2 · Sentinel-2 Data (Sections 1 & 2)

**What is Sentinel-2?**

Sentinel-2 is a twin-satellite constellation operated by the European Space Agency (ESA)
as part of the Copernicus Earth Observation programme. The Multispectral Instrument (MSI)
aboard each satellite captures 13 spectral bands ranging from visible to shortwave
infrared, at spatial resolutions of 10 m, 20 m, and 60 m (ESA, 2021). The combined
constellation provides a **5-day revisit frequency** at the equator and a **290 km swath
width**, making it one of the most capable freely available optical sensors for land
monitoring at scale.

We use **Level-2A** products — atmospherically corrected **surface reflectance** — downloaded
from the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu). All data
are free and open access under the Copernicus Open Access policy.

<p align="center">
  <img src="Figures/B3 - 560nm NDWI.png" 
       alt="Sentinel-2 MSI data acquisition over Rondônia, Brazil" 
       width="75%"/>
</p>

*The Sentinel-2 satellite constellation (ESA Copernicus) operates at 786 km altitude in a 
sun-synchronous orbit, capturing reflected solar radiance across 13 spectral bands via its 
Multispectral Instrument (MSI). The MSI uses a **pushbroom scanning** approach — rather than 
sweeping a mirror, a fixed linear detector array images the full 290 km swath width 
simultaneously as the satellite passes overhead. Light enters through a three-mirror TMA 
telescope (silicon carbide structure), is split by a dichroic beamsplitter into visible/NIR 
and SWIR wavelengths, and focused onto two focal plane assemblies: a CMOS array (VNIR, 
bands B1–B8a) and a cooled HgCdTe array (SWIR, B9–B12). Stripe filters mounted on the 
detectors separate individual spectral channels. This study extracts five bands 
(**B3, B4, B8, B11, B12**) from two dry-season Level-2A surface reflectance acquisitions 
over tile T20LKP (BR-364 corridor, Rondônia) in September 2019 and September 2022, 
deriving NDVI, NBR, and NDWI indices and their temporal deltas (ΔNDVI, ΔNBR) as 
features for Random Forest deforestation classification.*

**Scene selection rationale**

Two dry-season acquisitions over tile **T20LKP** (UTM Zone 20L) were selected,
covering the BR-364 corridor in central Rondônia:

| Scene | Filename | Acquisition date | Rationale |
|---|---|---|---|
| **Epoch 1** | `S2A_MSIL2A_20190916T143751_N0500_R096_T20LKP` | 16 Sept 2019 | Pre-change baseline; dry season minimises cloud cover and vegetation moisture stress |
| **Epoch 2** | `S2A_MSIL2A_20220920T143731_N0510_R096_T20LKP` | 20 Sept 2022 | Post-change image; same sensor, same orbit, same season — maximises spectral comparability |

Dry-season acquisition is critical because wet-season cloud cover in Rondônia routinely
exceeds 80%, making optical change detection impractical. Matching the calendar month
across years minimises phenological variation — ensuring that NDVI differences reflect
land cover change rather than seasonal vegetation cycles.

**Bands extracted and their roles**

| Band | Wavelength | Resolution | Role in pipeline |
|---|---|---|---|
| B3 | 560 nm (Green) | 10 m | NDWI (water content) |
| B4 | 665 nm (Red) | 10 m | NDVI (vegetation vigour) |
| B8 | 842 nm (NIR) | 10 m | NDVI, NBR, NDWI |
| B11 | 1610 nm (SWIR-1) | 20 m | Bare soil / moisture proxy |
| B12 | 2190 nm (SWIR-2) | 20 m | NBR (burn ratio) |
| SCL | — | 20 m | Scene Classification Layer → cloud mask |

**Preprocessing pipeline (Section 2)**

Each `.SAFE` archive is processed into a clean 5-band, NaN-masked, 10 m GeoTIFF:

1. B11, B12 and SCL are bilinearly resampled from 20 m to 10 m to match B3/B4/B8
2. All bands are converted from raw DN to surface reflectance by dividing by 10,000
3. SCL classes 3 (cloud shadow), 8 (medium cloud), 9 (high cloud) and 10 (cirrus)
   are set to `NaN` in all bands, removing contaminated pixels from all downstream analysis

Cloud cover was minimal for both acquisitions (<3%), which is typical for dry-season
Rondônia imagery and confirms the scene selection was appropriate.

---

### 5.3 · Hansen Global Forest Change Data

**What is Hansen GFC?**

The Hansen Global Forest Change dataset (Hansen et al., 2013) is the most widely used
global forest monitoring product in the scientific literature. It is derived from
time-series analysis of over 654,000 Landsat scenes and provides annual tree-cover
loss information at **30 m resolution** from 2000 to the present. The loss-year layer
encodes the year in which each pixel first experienced canopy loss, allowing
deforestation to be temporally attributed at annual resolution.

<p align="center">
  <img src="https://raw.githubusercontent.com/IonaBoulton/-GEOL0069-XAI-Deforestation_Detection_in_Rond-nia-Brazil-Random-Forest-Classification-and-TreeSHAP/main/Figures/Fig_LabelComparison.png" 
       width="90%"/>
</p>

<p align="center">
  <b>Figure X — Resolution mismatch: Hansen GFC (30 m) vs Spectral Threshold labels (10 m).</b>
  The zoomed regions (black boxes) reveal the core labelling challenge. Left: Hansen labels 
  show large rectangular blocks — artefacts of resampling 30 m Landsat pixels onto the 10 m 
  Sentinel-2 grid, where 1 Landsat pixel maps to 9 Sentinel-2 pixels. Many spectrally 
  deforested pixels are mislabelled as stable forest simply because they fall within a 
  majority-forest Landsat block. Right: Spectral threshold labels at native 10 m resolution 
  capture fine-grained individual clearings, narrow road incursions and small plot boundaries 
  invisible at 30 m. This spatial mismatch is the primary cause of Experiment A's near-random 
  accuracy (OA = 0.538, κ = 0.077) and directly motivates the dual-experiment design.
</p>

**Why Hansen for Experiment A?**

Hansen GFC is used as the label source in **Experiment A** because it represents an
**independent, externally validated ground truth** — the labels are entirely derived from
Landsat imagery and are not influenced by the Sentinel-2 features used to train the
model. This is methodologically the most rigorous approach and mirrors standard
practice in the remote sensing literature (Tyukavina et al., 2017).

**Tile and download**

The Hansen loss-year raster for tile T20LKP was downloaded via **Google Earth Engine
(GEE)** and exported as `Hansen_LossYear_T20LKP.tif`, co-registered to the Sentinel-2
tile extent. Pixels with loss-year values 19–22 (corresponding to 2019–2022) are
assigned **Class 1 (deforested)**; pixels with loss-year = 0 (no detected loss) are
assigned **Class 0 (stable forest)**.

**The resolution mismatch challenge**

A key limitation — and central motivation for Experiment B — is the spatial resolution
difference between Hansen (30 m) and Sentinel-2 (10 m). When resampled to 10 m via
nearest-neighbour interpolation, each Hansen pixel maps to approximately 9 Sentinel-2
pixels, creating "blocky" label boundaries that do not align with spectral edges in
the imagery. This misalignment is quantified in Section 7 (Experiment A achieves only
OA = 0.538) and critically discussed in Section 10.

