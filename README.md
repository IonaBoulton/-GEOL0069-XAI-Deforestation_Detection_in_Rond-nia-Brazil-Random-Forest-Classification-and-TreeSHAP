# GEOL0069-XAI-Deforestation_Detection_in_Rond-nia-Brazil-Random-Forest-Classification-and-TreeSHAP
Explainable Forest Loss Detection in Rondônia, Brazil: A Dual-Label Experiment Comparing Hansen GFC and Spectral Threshold Training Strategies for Sentinel-2 Random Forest Classification with TreeSHAP Analysis (2019–2022)

<p align="center">
  <img src="https://github.com/user-attachments/assets/1b7d8d83-687d-47c6-a088-8d71b777d9ef" alt="Amazon-deforestation-679x419" width="679" height="419">
  <br>
  <i>Amazon Rainforest Deforestation and logging </i>
</p>
COME BACK TO THIS ONCE CODE HAS RUN!!!!!

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

