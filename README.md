# data_code_offtrail_greenland
The data are GPS tracks of 532 cruise-ship visitors to the Ilulissat Icefjord UNESCO World Heritage Site, Greenland.

# Replication package

Data and code for the analyses of off-trail behaviour at the Ilulissat Icefjord, Greenland
(UNESCO World Heritage Site), from GPS tracks of 532 cruise-ship visitors on 11 ship-days in July
and August 2022. The package reproduces every table and figure in the main text and the
Supporting Information robustness checks.

## Quick start

```
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
python run_all.py
```

Total run time is about 15 minutes on a laptop. Results are written to `outputs/tables` (CSV)
and `outputs/figures` (PNG). Randomised steps use fixed seeds, so results reproduce exactly.

## Contents

```
data/raw/gpx/<ship-day>_CLEANED/*.gpx   534 GPS tracks, one per logger (5 s cadence)
data/raw/trails/*.csv                   trail network as WKT lines (EPSG:4326), by category
data/raw/dem/dem_32622.tif              2 m elevation model cropped to the site (EPSG:32622)
data/raw/dem/*_meta.txt                 provenance of the source ArcticDEM tile
code/common.py                          shared definitions and functions
code/1_process_data.py                  raw data -> data/processed/
code/2_main_analysis.py                 main-text tables and figures
code/3_si_robustness.py                 Supporting Information robustness checks
run_all.py                              runs the three scripts in order
```

## Data

**GPS tracks.** Collected for a randomised trial of a commitment pledge (Cooper, Dolnicar & Grün,
2024, *Tourism Management* 104, 104928). File names give the ship-day, the trial arm (`CNT`
no pledge, `TMT` pledge) and the logger number. Participants are identified only by ship-day,
arm and logger number. The trial arm is used in one SI check and nowhere else.

Two processing decisions are made in `1_process_data.py`:
- The exporting software shifted every calendar date by a constant 7,168 days but kept the time
  of day. Timestamps are rebuilt from the ship-day in the folder name and the GPS time of day
  (UTC).
- Logger 60 on 29 July (L'Austral, CNT) was exported three times. Only the final re-clean,
  `..._CLEANED_NEWNEW.gpx`, is used; the other two files are kept for transparency and skipped.

**Trails.** Hand-mapped lines in five categories: 1 boardwalk, 2 ATV track, 3 narrow path,
4 marked route, 5 informal paths. Categories 1–4 form the designated network. A fix more than
25 m from it is off-trail. Informal paths are themselves the product of off-trail walking and
count as off-trail. `Mapping Trails- All marked trails.csv` is used only to draw maps.

**Elevation.** ArcticDEM mosaic tile 18_38_1_2, 2 m, version 4.1 (Polar Geospatial Center),
cropped to the site with a 300 m margin and reprojected to UTM 22N. The full tile (1.4 GB) is
not included. To rebuild the cropped raster from it, download the tile, place
`18_38_1_2_2m_v4.1_dem.tif` in `data/raw/dem/`, and run step 1; it detects the tile and
re-crops.

## Definitions

- **At-risk fix:** a 5 s fix from a visitor on the trail, who can therefore step off.
- **Transition:** the last on-trail fix of a spell that ends by stepping off. A recording gap
  longer than 60 s ends a spell without a transition.
- **Off-trail spell:** a run of consecutive off-trail fixes. Its first fix is the onset; its
  duration is the number of fixes times 5 s. A sustained excursion lasts more than five minutes.
- **The five cells:** the five 100 m grid cells holding the most events of the kind in question,
  selected by count. Rates order the Lorenz curve and the Hoover index.
- **Concentration ratio:** the five cells' share of events divided by their share of at-risk time.
- **Line of sight:** terrain only. Two people 1.6 m tall can see each other when no ground
  between them rises above the line joining their eyes.

## Outputs and where they appear

| Paper item | File |
|---|---|
| Table 1 | `table1_design_sample.csv` |
| Table 2 | `table2_concentration.csv` |
| Table 3 | `table3_heldout_targeting.csv` |
| Table 4 | `table4_envelope.csv` |
| Figure 1 | `fig1_site.png` |
| Figure 2 | `fig2_concentration.png` |
| Figure 3 | `fig3_leadlag.png` (data: `tableS5_leadlag.csv`) |
| Figure 4 | `fig4_envelope.png` |
| Table S1 | `tableS1_cells.csv` |
| Table S2, Figure S1 | `tableS2_robustness.csv`, `figS1_robustness.png` |
| Table S3, Figure S2 | `tableS3_heldout_per_day.csv`, `figS2_heldout_days.png` |
| Figure S3 | `figS3_k_sensitivity.png` |
| Table S4, Figure S4 | `tableS4_bursts.csv`, `figS4_bursts.png` |
| Table S5 | `tableS5_leadlag.csv` |
| Figure S5 | `figS5_envelope_los.png` |
| Trail buffer and informal-path sensitivity | `tableS_trail_definition.csv` |
| Concentration within each trial arm | `tableS_pledge_arm.csv` |
| Duration classes in fixed cells | `tableS_duration_fixed_cells.csv` |
| Visitor already off the trail nearby | `tableS_state_cue.csv` |

Files named `result_*.csv` are intermediate results written by step 2 and reused by step 3.

## Not included

The SI also describes analyses of why peer influence cannot be identified in these data (the
initiator-classification problem and a halted sequential-arrival design). Those are reported in
the text but are not part of this package.


