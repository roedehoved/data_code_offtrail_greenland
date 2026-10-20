"""
Step 1 - From raw GPS tracks to the analysis tables.

Inputs (data/raw/)
  gpx/<ship-day>_CLEANED/*.gpx   534 tracks from 11 cruise-ship visits, 5 s cadence
  trails/*.csv                    the designated trail network (WKT, EPSG:4326), by category
  dem/dem_32622.tif               ArcticDEM 2 m tile 18_38_1_2 v4.1, cropped to the site and
                                  reprojected to EPSG:32622 (see dem/*_meta.txt)

Outputs (data/processed/)
  pings.parquet       every fix: visitor, ship-day, arm, UTC time, metres, off_trail
  risk_set.parquet    every at-risk on-trail fix: spell, transition flag, and exposure to
                      another visitor already off the trail within 50 m in the previous 60 s
                      (with and without terrain line of sight)
  off_spells.parquet  every off-trail spell: onset time and place, duration, cell

Steps
  1. Read the GPX files. The exporting software shifted every calendar date by a constant
     7,168 days but kept the time of day, so each timestamp is rebuilt from the ship-day in the
     folder name plus the GPS time of day (UTC). Logger 60 on 29 July (L'Austral, CNT) was
     exported three times; only the final re-clean (_NEWNEW) is kept.
  2. Project to UTM zone 22N (EPSG:32622).
  3. DEM: the shipped cropped raster is used. If the full ArcticDEM tile is placed in
     data/raw/dem/, it is cropped and reprojected here instead.
  4. Classify each fix: off-trail if more than 25 m from the designated network (categories
     1-4). Informal paths (category 5) are themselves products of off-trail use and count as
     off-trail.
  5. Build the risk set of on-trail spells and transitions.
  6. Build off-trail spells (onsets and durations).
  7. Exposure at each at-risk fix to other visitors' off-trail fixes (used in the SI).

Run:  python code/1_process_data.py        (~10 min)
"""

from __future__ import annotations

import glob
import os
import re
import time
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

import common as C

FNAME_RE = re.compile(r"^(?P<day>\d{2})\.(?P<month>\d{2})_(?P<ship>[^_]+)_(?P<group>CNT|TMT)_"
                      r"Logger(?P<logger>\w+?)_CLEANED", re.IGNORECASE)
DROP_FILES = {"29.07_LAustral_CNT_Logger60_CLEANED.gpx",
              "29.07_LAustral_CNT_Logger60_CLEANED_NEW.gpx"}


def ingest() -> pd.DataFrame:
    import gpxpy
    import geopandas as gpd
    files = sorted(glob.glob(os.path.join(C.GPX_DIR, "**", "*.gpx"), recursive=True))
    recs, n_read = [], 0
    for fp in files:
        name = os.path.basename(fp)
        if name in DROP_FILES:
            continue
        m = FNAME_RE.match(name)
        if not m:
            raise ValueError(f"unexpected file name: {name}")
        n_read += 1
        day = date(2022, int(m["month"]), int(m["day"]))
        shipday = f"{m['day']}.{m['month']}_{m['ship']}"
        group, logger = m["group"].upper(), m["logger"]
        with open(fp, "rb") as fh:
            gpx = gpxpy.parse(fh.read().decode("utf-8", errors="replace"))
        for trk in gpx.tracks:
            for seg in trk.segments:
                for p in seg.points:
                    if p.time is None:
                        continue
                    ts = datetime(day.year, day.month, day.day, p.time.hour, p.time.minute,
                                  p.time.second, p.time.microsecond, tzinfo=timezone.utc)
                    recs.append((f"{shipday}_{group}_{logger}", shipday, group, logger, ts,
                                 p.longitude, p.latitude))
    df = pd.DataFrame(recs, columns=["pid", "shipday", "group", "logger", "timestamp", "lon", "lat"])
    df = df.drop_duplicates(subset=["pid", "timestamp"]).reset_index(drop=True)
    g = gpd.GeoSeries(gpd.points_from_xy(df["lon"], df["lat"]), crs=C.CRS_LL).to_crs(C.CRS_M)
    df["x_m"], df["y_m"] = g.x.to_numpy(), g.y.to_numpy()
    df["treatment"] = (df["group"] == "TMT").astype(int)     # pledge arm of the original trial
    print(f"  files read {n_read} (of {len(files)}; {len(files) - n_read} duplicate exports "
          f"dropped) | fixes {len(df):,} | visitors {df.pid.nunique()} | "
          f"ship-days {df.shipday.nunique()}")
    print(f"  dates {df.timestamp.min():%Y-%m-%d} to {df.timestamp.max():%Y-%m-%d}")
    return df


def crop_dem_if_tile_present(pings: pd.DataFrame, pad_m: float = 300.0) -> None:
    """Re-create dem_32622.tif from the full ArcticDEM tile, if the user has downloaded it."""
    if not os.path.exists(C.FULL_DEM_TILE):
        print(f"  using shipped {os.path.relpath(C.DEM_PATH, C.ROOT)}")
        return
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
    from rasterio.windows import from_bounds
    minx, maxx = pings.x_m.min() - pad_m, pings.x_m.max() + pad_m
    miny, maxy = pings.y_m.min() - pad_m, pings.y_m.max() + pad_m
    with rasterio.open(C.FULL_DEM_TILE) as src:
        win = from_bounds(*transform_bounds(C.CRS_M, src.crs, minx, miny, maxx, maxy),
                          transform=src.transform)
        data, wt, nodata = src.read(1, window=win), src.window_transform(win), src.nodata
        dt, w, h = calculate_default_transform(
            src.crs, C.CRS_M, data.shape[1], data.shape[0],
            *rasterio.transform.array_bounds(data.shape[0], data.shape[1], wt), resolution=2.0)
        dst = np.full((h, w), nodata, dtype=data.dtype)
        reproject(source=data, destination=dst, src_transform=wt, src_crs=src.crs,
                  dst_transform=dt, dst_crs=C.CRS_M, src_nodata=nodata, dst_nodata=nodata,
                  resampling=Resampling.bilinear)
    with rasterio.open(C.DEM_PATH, "w", driver="GTiff", dtype=dst.dtype, count=1, height=h,
                       width=w, crs=C.CRS_M, transform=dt, nodata=nodata, compress="deflate",
                       tiled=True) as out:
        out.write(dst, 1)
    print(f"  cropped full tile -> {os.path.relpath(C.DEM_PATH, C.ROOT)} ({w}x{h} px)")


def main():
    t0 = time.time()
    C.hr("1. INGEST GPX AND REBUILD TIMESTAMPS")
    pings = ingest()

    C.hr("2. DEM")
    crop_dem_if_tile_present(pings)

    C.hr("3. ON/OFF-TRAIL CLASSIFICATION (25 m, designated trails; informal paths off-trail)")
    lines = C.load_trail_lines()
    pings["off_trail"] = C.classify(pings.x_m.to_numpy(), pings.y_m.to_numpy(),
                                    C.build_buffer(lines, C.DESIGNATED, C.BUFFER_M))
    print(f"  off-trail fixes: {100 * pings.off_trail.mean():.2f}%")
    pings.to_parquet(os.path.join(C.PROC, "pings.parquet"), index=False)

    C.hr("4. RISK SET (on-trail spells and transitions)")
    risk = C.build_risk_set(pings[["pid", "shipday", "group", "treatment", "timestamp",
                                   "x_m", "y_m", "off_trail"]])
    risk = risk.drop(columns="off_trail")
    print(f"  at-risk fixes {len(risk):,} | spells {risk.spell_id.nunique():,} | "
          f"transitions {int(risk.event_next.sum())}")

    C.hr("5. OFF-TRAIL SPELLS")
    off = C.build_off_spells(pings)
    print(f"  spells {len(off)} | longer than five minutes {(off.dur_s > C.SEVERE_S).sum()}")
    off.to_parquet(os.path.join(C.PROC, "off_spells.parquet"), index=False)

    C.hr("6. EXPOSURE: ANOTHER VISITOR OFF THE TRAIL WITHIN 50 m IN THE PREVIOUS 60 s")
    sources = pings[pings.off_trail == 1]
    E = C.exposure(risk, sources,
                   variants=[("state", 50.0, 0, 60, True),            # terrain line of sight
                             ("state_distonly", 50.0, 0, 60, False),  # distance and time only
                             ("state120", 50.0, 0, 120, True)],
                   max_lag=120, max_r=50.0)
    for c in ("exp_state", "exp_state_distonly", "exp_state120"):
        risk[c] = E[c].to_numpy()
    risk.to_parquet(os.path.join(C.PROC, "risk_set.parquet"), index=False)

    C.hr("CHECKSUM (Table 1)")
    ph = len(risk) * C.FIX_S / 3600
    print(f"  fixes {len(pings):,} | visitors {pings.pid.nunique()} | at-risk fixes {len(risk):,} | "
          f"person-hours {ph:,.0f} | transitions {int(risk.event_next.sum())} | spells {len(off)} | "
          f"spells > 5 min {(off.dur_s > C.SEVERE_S).sum()}")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
