"""
Shared constants and functions for the replication code.

Imported by 1_process_data.py, 2_main_analysis.py and 3_si_robustness.py. Nothing here runs on
import except the path set-up.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- paths ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
GPX_DIR = os.path.join(RAW, "gpx")
TRAIL_DIR = os.path.join(RAW, "trails")
DEM_PATH = os.path.join(RAW, "dem", "dem_32622.tif")
FULL_DEM_TILE = os.path.join(RAW, "dem", "18_38_1_2_2m_v4.1_dem.tif")   # optional, not shipped
PROC = os.path.join(ROOT, "data", "processed")
TAB = os.path.join(ROOT, "outputs", "tables")
FIG = os.path.join(ROOT, "outputs", "figures")
for _d in (PROC, TAB, FIG):
    os.makedirs(_d, exist_ok=True)

# ------------------------------------------------------------------------- constants ---
CRS_LL, CRS_M = "EPSG:4326", "EPSG:32622"   # input WGS84; analysis in UTM 22N metres
FIX_S = 5.0            # GPS cadence (s)
GAP_MAX_S = 60.0       # a recording gap longer than this breaks a spell
BUFFER_M = 25.0        # a fix > 25 m from the designated network is off-trail
CELL_M = 100.0         # primary spatial cell
SEVERE_S = 300.0       # sustained excursion: off-trail spell longer than five minutes
TOPK = 5               # number of cells in the primary concentration statistic
EYE = 1.6              # eye height for terrain line of sight (m)
STEP_S = 5             # time step used to batch the exposure search
SNAP_M = 4.0           # cell size for caching line-of-sight profiles

CATEGORY_FILES = {
    "cat1_boardwalk": "Mapping Trails- Category 1 Boardwalk.csv",
    "cat2_atv": "Mapping Trails- Category 2 ATV track (wide, clear path + dots).csv",
    "cat3_narrow": "Mapping Trails- Category 3 Narrow path + dots.csv",
    "cat4_dots": "Mapping Trails- Category 4 Dots only.csv",
    "cat5_informal": "Mapping Trails- Category 5 Pirate paths.csv",
}
DESIGNATED = ["cat1_boardwalk", "cat2_atv", "cat3_narrow", "cat4_dots"]   # informal = off-trail
MAP_TRAILS = "Mapping Trails- All marked trails.csv"                      # drawn on maps only


def hr(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def to_epoch_s(ts) -> pd.Series:
    """Epoch seconds from a datetime column, whatever its storage unit.

    Parquet stores microseconds. Casting to int64 and dividing by 1e9 (as if nanoseconds) silently
    collapses the clock into ~1,000 s blocks; read the unit instead."""
    ts = pd.to_datetime(ts, utc=True)
    unit = {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}[ts.dt.unit]
    return ts.astype("int64") // unit


def cell_of(x, y, size: float = CELL_M, ox: float = 0.0, oy: float = 0.0) -> np.ndarray:
    """Grid-cell label 'cx_cy' for metre coordinates."""
    cx = np.floor((np.asarray(x) - ox) / size).astype(np.int64)
    cy = np.floor((np.asarray(y) - oy) / size).astype(np.int64)
    return np.char.add(np.char.add(cx.astype(str), "_"), cy.astype(str))


# --------------------------------------------------------------------- trail network ---
def load_trail_lines() -> dict:
    """{category: GeoSeries of trail lines in EPSG:32622}."""
    import geopandas as gpd
    out = {}
    for cat, fname in CATEGORY_FILES.items():
        df = pd.read_csv(os.path.join(TRAIL_DIR, fname))
        out[cat] = gpd.GeoSeries.from_wkt(df["WKT"], crs=CRS_LL).to_crs(CRS_M)
    return out


def build_buffer(lines: dict, cats: list, width: float):
    from shapely.ops import unary_union
    return unary_union([g for c in cats for g in lines[c].geometry]).buffer(width)


def classify(x: np.ndarray, y: np.ndarray, buffer_geom) -> np.ndarray:
    """1 = off-trail (outside the buffer), 0 = on-trail."""
    import geopandas as gpd
    pts = gpd.GeoSeries(gpd.points_from_xy(x, y), crs=CRS_M)
    return (~pts.within(buffer_geom)).to_numpy().astype(int)


# ------------------------------------------------------------------------- risk set ---
def build_risk_set(pings: pd.DataFrame) -> pd.DataFrame:
    """One row per at-risk on-trail fix.

    A spell is a maximal run of consecutive on-trail fixes within a track segment (a gap
    > GAP_MAX_S breaks the segment). A spell ending in an off-trail fix without a gap is a
    transition; event_next = 1 on its last on-trail fix. Spells ending at a gap or at the end of
    the track are censored. Spell ids are 'pid__s<k>' in order within each visitor."""
    pings = pings.sort_values(["pid", "timestamp"]).reset_index(drop=True)
    frames = []
    for pid, g in pings.groupby("pid", sort=False):
        g = g.reset_index(drop=True)
        off = g["off_trail"].to_numpy().astype(int)
        dt = np.full(len(g), np.inf)
        if len(g) > 1:
            dt[1:] = g["timestamp"].diff().dt.total_seconds().to_numpy()[1:]
        seg_break = dt > GAP_MAX_S
        rows, k, start, in_spell = [], -1, None, False

        def close(transition: bool):
            nonlocal rows, in_spell
            if rows:
                sub = g.loc[rows].copy()
                sub["spell_id"] = f"{pid}__s{k}"
                sub["time_in_spell_s"] = (sub["timestamp"] - start).dt.total_seconds()
                sub["spell_outcome"] = "transition" if transition else "censored"
                sub["event_next"] = 0
                if transition:
                    sub.iloc[-1, sub.columns.get_loc("event_next")] = 1
                frames.append(sub)
            rows, in_spell = [], False

        for i in range(len(g)):
            if seg_break[i] and in_spell:
                close(False)
            if off[i] == 0:
                if not in_spell:
                    in_spell, k, start, rows = True, k + 1, g.loc[i, "timestamp"], []
                rows.append(i)
            elif in_spell:
                close(not seg_break[i])
        if in_spell:
            close(False)
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------- off-trail spells ---
def build_off_spells(p: pd.DataFrame) -> pd.DataFrame:
    """One row per off-trail spell: visitor, ship-day, start time and position, duration, cell.

    A spell is a maximal run of off-trail fixes; a gap > GAP_MAX_S starts a new one. The first
    fix of each spell is its onset. Duration = number of fixes x 5 s."""
    p = p.sort_values(["pid", "timestamp"]).copy()
    p["off"] = p["off_trail"].astype(int)
    dt = p.groupby("pid")["timestamp"].diff().dt.total_seconds()
    newseg = (p.groupby("pid")["off"].diff().fillna(1) != 0) | dt.isna() | (dt > GAP_MAX_S)
    p["seg"] = newseg.groupby(p["pid"]).cumsum()
    off = (p[p["off"] == 1].groupby(["pid", "seg"])
           .agg(fixes=("off", "size"), timestamp=("timestamp", "first"),
                x_m=("x_m", "first"), y_m=("y_m", "first"), shipday=("shipday", "first"))
           .reset_index())
    off["dur_s"] = off["fixes"] * FIX_S
    off["cell"] = cell_of(off["x_m"], off["y_m"])
    off["t_s"] = to_epoch_s(off["timestamp"]).astype(np.int64)
    return off.drop(columns="seg")


# ---------------------------------------------------------------------- concentration ---
def cell_table(at: pd.DataFrame, size: float = CELL_M, ox: float = 0.0, oy: float = 0.0):
    """Per-cell at-risk person-hours, transitions and rate, sorted by transitions."""
    cx = np.floor((at["x_m"].to_numpy() - ox) / size).astype(np.int64)
    cy = np.floor((at["y_m"].to_numpy() - oy) / size).astype(np.int64)
    d = pd.DataFrame({"cx": cx, "cy": cy, "ev": at["event_next"].to_numpy().astype(float)})
    g = d.groupby(["cx", "cy"]).agg(fixes=("ev", "size"), events=("ev", "sum")).reset_index()
    g["person_hours"] = g["fixes"] * FIX_S / 3600.0
    g["rate_per_100ph"] = 100.0 * g["events"] / g["person_hours"]
    return g.sort_values("events", ascending=False).reset_index(drop=True)


def top_k_concentration(g: pd.DataFrame, k: int):
    """(share of events, share of at-risk time, ratio) for the k cells with the most events.
    g must be sorted by events, as cell_table returns it."""
    top = g.head(k)
    se = top["events"].sum() / g["events"].sum()
    sh = top["person_hours"].sum() / g["person_hours"].sum()
    return se, sh, (se / sh if sh > 0 else np.nan)


def hoover(g: pd.DataFrame) -> float:
    """Largest gap between cumulative share of events and of at-risk time, cells ranked by rate."""
    d = g.sort_values("rate_per_100ph", ascending=False)
    ce = np.cumsum(d["events"].to_numpy()) / d["events"].sum()
    ch = np.cumsum(d["person_hours"].to_numpy()) / d["person_hours"].sum()
    return float(np.max(ce - ch))


def count_concentration(counts: pd.Series, ph: pd.Series, k: int = TOPK):
    """As top_k_concentration, for event counts and person-hours indexed by cell label."""
    if counts.sum() == 0:
        return np.nan, np.nan, np.nan
    top = counts.sort_values(ascending=False).head(k)
    se = top.sum() / counts.sum()
    sh = ph.reindex(top.index).fillna(0).sum() / ph.sum()
    return se, sh, (se / sh if sh > 0 else np.nan)


def concentration_variants(at: pd.DataFrame) -> pd.DataFrame:
    """Top-k concentration under the primary grid and every robustness variant."""
    rows = []
    g = cell_table(at)
    for k in (5, 10, 20):
        se, sh, r = top_k_concentration(g, k)
        rows.append(dict(variant="primary 100 m", detail=f"top {k}", k=k, share_events=se,
                         share_time=sh, conc_ratio=r, hoover=hoover(g)))
    for ox in (0.0, 25.0, 50.0, 75.0):
        for oy in (0.0, 25.0, 50.0, 75.0):
            gg = cell_table(at, CELL_M, ox, oy)
            for k in (5, 10, 20):
                se, sh, r = top_k_concentration(gg, k)
                rows.append(dict(variant="shifted origin", detail=f"dx={ox:.0f} dy={oy:.0f}",
                                 k=k, share_events=se, share_time=sh, conc_ratio=r,
                                 hoover=hoover(gg)))
    for size in (50.0, 100.0, 200.0):
        gg = cell_table(at, size)
        for k in (5, 10, 20):
            se, sh, r = top_k_concentration(gg, k)
            rows.append(dict(variant="resolution", detail=f"{size:.0f} m", k=k, share_events=se,
                             share_time=sh, conc_ratio=r, hoover=hoover(gg)))
    for day, sub in at.groupby("shipday"):
        gg = cell_table(sub)
        if gg["events"].sum() < 5:
            continue
        se, sh, r = top_k_concentration(gg, 5)
        rows.append(dict(variant="ship-day", detail=str(day), k=5, share_events=se,
                         share_time=sh, conc_ratio=r, hoover=hoover(gg)))
    for mp in (0.0, 1.0, 5.0, 10.0):
        gg = g[g["person_hours"] >= mp].copy()
        se, sh, r = top_k_concentration(gg, 5)
        rows.append(dict(variant="min person-hours", detail=f">= {mp:.0f} h", k=5,
                         share_events=se, share_time=sh, conc_ratio=r, hoover=hoover(gg)))
    return pd.DataFrame(rows)


DUR_CLASSES = [(0, 15, "blip <=15 s"), (15, 60, "short 16-60 s"),
               (60, 300, "medium 61-300 s"), (300, np.inf, "substantial >5 min")]


def severity_concentration(off: pd.DataFrame, at: pd.DataFrame) -> pd.DataFrame:
    """Concentration by duration class; each class takes the five cells holding most of it."""
    ph_all = at.groupby("cell").size() * FIX_S / 3600.0
    off = off.copy()
    off["dur_class"] = pd.cut(off["dur_s"], [b[0] for b in DUR_CLASSES] + [np.inf],
                              labels=[b[2] for b in DUR_CLASSES], right=True, include_lowest=True)
    rows = []
    for lab in [b[2] for b in DUR_CLASSES] + ["ALL"]:
        sub = off if lab == "ALL" else off[off["dur_class"] == lab]
        se, sh, r = count_concentration(sub.groupby("cell").size(), ph_all)
        rows.append(dict(dur_class=lab, n=len(sub), pct_of_spells=len(sub) / len(off),
                         pct_of_off_time=sub["dur_s"].sum() / off["dur_s"].sum(),
                         share_events=se, share_time=sh, conc_ratio=r))
    return pd.DataFrame(rows)


def heldout_targeting(at: pd.DataFrame, sev: pd.DataFrame, kgrid=(1, 2, 3, 5, 10, 20)):
    """Leave-one-ship-day-out targeting of sustained excursions.

    Returns (per_day, summary). per_day has one row per held-out day for: the primary rule (five
    cells with the most such excursions on the training days), the nested rule (k chosen by
    leave-one-day-out lift inside the training days) and the traffic benchmark (five cells with
    the most at-risk time on the training days). summary has the k grid and the means."""
    days = sorted(sev.shipday.unique())

    def score(top, d):
        te = sev[sev.shipday == d]
        phd = at[at.shipday == d].groupby("cell").size() * FIX_S / 3600.0
        cap = te.cell.isin(top).mean()
        shr = phd.reindex(top).fillna(0).sum() / phd.sum()
        return len(te), cap, shr, (cap / shr if shr > 0 else np.nan)

    def by_count(train, k):
        return train.groupby("cell").size().sort_values(ascending=False).head(k).index

    per_day, ksum = [], []
    for k in kgrid:
        res = []
        for d in days:
            if (sev.shipday == d).sum() < 3:
                continue
            res.append(score(by_count(sev[sev.shipday != d], k), d))
        res = np.array(res)
        ksum.append(dict(check="k sensitivity", k=k, mean_capture=res[:, 1].mean(),
                         mean_time_share=res[:, 2].mean(), mean_lift=np.nanmean(res[:, 3]),
                         days_lift_gt1=int((res[:, 3] > 1).sum()), n_days=len(res)))
    for d in days:
        if (sev.shipday == d).sum() < 3:
            continue
        tr = sev[sev.shipday != d]
        n, cap, shr, lift = score(by_count(tr, TOPK), d)
        per_day.append(dict(rule="primary", shipday=d, k=TOPK, n_test=n, capture=cap,
                            time_share=shr, lift=lift))
        best_k, best = None, -np.inf
        for k in kgrid:
            inner = []
            for d2 in [x for x in days if x != d]:
                tr2, te2 = tr[tr.shipday != d2], tr[tr.shipday == d2]
                if len(te2) < 3:
                    continue
                top2 = by_count(tr2, k)
                ph2 = at[at.shipday == d2].groupby("cell").size() * FIX_S / 3600.0
                s2 = ph2.reindex(top2).fillna(0).sum() / ph2.sum()
                if s2 > 0:
                    inner.append(te2.cell.isin(top2).mean() / s2)
            if inner and np.mean(inner) > best:
                best, best_k = np.mean(inner), k
        n, cap, shr, lift = score(by_count(tr, best_k), d)
        per_day.append(dict(rule="nested", shipday=d, k=best_k, n_test=n, capture=cap,
                            time_share=shr, lift=lift))
        top = at[at.shipday != d].groupby("cell").size().sort_values(ascending=False).head(TOPK).index
        n, cap, shr, lift = score(top, d)
        per_day.append(dict(rule="traffic benchmark", shipday=d, k=TOPK, n_test=n, capture=cap,
                            time_share=shr, lift=lift))
    per_day = pd.DataFrame(per_day)
    summ = pd.DataFrame(ksum)
    for rule in ("nested", "traffic benchmark"):
        s = per_day[per_day.rule == rule]
        summ = pd.concat([summ, pd.DataFrame([dict(
            check=rule, k=int(s.k.mode().iloc[0]), mean_capture=s.capture.mean(),
            mean_time_share=s.time_share.mean(), mean_lift=s.lift.mean(),
            days_lift_gt1=int((s.lift > 1).sum()), n_days=len(s))])], ignore_index=True)
    return per_day, summ


# ------------------------------------------------------------- terrain line of sight ---
class DEM:
    """In-memory DEM with bilinear sampling in projected (metre) coordinates."""

    def __init__(self, path: str = DEM_PATH):
        import rasterio
        with rasterio.open(path) as src:
            self.arr = src.read(1).astype(np.float64)
            self.nodata = src.nodata
            self.transform = src.transform
            self.inv = ~src.transform
            self.height, self.width = self.arr.shape
        self.arr = np.where(self.arr == self.nodata, np.nan, self.arr)

    def sample(self, xs, ys) -> np.ndarray:
        cols, rows = self.inv * (xs, ys)
        cols, rows = np.asarray(cols), np.asarray(rows)
        c0, r0 = np.floor(cols).astype(int), np.floor(rows).astype(int)
        fc, fr = cols - c0, rows - r0
        c1, r1 = c0 + 1, r0 + 1
        ok = (c0 >= 0) & (r0 >= 0) & (c1 < self.width) & (r1 < self.height)
        out = np.full(cols.shape, np.nan)
        if not ok.any():
            return out
        z00 = self.arr[r0[ok], c0[ok]]; z01 = self.arr[r0[ok], c1[ok]]
        z10 = self.arr[r1[ok], c0[ok]]; z11 = self.arr[r1[ok], c1[ok]]
        top = z00 * (1 - fc[ok]) + z01 * fc[ok]
        bot = z10 * (1 - fc[ok]) + z11 * fc[ok]
        out[ok] = top * (1 - fr[ok]) + bot * fr[ok]
        return out


class ReliefCache:
    """Maximum interior height of the ground above the straight ground-to-ground chord between two
    points, sampled every 2 m. Two people at eye height EYE can see each other when this is
    <= EYE. Cached by pairs of snapped 4 m cells."""

    def __init__(self, dem: DEM):
        self.dem = dem
        self.cache: dict = {}

    def _compute(self, ax, ay, bx, by) -> float:
        dist = float(np.hypot(bx - ax, by - ay))
        f = np.linspace(0.0, 1.0, max(int(dist / 2.0) + 1, 2))
        g = self.dem.sample(ax + f * (bx - ax), ay + f * (by - ay))
        relief = (g - (g[0] + f * (g[-1] - g[0])))[1:-1]
        relief = relief[np.isfinite(relief)]
        return float(relief.max()) if relief.size else -np.inf

    def get_many(self, axs, ays, bxs, bys) -> np.ndarray:
        ca = np.round(np.asarray(axs) / SNAP_M).astype(np.int64)
        cb = np.round(np.asarray(ays) / SNAP_M).astype(np.int64)
        cc = np.round(np.asarray(bxs) / SNAP_M).astype(np.int64)
        cd = np.round(np.asarray(bys) / SNAP_M).astype(np.int64)
        swap = (ca > cc) | ((ca == cc) & (cb > cd))
        keys = pd.MultiIndex.from_arrays([np.where(swap, cc, ca), np.where(swap, cd, cb),
                                          np.where(swap, ca, cc), np.where(swap, cb, cd)])
        codes, uniq = pd.factorize(keys)
        vals = np.empty(len(uniq), dtype=np.float32)
        for u, key in enumerate(uniq):
            v = self.cache.get(key)
            if v is None:
                v = self._compute(key[0] * SNAP_M, key[1] * SNAP_M, key[2] * SNAP_M,
                                  key[3] * SNAP_M)
                self.cache[key] = v
            vals[u] = v
        return vals[codes]


def exposure_one_day(targets: pd.DataFrame, sources: pd.DataFrame, relief: ReliefCache,
                     variants, max_lag: int, max_r: float) -> pd.DataFrame:
    """Exposure of each target fix to source events of OTHER visitors on one ship-day.

    targets/sources need x_m, y_m, t_s, pid_code, step; targets also g_idx. variants is a list of
    (name, radius, lag_lo, lag_hi, use_los) with lag = t_target - t_source (lag > 0: the source
    came first). Returns g_idx plus exp_<name> (any qualifying source) and nv_<name> (number of
    distinct qualifying visitors)."""
    tx, ty = targets["x_m"].to_numpy(), targets["y_m"].to_numpy()
    tt, tpid = targets["t_s"].to_numpy(), targets["pid_code"].to_numpy()
    tstep, g_idx = targets["step"].to_numpy(), targets["g_idx"].to_numpy()
    out = {f"exp_{v[0]}": np.zeros(len(targets), bool) for v in variants}
    out.update({f"nv_{v[0]}": np.zeros(len(targets), np.int16) for v in variants})
    if len(sources) == 0:
        return pd.DataFrame({"g_idx": g_idx, **out})
    ds = sources.sort_values("step")
    sx, sy = ds["x_m"].to_numpy(), ds["y_m"].to_numpy()
    st, spid, sstep = ds["t_s"].to_numpy(), ds["pid_code"].to_numpy(), ds["step"].to_numpy()
    reach = max_lag // STEP_S + 1
    order = np.argsort(tstep)
    steps, starts = np.unique(tstep[order], return_index=True)
    starts = list(starts) + [len(order)]
    chunks = []
    for i, step in enumerate(steps):
        loc = order[starts[i]:starts[i + 1]]
        lo = np.searchsorted(sstep, step - reach, "left")
        hi = np.searchsorted(sstep, step + reach, "right")
        if hi <= lo:
            continue
        S = slice(lo, hi)
        dx = tx[loc][:, None] - sx[S][None, :]
        dy = ty[loc][:, None] - sy[S][None, :]
        d2 = dx * dx + dy * dy
        lag = tt[loc][:, None] - st[S][None, :]
        m = (d2 <= max_r * max_r) & (tpid[loc][:, None] != spid[S][None, :]) & \
            (lag >= -max_lag) & (lag <= max_lag)
        ii, jj = np.where(m)
        if ii.size:
            chunks.append((loc[ii], spid[S][jj], np.sqrt(d2[ii, jj]), lag[ii, jj],
                           tx[loc[ii]], ty[loc[ii]], sx[S][jj], sy[S][jj]))
    if not chunks:
        return pd.DataFrame({"g_idx": g_idx, **out})
    cat = [np.concatenate([c[j] for c in chunks]) for j in range(8)]
    pairs = pd.DataFrame({"tloc": cat[0], "spid": cat[1], "dist": cat[2].astype(np.float32),
                          "lag": cat[3].astype(np.float32),
                          "mrel": relief.get_many(cat[4], cat[5], cat[6], cat[7])})
    for name, r, lag_lo, lag_hi, use_los in variants:
        m = (pairs["dist"] <= r) & (pairs["lag"] >= lag_lo) & (pairs["lag"] <= lag_hi)
        if use_los:
            m &= pairs["mrel"] <= EYE
        nv = pairs[m].groupby("tloc")["spid"].nunique()
        out[f"exp_{name}"][nv.index.to_numpy()] = True
        out[f"nv_{name}"][nv.index.to_numpy()] = nv.to_numpy().astype(np.int16)
    return pd.DataFrame({"g_idx": g_idx, **out})


def exposure(targets: pd.DataFrame, sources: pd.DataFrame, variants, max_lag: int,
             max_r: float, relief: ReliefCache | None = None) -> pd.DataFrame:
    """Run exposure_one_day over every ship-day. Returns one row per target, in target order."""
    t = targets[["pid", "shipday", "timestamp", "x_m", "y_m"]].copy()
    t["g_idx"] = np.arange(len(t))
    s = sources[["pid", "shipday", "timestamp", "x_m", "y_m"]].copy()
    code = {p: i for i, p in enumerate(sorted(set(t["pid"]) | set(s["pid"])))}
    for d in (t, s):
        d["pid_code"] = d["pid"].map(code).astype(np.int32)
        d["t_s"] = to_epoch_s(d["timestamp"]).astype(np.int64)
    relief = relief or ReliefCache(DEM())
    res = []
    for day in sorted(t["shipday"].unique()):
        rt, sd = t[t.shipday == day].copy(), s[s.shipday == day].copy()
        dmin = int(min(rt["t_s"].min(), sd["t_s"].min() if len(sd) else rt["t_s"].min()))
        rt["step"] = (rt["t_s"] - dmin) // STEP_S
        sd["step"] = (sd["t_s"] - dmin) // STEP_S if len(sd) else pd.Series([], dtype=int)
        res.append(exposure_one_day(rt, sd, relief, variants, max_lag, max_r))
        print(f"    {day:<24} targets {len(rt):>7,}  sources {len(sd):>6,}", flush=True)
    E = pd.concat(res, ignore_index=True).set_index("g_idx").reindex(range(len(t)))
    for c in E.columns:
        E[c] = E[c].fillna(0).astype(bool if c.startswith("exp_") else np.int16)
    E["pid_code"] = t["pid_code"].to_numpy()
    return E.reset_index(drop=True)


# ------------------------------------------------------------------- space-time bursts ---
def components(src: pd.DataFrame, r: float, w: float) -> np.ndarray:
    """Single-linkage space-time components of onsets within r metres and w seconds."""
    lab = np.full(len(src), -1, dtype=np.int64)
    nxt = 0
    for _, g in src.groupby("shipday"):
        idx = g.index.to_numpy()
        x, y, t = g["x_m"].to_numpy(), g["y_m"].to_numpy(), g["t_s"].to_numpy()
        n = len(idx)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a in range(n):
            for b in range(a + 1, n):
                if abs(t[a] - t[b]) <= w and (x[a] - x[b]) ** 2 + (y[a] - y[b]) ** 2 <= r * r:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
        roots = {}
        for a in range(n):
            rt = find(a)
            if rt not in roots:
                roots[rt] = nxt + len(roots)
            lab[idx[a]] = roots[rt]
        nxt += len(roots)
    return lab


def envelope(src: pd.DataFrame, relief: ReliefCache, windows=(15, 30, 60, 120),
             radii=(25.0, 50.0, 100.0)) -> pd.DataFrame:
    """Compatibility envelope: number of onsets with another visitor's onset within w seconds
    before and r metres, with and without terrain line of sight."""
    pairs = []
    for day, g in src.groupby("shipday"):
        g = g.sort_values("t_s")
        x, y, t, pid = (g["x_m"].to_numpy(), g["y_m"].to_numpy(), g["t_s"].to_numpy(),
                        g["pid"].to_numpy())
        gi = g.index.to_numpy()
        for a in range(len(g)):
            dt = t[a] - t
            dist = np.hypot(x[a] - x, y[a] - y)
            m = (dt > 0) & (dt <= max(windows)) & (dist <= max(radii)) & (pid != pid[a])
            for b in np.where(m)[0]:
                pairs.append((int(gi[a]), float(dt[b]), float(dist[b]), x[a], y[a], x[b], y[b]))
    P = pd.DataFrame(pairs, columns=["i", "dt", "dist", "xi", "yi", "xj", "yj"])
    P["los"] = relief.get_many(P.xi.to_numpy(), P.yi.to_numpy(), P.xj.to_numpy(),
                               P.yj.to_numpy()) <= EYE
    rows = []
    for los in (True, False):
        for w in windows:
            for r in radii:
                m = (P.dt <= w) & (P.dist <= r)
                if los:
                    m &= P.los
                C = int(P.loc[m, "i"].nunique())
                rows.append(dict(los="on" if los else "off", window_s=w, radius_m=r,
                                 n_onsets=len(src), C_compatible=C,
                                 pct=round(100 * C / len(src), 1)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- loading ---
def load_processed():
    """The processed tables written by 1_process_data.py."""
    at = pd.read_parquet(os.path.join(PROC, "risk_set.parquet"))
    at["event_next"] = at["event_next"].astype(int)
    at["cell"] = cell_of(at["x_m"], at["y_m"])
    off = pd.read_parquet(os.path.join(PROC, "off_spells.parquet"))
    pings = pd.read_parquet(os.path.join(PROC, "pings.parquet"))
    return at, off, pings


# ---------------------------------------------------------------------- figure style ---
INK, GREY, LIGHT, PALE = "#1F1F1F", "#8C8C8C", "#D9D9D9", "#F2F2F2"
ACCENT, ACCENT2 = "#B5432E", "#2B5D8C"
SEQ_CMAP, HEAT_CMAP = "YlOrRd", "Blues"


def apply_figure_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"], "font.size": 8.5, "axes.titlesize": 9,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 8.5,
        "axes.labelcolor": INK, "axes.edgecolor": INK, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "xtick.color": INK, "ytick.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3,
        "ytick.major.size": 3, "legend.fontsize": 8, "legend.frameon": False,
        "lines.linewidth": 1.2, "figure.dpi": 100, "savefig.dpi": 400, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04, "text.color": INK, "axes.unicode_minus": False})


def panel_label(ax, letter, x=-0.02, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="right", color=INK)


def scale_bar(ax, length_m=500, loc=(0.04, 0.06)):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    bx, by = x0 + loc[0] * (x1 - x0), y0 + loc[1] * (y1 - y0)
    ax.plot([bx, bx + length_m], [by, by], color=INK, lw=2.0, solid_capstyle="butt")
    ax.text(bx + length_m / 2, by + 0.025 * (y1 - y0), f"{length_m:.0f} m", ha="center",
            va="bottom", fontsize=7.5, color=INK)


def north_arrow(ax, loc=(0.965, 0.90)):
    ax.annotate("N", xy=loc, xytext=(loc[0], loc[1] - 0.10), xycoords="axes fraction",
                textcoords="axes fraction", ha="center", va="center", fontsize=8.5,
                color=INK, arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK))


def map_axes(ax):
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def map_layers(pings: pd.DataFrame, pad_m: float = 220.0) -> dict:
    """Trail lines, a sample of on-trail fixes, all off-trail fixes and the map extent."""
    import geopandas as gpd
    trails = gpd.GeoSeries.from_wkt(pd.read_csv(os.path.join(TRAIL_DIR, MAP_TRAILS))["WKT"],
                                    crs=CRS_LL).to_crs(CRS_M)
    x0, y0, x1, y1 = trails.total_bounds
    xy = pings[["x_m", "y_m", "off_trail"]]
    return dict(trails=trails, on=xy[xy.off_trail == 0].sample(45000, random_state=0),
                off=xy[xy.off_trail == 1], bbox=(x0 - pad_m, x1 + pad_m, y0 - pad_m, y1 + pad_m))


def draw_trails(ax, trails, lw=1.1, color=INK):
    for geom in trails:
        for g in ([geom] if geom.geom_type == "LineString" else list(geom.geoms)):
            xs, ys = g.xy
            ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=3)


def set_bbox(ax, bbox):
    ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3])
