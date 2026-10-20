"""
Step 2 - Main-text results: Tables 1-4 and Figures 1-4.

Reads data/processed/ (written by 1_process_data.py) and writes outputs/tables/ and
outputs/figures/. Also writes the result files the SI script reuses (lead-lag profile, held-out
results, cell table), so the expensive lead-lag computation runs once.

  Table 1   table1_design_sample.csv
  Table 2   table2_concentration.csv
  Table 3   table3_heldout_targeting.csv
  Table 4   table4_envelope.csv
  Figure 1  fig1_site.png              study site, trails, on- and off-trail fixes
  Figure 2  fig2_concentration.png     (a) rate map (b) concentration against opportunity
                                       (c) by duration class (d) held-out targeting
  Figure 3  fig3_leadlag.png           lead-lag association profile
  Figure 4  fig4_envelope.png          compatibility envelope

Definitions used throughout
  * Unit: one at-risk on-trail 5 s fix. Event: the last on-trail fix of a spell that ends by
    stepping off (a transition). 700 transitions on 560,048 fixes.
  * "The five cells": the five 100 m cells holding the MOST events of the kind in question
    (selected by count). Rates order the Lorenz curve and the Hoover index.
  * Concentration ratio: share of events in the five cells / their share of at-risk time.
  * Sustained excursion: off-trail spell longer than five minutes.
  * Held-out targeting: each ship-day held out in turn; cells chosen on the other ten.

Run:  python code/2_main_analysis.py        (~6 min; the lead-lag bootstrap dominates)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch

import common as C

FIG, TAB, CELL = C.FIG, C.TAB, C.CELL_M
LL_HALF, LL_BIN, LL_RADIUS, N_BOOT, SEED = 300, 15, 50.0, 300, 20260902


# ---------------------------------------------------------------------- computations ---
def leadlag_profile(at: pd.DataFrame, onsets: pd.DataFrame) -> pd.DataFrame:
    """Transition rate among at-risk fixes of other visitors within 50 m and terrain line of sight
    of an off-trail onset, by signed lag (fix time minus onset time) in 15 s bins, as a ratio to
    the site-wide rate. 95% interval: bootstrap over visitors (300 draws)."""
    edges = np.arange(-LL_HALF, LL_HALF + LL_BIN, LL_BIN)
    V = [(f"b{int(lo)}", LL_RADIUS, int(lo), int(lo + LL_BIN - 1), True) for lo in edges[:-1]]
    E = C.exposure(at, onsets, V, max_lag=LL_HALF, max_r=LL_RADIUS)
    ev = at["event_next"].to_numpy().astype(float)
    pid_code = E["pid_code"].to_numpy()
    n_person = pid_code.max() + 1
    base = ev.mean()
    rng = np.random.default_rng(SEED)
    boot = [rng.integers(0, n_person, n_person) for _ in range(N_BOOT)]
    rows = []
    for name, *_ in V:
        m = E[f"exp_{name}"].to_numpy()
        n_fix, n_ev = int(m.sum()), float(ev[m].sum())
        rate = n_ev / n_fix if n_fix else np.nan
        pf = np.bincount(pid_code[m], minlength=n_person).astype(float)
        pe = np.bincount(pid_code[m], weights=ev[m], minlength=n_person)
        bs = [pe[i].sum() / pf[i].sum() for i in boot if pf[i].sum() > 0]
        lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
        lag_lo = int(name[1:])
        rows.append(dict(lag_lo=lag_lo, lag_mid=lag_lo + LL_BIN / 2, n_fixes=n_fix,
                         n_events=n_ev, rate=rate, rate_lo=lo, rate_hi=hi, rr=rate / base,
                         rr_lo=lo / base, rr_hi=hi / base))
    P = pd.DataFrame(rows).sort_values("lag_mid")
    P["baseline_rate"] = base
    return P


def first_spell_concentration(at: pd.DataFrame) -> pd.DataFrame:
    """Concentration with every visitor counted once: each visitor's first on-trail spell only,
    numerator and denominator restricted together."""
    ev = at[at.event_next == 1]
    per_person = ev.groupby("pid").size()
    same_cell = ev.groupby(["pid", "cell"]).size()
    rows = [dict(variant="repetition counts", k=np.nan, n_events=len(ev), person_hours=np.nan,
                 n_visitors=int(per_person.size), n_visitors_multi=int((per_person > 1).sum()),
                 n_same_cell_pairs=int((same_cell > 1).sum()))]
    fs = at[at.spell_id == at.groupby("pid")["spell_id"].transform("min")]
    for lab, d in [("all spells", at), ("first spell per visitor", fs)]:
        ph = d.groupby("cell").size() * C.FIX_S / 3600.0
        counts = d[d.event_next == 1].groupby("cell").size()
        for k in (5, 10):
            se, sh, r = C.count_concentration(counts, ph, k)
            rows.append(dict(variant=lab, k=k, n_events=int(d.event_next.sum()),
                             person_hours=ph.sum(), share_events=se, share_time=sh, conc_ratio=r))
    return pd.DataFrame(rows)


def mr(v, nd=1, scale=1.0):
    v = np.asarray(v, float) * scale
    return f"{np.median(v):.{nd}f} ({v.min():.{nd}f}-{v.max():.{nd}f})"


# ------------------------------------------------------------------------ figures ---
def fig1_site(d):
    import rasterio
    with rasterio.open(C.DEM_PATH) as src:
        dem = src.read(1).astype(float)
        dem[dem == src.nodata] = np.nan
        b = src.bounds
        extent = (b.left, b.right, b.bottom, b.top)
    z = np.nan_to_num(dem, nan=np.nanmin(dem))
    hs = LightSource(azdeg=315, altdeg=45).hillshade(z, vert_exag=1.6, dx=2, dy=2)

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.imshow(hs, extent=extent, origin="upper", cmap="gray", vmin=-0.35, vmax=1.35,
              interpolation="bilinear", zorder=0)
    ax.scatter(d["on"].x_m, d["on"].y_m, s=1.4, c=C.ACCENT2, alpha=0.18, lw=0,
               rasterized=True, zorder=2)
    ax.scatter(d["off"].x_m, d["off"].y_m, s=3.2, c=C.ACCENT, alpha=0.55, lw=0,
               rasterized=True, zorder=4)
    C.draw_trails(ax, d["trails"], lw=1.2)
    C.set_bbox(ax, d["bbox"]); C.map_axes(ax)
    C.scale_bar(ax, 500, loc=(0.03, 0.06)); C.north_arrow(ax)
    # legend goes in the empty relief above the central loop, clear of the trail
    ax.legend(handles=[
        Line2D([0], [0], color=C.INK, lw=1.4, label="Designated trail"),
        Line2D([0], [0], marker="o", ls="none", mfc=C.ACCENT2, mec="none", ms=5,
               alpha=0.7, label="GPS fix, on trail"),
        Line2D([0], [0], marker="o", ls="none", mfc=C.ACCENT, mec="none", ms=5,
               label="GPS fix, off trail"),
    ], loc="upper left", bbox_to_anchor=(0.27, 0.99), handletextpad=0.6, borderaxespad=0.2)
    fig.savefig(os.path.join(FIG, "fig1_site.png"))
    plt.close(fig)


def _rate_map(ax, d, colorbar=True):
    PI = d["PI"]
    ax.scatter(d["on"].x_m, d["on"].y_m, s=0.45, c=C.LIGHT, lw=0, rasterized=True, zorder=1)
    C.draw_trails(ax, d["trails"], lw=0.6, color="#B0B0B0")
    el = PI[(PI.person_hours >= 1.0) & (PI.events > 0)]
    r = el.rate_per_100ph.to_numpy()
    norm = LogNorm(vmin=max(r.min(), 5), vmax=r.max())
    cmap = plt.get_cmap(C.SEQ_CMAP)
    for row in el.itertuples():
        ax.add_patch(Rectangle((row.cx * CELL, row.cy * CELL), CELL, CELL,
                               facecolor=cmap(norm(max(row.rate_per_100ph, 5))),
                               edgecolor="white", lw=0.35, zorder=5))
    C.set_bbox(ax, d["bbox"]); C.map_axes(ax)
    C.scale_bar(ax, 500, loc=(0.03, 0.06))
    if colorbar:
        # inside the map's lower-right margin, so it stays attached to the map whatever
        # the surrounding gridspec cell does
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cax = ax.inset_axes([0.50, 0.045, 0.46, 0.038])
        cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("Off-trail transitions per 100 person-hours at risk (log scale)",
                     fontsize=6.8, labelpad=2)
        cb.ax.tick_params(labelsize=6.5, length=2, width=0.5, pad=1.5)
        cb.outline.set_linewidth(0.5)
    return norm, cmap


def _lorenz(ax, d):
    g = d["PI"].sort_values("rate_per_100ph", ascending=False)
    ce = np.concatenate([[0], np.cumsum(g.events) / g.events.sum()])
    ch = np.concatenate([[0], np.cumsum(g.person_hours) / g.person_hours.sum()])
    ax.plot([0, 1], [0, 1], ls=(0, (3, 2)), c=C.GREY, lw=0.9, zorder=1)
    ax.plot(ch, ce, c=C.ACCENT, lw=1.6, zorder=3)
    PC = d["PC"]; p5 = PC[(PC.variant == "primary 100 m") & (PC.k == 5)].iloc[0]
    ax.plot([p5.share_time], [p5.share_events], "o", mfc="white", mec=C.ACCENT, mew=1.4,
            ms=6, zorder=4)
    ax.annotate(f"Five cells with the most transitions:\n{100*p5.share_events:.0f}% of transitions on\n"
                f"{100*p5.share_time:.0f}% of at-risk time",
                xy=(p5.share_time, p5.share_events), xytext=(0.36, 0.52), fontsize=7.5,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C.GREY, shrinkB=4))
    ax.text(0.97, 0.90, "Equal to\ntime spent", fontsize=7, color=C.GREY, ha="right",
            va="top", transform=ax.transAxes)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("Cumulative share of at-risk person-time (%)")
    ax.set_ylabel("Cumulative share of off-trail transitions (%)")


def _severity(ax, d):
    SC = d["SC"]
    order = ["blip <=15 s", "short 16-60 s", "medium 61-300 s", "substantial >5 min"]
    labels = ["≤15 s", "16–60 s", "61–300 s", ">5 min"]
    sub = SC.set_index("dur_class").loc[order]
    allv = SC[SC.dur_class == "ALL"].conc_ratio.iloc[0]
    x = np.arange(4)
    # reference lines run the width of the bars; their labels sit in a gutter to the right
    ax.hlines([1.0, allv], -0.5, 3.5, colors=C.GREY, lw=0.7,
              linestyles=["-", (0, (2, 2))], zorder=1)
    ax.text(3.58, allv, f"All spells\n{allv:.1f}×", fontsize=6.8, color=C.GREY, ha="left",
            va="center")
    ax.text(3.58, 1.0, "Equal to\ntime spent", fontsize=6.8, color=C.GREY, ha="left",
            va="center")
    cols = [C.LIGHT, C.LIGHT, C.LIGHT, C.ACCENT]
    ax.bar(x, sub.conc_ratio, width=0.62, color=cols, edgecolor=C.INK, lw=0.5, zorder=3)
    for xi, (v, n) in enumerate(zip(sub.conc_ratio, sub.n)):
        ax.text(xi, v + 0.18, f"{v:.1f}×", ha="center", va="bottom", fontsize=8,
                fontweight="bold" if xi == 3 else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\nn = {int(n)}" for l, n in zip(labels, sub.n)], fontsize=7.5)
    ax.set_xlim(-0.5, 4.55); ax.set_ylim(0, 11.2)
    ax.set_xlabel("Duration of the off-trail spell")
    ax.set_ylabel("Concentration ratio, five cells\nwith the most spells of that class")


def _targeting(ax, d, inset=True):
    SR, TR = d["SR"], d["TR"]
    bench = TR[TR.check == "traffic benchmark"]
    # per-day, light
    ax.plot([0, 100], [0, 100], ls=(0, (3, 2)), c=C.GREY, lw=0.9, zorder=1)
    ax.text(84, 89, "Capture equal\nto coverage", fontsize=7, color=C.GREY, ha="right",
            va="bottom")
    ax.scatter(100 * SR.time_share, 100 * SR.capture, s=16, c=C.ACCENT, alpha=0.35, lw=0,
               zorder=3)
    ax.scatter(100 * bench.mean_time_share, 100 * bench.mean_capture, s=16, c=C.ACCENT2,
               alpha=0.35, lw=0, zorder=3)
    # means, bold
    ax.scatter([100 * SR.time_share.mean()], [100 * SR.capture.mean()], s=70, c=C.ACCENT,
               edgecolor="white", lw=1.0, zorder=5)
    ax.scatter([100 * bench.mean_time_share.mean()], [100 * bench.mean_capture.mean()], s=70,
               c=C.ACCENT2, edgecolor="white", lw=1.0, zorder=5)
    ax.annotate(f"Selected by excursion count:\n{100*SR.capture.mean():.0f}% captured,\n"
                f"{100*SR.time_share.mean():.0f}% of time covered",
                xy=(100 * SR.time_share.mean(), 100 * SR.capture.mean()),
                xytext=(17, 70), fontsize=7.3, color=C.ACCENT, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C.ACCENT, shrinkB=6))
    ax.annotate(f"Selected by visitor time:\n{100*bench.mean_capture.mean():.1f}% captured,\n"
                f"{100*bench.mean_time_share.mean():.0f}% of time covered",
                xy=(100 * bench.mean_time_share.mean(), 100 * bench.mean_capture.mean()),
                xytext=(46, 14), fontsize=7.3, color=C.ACCENT2, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C.ACCENT2, shrinkB=6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of held-out day's at-risk time covered (%)")
    ax.set_ylabel("Share of >5-min excursions captured (%)")
    if inset:
        _selection_inset(ax, d)


def _selection_inset(ax, d):
    """Mini map: the five cells each rule selects on the full data, showing the overlap."""
    PI = d["PI"]
    by_rate = set(zip(*PI.sort_values("events", ascending=False).head(5)[["cx", "cy"]].values.T))
    by_time = set(zip(*PI.sort_values("person_hours", ascending=False).head(5)[["cx", "cy"]].values.T))
    ins = ax.inset_axes([0.55, 0.40, 0.44, 0.40])
    ins.scatter(d["on"].x_m, d["on"].y_m, s=0.25, c=C.LIGHT, lw=0, rasterized=True, zorder=1)
    C.draw_trails(ins, d["trails"], lw=0.45, color="#B0B0B0")
    for (cx, cy) in by_rate | by_time:
        both = (cx, cy) in by_rate and (cx, cy) in by_time
        fc = C.ACCENT if (cx, cy) in by_rate else "none"
        ec = C.ACCENT2 if (cx, cy) in by_time else C.ACCENT
        ins.add_patch(Rectangle((cx * CELL, cy * CELL), CELL, CELL, facecolor=fc,
                                edgecolor=ec, lw=1.3 if (cx, cy) in by_time else 0.9,
                                alpha=0.9, hatch="////" if both else None, zorder=5))
    C.set_bbox(ins, d["bbox"]); C.map_axes(ins)
    ins.set_title("Five cells selected by each rule", fontsize=7, fontweight="normal",
                  loc="left", pad=2)
    ins.legend(handles=[Patch(facecolor=C.ACCENT, edgecolor=C.ACCENT, label="by transition count"),
                        Patch(facecolor="none", edgecolor=C.ACCENT2, lw=1.3,
                              label="by visitor time")],
               loc="upper left", bbox_to_anchor=(0.0, -0.02), fontsize=6.3, frameon=False,
               handlelength=1.2, borderaxespad=0.0, labelspacing=0.25, ncol=2,
               columnspacing=1.0)


def fig2_concentration(d):
    """Two rows with different column splits: the map (a) and the targeting panel (d) each
    take the wider slot so the site is legible and (d) has room for its inset."""
    fig = plt.figure(figsize=(7.2, 6.3))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.84, 1], hspace=0.38, left=0.07,
                             right=0.985, top=0.965, bottom=0.075)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.5, 1], wspace=0.28)
    bot = outer[1].subgridspec(1, 2, width_ratios=[1, 1.5], wspace=0.30)
    axA = fig.add_subplot(top[0, 0]); axB = fig.add_subplot(top[0, 1])
    axC = fig.add_subplot(bot[0, 0]); axD = fig.add_subplot(bot[0, 1])
    _rate_map(axA, d); C.panel_label(axA, "(a)", x=0.0, y=1.0)
    axA.set_anchor("N")
    _lorenz(axB, d); C.panel_label(axB, "(b)")
    _severity(axC, d); C.panel_label(axC, "(c)")
    _targeting(axD, d); C.panel_label(axD, "(d)")
    fig.savefig(os.path.join(FIG, "fig2_concentration.png"))
    plt.close(fig)


def fig3_leadlag(d):
    LL = d["LL"].sort_values("lag_mid")
    pre, post = LL[LL.lag_mid < 0], LL[LL.lag_mid > 0]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axvspan(-300, 0, color=C.PALE, lw=0, zorder=0)
    ax.axhline(1, c=C.GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.axvline(0, c=C.INK, lw=0.9, zorder=2)
    ax.fill_between(LL.lag_mid, LL.rr_lo, LL.rr_hi, color=C.ACCENT, alpha=0.16, lw=0, zorder=2)
    ax.plot(LL.lag_mid, LL.rr, c=C.ACCENT, lw=1.3, marker="o", ms=2.6, mec="none", zorder=3)
    for sub, x0, x1 in ((pre, -300, 0), (post, 0, 300)):
        m = sub.rr.mean()
        ax.hlines(m, x0, x1, colors=C.INK, lw=0.8, ls=(0, (1.5, 2)), zorder=3)
    ax.text(-150, pre.rr.mean() + 1.2, f"Mean before: {pre.rr.mean():.1f}×", ha="center",
            va="bottom", fontsize=7.5)
    ax.text(150, post.rr.mean() + 1.2, f"Mean after: {post.rr.mean():.1f}×", ha="center",
            va="bottom", fontsize=7.5)
    pk_pre = pre.loc[pre.rr.idxmax()]; pk_post = post.loc[post.rr.idxmax()]
    ax.annotate(f"{pk_pre.rr:.0f}×", xy=(pk_pre.lag_mid, pk_pre.rr),
                xytext=(pk_pre.lag_mid - 40, pk_pre.rr + 1.5), fontsize=8, ha="right",
                va="center", arrowprops=dict(arrowstyle="-", lw=0.6, color=C.INK, shrinkB=3))
    ax.annotate(f"{pk_post.rr:.0f}×", xy=(pk_post.lag_mid, pk_post.rr),
                xytext=(pk_post.lag_mid + 40, pk_post.rr + 1.5), fontsize=8, ha="left",
                va="center", arrowprops=dict(arrowstyle="-", lw=0.6, color=C.INK, shrinkB=3))
    ax.text(-292, ax.get_ylim()[1] * 0.985, "At-risk fix occurs BEFORE\nthe reference transition",
            fontsize=7.5, ha="left", va="top", color="#555555")
    ax.text(292, ax.get_ylim()[1] * 0.985, "At-risk fix occurs AFTER\nthe reference transition",
            fontsize=7.5, ha="right", va="top", color="#555555")
    ax.set_xlim(-300, 300)
    ax.set_xticks([-300, -200, -100, 0, 100, 200, 300])
    ax.set_xlabel("Time of at-risk fix relative to another visitor's off-trail onset (s)")
    ax.set_ylabel("Transition rate relative to\nsite-wide baseline (×)")
    ax.set_ylim(0, None)
    fig.savefig(os.path.join(FIG, "fig3_leadlag.png"))
    plt.close(fig)


def _envelope_heat(ax, E, title=None, highlight=(60, 50.0), cbar=True):
    wins = [15, 30, 60, 120]; rads = [25.0, 50.0, 100.0]
    M = np.array([[E[(E.window_s == w) & (E.radius_m == r)].pct.iloc[0] for r in rads]
                  for w in wins])
    CNT = np.array([[E[(E.window_s == w) & (E.radius_m == r)].C_compatible.iloc[0] for r in rads]
                  for w in wins])
    im = ax.imshow(M, cmap=C.HEAT_CMAP, vmin=0, vmax=60, aspect="auto")
    for i in range(4):
        for j in range(3):
            dark = M[i, j] > 38
            ax.text(j, i - 0.10, f"{M[i, j]:.1f}%", ha="center", va="center", fontsize=8.5,
                    fontweight="bold", color="white" if dark else C.INK)
            ax.text(j, i + 0.22, f"{int(CNT[i, j])} of 706", ha="center", va="center",
                    fontsize=6.5, color="white" if dark else "#555555")
    hi, hj = wins.index(highlight[0]), rads.index(highlight[1])
    ax.add_patch(Rectangle((hj - 0.5, hi - 0.5), 1, 1, fill=False, edgecolor=C.INK, lw=1.8,
                           zorder=5))
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{int(r)} m" for r in rads])
    ax.set_yticks(range(4)); ax.set_yticklabels([f"{w} s" for w in wins])
    ax.set_xlabel("Radius around the later spell"); ax.set_ylabel("Preceding time window")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    if title:
        ax.set_title(title, fontsize=8.5, fontweight="normal", pad=6)
    return im


def fig4_envelope(d):
    E = d["EN"][d["EN"].los == "on"]
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    im = _envelope_heat(ax, E)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cb.set_label("Spells with a qualifying earlier onset (%)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7, length=2, width=0.5); cb.outline.set_linewidth(0.5)
    fig.savefig(os.path.join(FIG, "fig4_envelope.png"))
    plt.close(fig)


# ------------------------------------------------------------------------------ main ---
def main():
    C.apply_figure_style()
    at, off, pings = C.load_processed()
    n_ev, ph = int(at.event_next.sum()), len(at) * C.FIX_S / 3600
    site = 100 * n_ev / ph
    sev = off[off.dur_s > C.SEVERE_S]
    onsets = off.reset_index(drop=True)

    C.hr("CONCENTRATION")
    PI = C.cell_table(at)
    PI["rate_eligible"] = PI["person_hours"] >= 1.0
    PC = C.concentration_variants(at)
    SC = C.severity_concentration(off, at)
    UP = first_spell_concentration(at)
    PI.to_csv(os.path.join(TAB, "result_cells_100m.csv"), index=False)
    PC.to_csv(os.path.join(TAB, "result_concentration_variants.csv"), index=False)
    SC.to_csv(os.path.join(TAB, "result_duration_classes.csv"), index=False)
    UP.to_csv(os.path.join(TAB, "result_first_spell.csv"), index=False)
    prim = {k: PC[(PC.variant == "primary 100 m") & (PC.k == k)].iloc[0] for k in (5, 10, 20)}
    print(f"  five cells: {100*prim[5].share_events:.1f}% of transitions on "
          f"{100*prim[5].share_time:.1f}% of at-risk time = {prim[5].conc_ratio:.2f}; "
          f"Hoover {prim[5].hoover:.3f}")

    C.hr("HELD-OUT TARGETING OF SUSTAINED EXCURSIONS")
    per_day, summ = C.heldout_targeting(at, sev)
    per_day.to_csv(os.path.join(TAB, "result_heldout_per_day.csv"), index=False)
    summ.to_csv(os.path.join(TAB, "result_heldout_summary.csv"), index=False)
    print(summ.round(3).to_string(index=False))

    C.hr("COMPATIBILITY ENVELOPE")
    relief = C.ReliefCache(C.DEM())
    EN = C.envelope(onsets, relief)
    EN.to_csv(os.path.join(TAB, "table4_envelope.csv"), index=False)
    e = EN[(EN.los == "on") & (EN.window_s == 60) & (EN.radius_m == 50.0)].iloc[0]
    print(f"  primary (60 s, 50 m, line of sight): {int(e.C_compatible)} of {len(onsets)} "
          f"({e.pct:.1f}%)")

    C.hr("LEAD-LAG PROFILE")
    LL = leadlag_profile(at, onsets)
    LL.to_csv(os.path.join(TAB, "result_leadlag_profile.csv"), index=False)
    pk = LL.loc[LL.rr.idxmax()]
    print(f"  peak {pk.rr:.1f}x at {pk.lag_lo:+.0f} s; mean before "
          f"{LL[LL.lag_mid < 0].rr.mean():.2f}x, after {LL[LL.lag_mid > 0].rr.mean():.2f}x")

    C.hr("TABLES 1-3")
    t1 = [("Visitors tracked", pings.pid.nunique()), ("Ship-days", pings.shipday.nunique()),
          ("GPS fixes (5 s cadence)", len(pings)), ("At-risk person-intervals", len(at)),
          ("At-risk person-hours", round(ph, 1)), ("On-trail spells", at.spell_id.nunique()),
          ("Off-trail transitions", n_ev), ("Off-trail spells", len(off)),
          ("Off-trail spells lasting more than 5 min", len(sev)),
          ("Site-wide transition rate per 100 person-hours", round(site, 1)),
          ("Site-wide transition rate per fix (%)", round(100 * n_ev / len(at), 3)),
          ("100 m cells with any at-risk time", len(PI)),
          ("100 m cells with at least one transition", int((PI.events > 0).sum()))]
    pd.DataFrame(t1, columns=["quantity", "value"]).to_csv(
        os.path.join(TAB, "table1_design_sample.csv"), index=False)

    t2 = []
    for k, lab in ((5, "Five cells with the most transitions"), (10, "Ten cells"),
                   (20, "Twenty cells")):
        p = prim[k]
        t2.append(("All transitions, 100 m cells", lab, f"{100*p.share_events:.1f}",
                   f"{100*p.share_time:.1f}", f"{p.conc_ratio:.2f}"))
    k5 = PC[PC.k == 5]
    for v, lab in (("shifted origin", "Sixteen shifted grid origins"),
                   ("resolution", "Cell sizes of 50, 100 and 200 m"),
                   ("ship-day", "Each of the 11 ship-days on its own"),
                   ("min person-hours", "Minimum at-risk time per cell of 0, 1, 5 or 10 h")):
        s = k5[k5.variant == v]
        t2.append(("Robustness: median (range)", lab, mr(s.share_events, 1, 100),
                   mr(s.share_time, 1, 100), mr(s.conc_ratio, 2)))
    for v, lab in (("all spells", "All on-trail spells"),
                   ("first spell per visitor", "First on-trail spell per visitor only")):
        u = UP[(UP.variant == v) & (UP.k == 5)].iloc[0]
        t2.append(("Each visitor counted once", f"{lab} ({int(u.n_events)} transitions, "
                   f"{u.person_hours:.0f} person-hours)", f"{100*u.share_events:.1f}",
                   f"{100*u.share_time:.1f}", f"{u.conc_ratio:.2f}"))
    for key, lab in (("blip <=15 s", "15 s or less"), ("short 16-60 s", "16-60 s"),
                     ("medium 61-300 s", "61-300 s"), ("substantial >5 min", "More than 5 min")):
        r = SC[SC.dur_class == key].iloc[0]
        t2.append(("By duration of the off-trail spell (cells selected within class)",
                   f"{lab} ({int(r.n)} spells; {100*r.pct_of_off_time:.1f}% of off-trail time)",
                   f"{100*r.share_events:.1f}", f"{100*r.share_time:.1f}", f"{r.conc_ratio:.2f}"))
    pd.DataFrame(t2, columns=["block", "row", "events_in_five_cells_pct",
                              "at_risk_time_in_five_cells_pct", "concentration_ratio"]).to_csv(
        os.path.join(TAB, "table2_concentration.csv"), index=False)

    t3 = []
    for r in summ.itertuples():
        lab = {"k sensitivity": f"Most excursions on training days, k = {r.k}",
               "nested": f"Nested selection (k = {r.k} chosen on all days)" ,
               "traffic benchmark": "Benchmark: five cells with the most at-risk time"}[r.check]
        t3.append((lab, f"{100*r.mean_capture:.1f}", f"{100*r.mean_time_share:.1f}",
                   f"{r.mean_lift:.2f}", f"{r.days_lift_gt1} of {r.n_days}"))
    pd.DataFrame(t3, columns=["selection_rule", "captured_pct", "covered_pct", "lift",
                              "days_with_lift_gt_1"]).to_csv(
        os.path.join(TAB, "table3_heldout_targeting.csv"), index=False)

    C.hr("FIGURES 1-4")
    prim_day = per_day[per_day.rule == "primary"].rename(columns={"lift": "ratio"})
    TR = pd.concat([
        summ[summ.check == "k sensitivity"][["check", "k", "mean_capture", "mean_time_share",
                                             "mean_lift"]],
        per_day[per_day.rule != "primary"].rename(columns={
            "rule": "check", "capture": "mean_capture", "time_share": "mean_time_share",
            "lift": "mean_lift"})[["check", "k", "mean_capture", "mean_time_share",
                                   "mean_lift", "shipday"]]], ignore_index=True)
    d = dict(PI=PI, PC=PC, SC=SC, SR=prim_day[["shipday", "n_test", "capture", "time_share",
                                               "ratio"]], TR=TR, LL=LL, EN=EN,
             **C.map_layers(pings))
    for f in (fig1_site, fig2_concentration, fig3_leadlag, fig4_envelope):
        f(d)
        print(f"  wrote {f.__name__}.png")


if __name__ == "__main__":
    main()
