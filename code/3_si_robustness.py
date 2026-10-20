"""
Step 3 - Supporting Information robustness checks.

Reads data/processed/ and the result files written by 2_main_analysis.py (run that first).

Concentration
  tableS1_cells.csv                  every 100 m cell holding at least one transition
  tableS2_robustness.csv             five-cell statistic under every grid variant   (Fig S1)
  tableS3_heldout_per_day.csv        held-out targeting, one row per ship-day       (Fig S2)
  figS3_k_sensitivity.png            capture and lift as the number of cells varies
  tableS_trail_definition.csv        15/25/35 m buffer x informal paths off/on trail
  tableS_pledge_arm.csv              concentration within each arm of the original trial
  tableS_duration_fixed_cells.csv    duration classes in the five all-transition cells
Temporal
  tableS4_bursts.csv                 space-time components across linkage thresholds (Fig S4)
  tableS5_leadlag.csv                the lead-lag profile in full                   (Figure 3)
  figS5_envelope_los.png             compatibility envelope with and without line of sight
  tableS_state_cue.csv               transitions with another visitor already off the trail
                                     nearby (the cue the onset-based envelope does not count)

Run:  python code/3_si_robustness.py        (~10 min; the trail-buffer check rebuilds the
                                             risk set six times)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import common as C

FIG, TAB = C.FIG, C.TAB
PRIMARY_TOP_CELL = "4950_76766"


# ---------------------------------------------------------------------- computations ---
def trail_definition_sensitivity(pings: pd.DataFrame) -> pd.DataFrame:
    """Re-classify every fix at 15, 25 and 35 m around the designated network, with informal
    paths counted as off-trail (primary) and as trail, rebuild the risk set and the off-trail
    spells, and recompute the five-cell statistics."""
    lines = C.load_trail_lines()
    rows = []
    for label, cats in (("designated only", C.DESIGNATED),
                        ("designated + informal", list(C.CATEGORY_FILES))):
        for width in (15.0, 25.0, 35.0):
            p = pings[["pid", "shipday", "group", "treatment", "timestamp", "x_m", "y_m"]].copy()
            p["off_trail"] = C.classify(p.x_m.to_numpy(), p.y_m.to_numpy(),
                                        C.build_buffer(lines, cats, width))
            risk = C.build_risk_set(p)
            risk["cell"] = C.cell_of(risk.x_m, risk.y_m)
            ev = risk.groupby("cell")["event_next"].sum().astype(float)
            ph = risk.groupby("cell").size() * C.FIX_S / 3600.0
            se, sh, ratio = C.count_concentration(ev, ph)
            g = pd.DataFrame({"events": ev, "person_hours": ph})
            g["rate_per_100ph"] = 100 * g.events / g.person_hours
            site = ev.sum() / ph.sum()
            top = ev.sort_values(ascending=False).index[0]
            off = C.build_off_spells(p)
            sev = off[off.dur_s > C.SEVERE_S]
            se5, sh5, r5 = C.count_concentration(sev.groupby("cell").size().astype(float), ph)
            rows.append(dict(trails=label, buffer_m=width, off_trail_pct=100 * p.off_trail.mean(),
                             transitions=int(ev.sum()), person_hours=ph.sum(),
                             top5_share_events=se, top5_share_time=sh, top5_ratio=ratio,
                             hoover=C.hoover(g), top_cell=top,
                             top_cell_rate_ratio=(ev[top] / ph[top]) / site,
                             primary_top_cell_has_at_risk_time=PRIMARY_TOP_CELL in ph.index,
                             spells_gt5min=len(sev), sev_top5_share_events=se5,
                             sev_top5_share_time=sh5, sev_top5_ratio=r5))
            print(f"    {label:<22} {width:>3.0f} m | transitions {int(ev.sum()):>5} | "
                  f"five cells {100*se:5.1f}% on {100*sh:5.1f}% = {ratio:.2f} | >5 min "
                  f"{100*se5:5.1f}% on {100*sh5:5.1f}% = {r5:.2f}", flush=True)
    return pd.DataFrame(rows)


def by_arm(at: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lab, g in (("pooled", at), ("no pledge", at[at.treatment == 0]),
                   ("pledge", at[at.treatment == 1])):
        ev = g.groupby("cell")["event_next"].sum().astype(float)
        ph = g.groupby("cell").size() * C.FIX_S / 3600.0
        se, sh, r = C.count_concentration(ev, ph)
        rows.append(dict(arm=lab, visitors=g.pid.nunique(), transitions=int(ev.sum()),
                         person_hours=ph.sum(), share_events=se, share_time=sh, conc_ratio=r,
                         cells=";".join(ev.sort_values(ascending=False).head(C.TOPK).index)))
    out = pd.DataFrame(rows)
    shared = len(set(out.cells[1].split(";")) & set(out.cells[2].split(";")))
    out["cells_shared_between_arms"] = [np.nan, shared, shared]
    return out


def duration_fixed_cells(off: pd.DataFrame, PI: pd.DataFrame) -> pd.DataFrame:
    """Share of each duration class in the same five cells: those with the most transitions."""
    top = PI.head(C.TOPK)
    cells = set(top.cx.astype(str) + "_" + top.cy.astype(str))
    share_time = top.person_hours.sum() / PI.person_hours.sum()
    cls = pd.cut(off["dur_s"], [b[0] for b in C.DUR_CLASSES] + [np.inf],
                 labels=[b[2] for b in C.DUR_CLASSES], right=True, include_lowest=True)
    rows = []
    for lab in [b[2] for b in C.DUR_CLASSES] + ["ALL"]:
        sub = off if lab == "ALL" else off[cls == lab]
        s = sub.cell.isin(cells).mean()
        rows.append(dict(dur_class=lab, n=len(sub), share_in_five_cells=s,
                         share_time_five_cells=share_time, conc_ratio=s / share_time))
    return pd.DataFrame(rows)


def state_cue(at: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, lab in (("exp_state", "within 50 m, previous 60 s, terrain line of sight"),
                     ("exp_state_distonly", "within 50 m, previous 60 s, distance and time only"),
                     ("exp_state120", "within 50 m, previous 120 s, terrain line of sight")):
        rows.append(dict(definition=lab,
                         pct_of_transitions=100 * at.loc[at.event_next == 1, col].mean(),
                         pct_of_other_fixes=100 * at.loc[at.event_next == 0, col].mean()))
    return pd.DataFrame(rows)


def burst_grid(onsets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in (25.0, 50.0, 100.0):
        for w in (60.0, 120.0, 180.0):
            d = onsets.copy()
            d["comp"] = C.components(d, r, w)
            g = d.groupby("comp").agg(size=("pid", "size"), t0=("t_s", "min"), t1=("t_s", "max"))
            g["duration_s"] = g.t1 - g.t0
            m = g[g["size"] > 1]
            rows.append(dict(radius_m=r, window_s=w, n_onsets=len(d), n_components=len(g),
                             n_singletons=int((g["size"] == 1).sum()),
                             pct_singleton=round(100 * (g["size"] == 1).mean(), 1),
                             n_multi=len(m), pct_onsets_in_multi=round(100 * m["size"].sum() / len(d), 1),
                             max_size=int(g["size"].max()), median_multi_size=float(m["size"].median()),
                             median_multi_duration_s=float(m["duration_s"].median())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ figures ---
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


def figS1_robustness(d):
    PC = d["PC"]; R = PC[PC.k == 5]
    groups = [("shifted origin", "Grid origin\n(16 offsets)"),
              ("resolution", "Cell size\n(50 / 100 / 200 m)"),
              ("ship-day", "Ship-day\n(11 days)"),
              ("min person-hours", "Minimum at-risk\ntime rule")]
    prim = R[R.variant == "primary 100 m"].conc_ratio.iloc[0]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    rs = np.random.RandomState(0)
    for i, (v, lab) in enumerate(groups):
        vals = R[R.variant == v].conc_ratio.dropna().to_numpy()
        ax.scatter(np.full(len(vals), i) + rs.uniform(-.10, .10, len(vals)), vals, s=18,
                   c=C.ACCENT2, alpha=0.7, lw=0, zorder=3)
        ax.hlines(np.median(vals), i - .22, i + .22, color=C.INK, lw=1.6, zorder=4)
    ax.axhline(prim, ls=(0, (2, 2)), c=C.GREY, lw=0.8)
    ax.axhline(1, c=C.GREY, lw=0.8)
    ax.text(3.45, prim + 0.15, f"Primary {prim:.2f}×", fontsize=7, color=C.GREY, ha="right")
    ax.text(3.45, 1.15, "Proportional to time spent", fontsize=7, color=C.GREY, ha="right")
    ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in groups], fontsize=7.5)
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(0, None)
    ax.set_ylabel("Top-5 concentration ratio")
    fig.savefig(os.path.join(FIG, "figS1_robustness.png"))
    plt.close(fig)


def figS2_heldout_days(d):
    SR = d["SR"].copy(); TR = d["TR"]
    bench = TR[TR.check == "traffic benchmark"].set_index("shipday")
    SR["day"] = SR.shipday.str.replace("_", " ")
    y = np.arange(len(SR))[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for yi, r in zip(y, SR.itertuples()):
        b = bench.loc[r.shipday]
        ax.plot([100 * b.mean_capture, 100 * r.capture], [yi, yi], c=C.LIGHT, lw=1.2, zorder=1)
        ax.scatter(100 * r.capture, yi, s=34, c=C.ACCENT, zorder=3, lw=0)
        ax.scatter(100 * b.mean_capture, yi, s=34, c=C.ACCENT2, zorder=3, lw=0)
        ax.text(101.5, yi, f"n = {int(r.n_test)}", fontsize=7, va="center", color="#555555")
    ax.set_yticks(y); ax.set_yticklabels(SR.day, fontsize=7.5)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of that day's >5-min excursions captured by five cells\n"
                  "chosen on the other ten days (%)")
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="none", mfc=C.ACCENT, mec="none",
                              label="Selected by excursion count"),
                       Line2D([0], [0], marker="o", ls="none", mfc=C.ACCENT2, mec="none",
                              label="Selected by visitor time")],
              loc="upper center", bbox_to_anchor=(0.50, 1.0), ncol=1)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.savefig(os.path.join(FIG, "figS2_heldout_days.png"))
    plt.close(fig)


def figS3_k_sensitivity(d):
    TR = d["TR"]; K = TR[TR.check == "k sensitivity"].sort_values("k")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.7))
    a1.plot(K.k, 100 * K.mean_capture, "-o", c=C.ACCENT, ms=4, mec="none")
    a1.plot(K.k, 100 * K.mean_time_share, "-o", c=C.GREY, ms=4, mec="none")
    a1.set_xscale("log"); a1.set_xticks(K.k); a1.set_xticklabels([int(k) for k in K.k])
    a1.set_ylim(0, 100); a1.set_xlabel("Number of cells selected, k")
    a1.set_ylabel("Mean over held-out days (%)")
    a1.text(K.k.iloc[-1], 100 * K.mean_capture.iloc[-1] + 3, "Excursions captured",
            fontsize=7, color=C.ACCENT, ha="right")
    a1.text(K.k.iloc[-1], 100 * K.mean_time_share.iloc[-1] + 3, "At-risk time covered",
            fontsize=7, color=C.GREY, ha="right")
    a2.plot(K.k, K.mean_lift, "-o", c=C.INK, ms=4, mec="none")
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xticks(K.k); a2.set_xticklabels([int(k) for k in K.k])
    a2.set_xlabel("Number of cells selected, k"); a2.set_ylabel("Lift (captured ÷ covered)")
    a2.axhline(1, c=C.GREY, lw=0.8)
    C.panel_label(a1, "(a)"); C.panel_label(a2, "(b)")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(FIG, "figS3_k_sensitivity.png"))
    plt.close(fig)


def figS4_bursts(d):
    src = d["onsets"]
    lab = C.components(src, 50.0, 120.0)
    sizes = pd.Series(lab).value_counts().to_numpy()
    BC = d["BC"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15]))
    vals, counts = np.unique(sizes, return_counts=True)
    a1.bar(vals, counts, color=C.ACCENT2, width=0.8, lw=0)
    a1.set_xticks([1, 5, 10, 15])
    a1.set_xlabel("Onsets per space-time component"); a1.set_ylabel("Number of components")
    a1.text(0.97, 0.95, f"{int((sizes == 1).sum())} single-onset components\n"
                        f"{100 * sizes[sizes > 1].sum() / sizes.sum():.0f}% of onsets in "
                        f"multi-onset components", transform=a1.transAxes, ha="right", va="top",
            fontsize=7)
    # linkage sensitivity: % of onsets in multi-onset components, r x w
    piv = BC.pivot(index="window_s", columns="radius_m", values="pct_onsets_in_multi")
    im = a2.imshow(piv.values, cmap=C.HEAT_CMAP, vmin=50, vmax=90, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            a2.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if v > 76 else C.INK)
    a2.set_xticks(range(piv.shape[1])); a2.set_xticklabels([f"{int(c)} m" for c in piv.columns])
    a2.set_yticks(range(piv.shape[0])); a2.set_yticklabels([f"{int(i)} s" for i in piv.index])
    a2.set_xlabel("Linkage radius"); a2.set_ylabel("Linkage window")
    a2.set_title("Onsets in multi-onset components", fontsize=8, fontweight="normal", pad=5)
    for sp in a2.spines.values():
        sp.set_visible(False)
    a2.tick_params(length=0)
    a2.add_patch(Rectangle((list(piv.columns).index(50.0) - 0.5, list(piv.index).index(120.0) - 0.5),
                           1, 1, fill=False, edgecolor=C.INK, lw=1.6))
    C.panel_label(a1, "(a)"); C.panel_label(a2, "(b)")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(FIG, "figS4_bursts.png"))
    plt.close(fig)


def figS5_envelope_los(d):
    EN = d["EN"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.4))
    _envelope_heat(a1, EN[EN.los == "on"], title="Terrain line of sight required")
    im = _envelope_heat(a2, EN[EN.los == "off"], title="Distance and time only")
    a2.set_ylabel("")
    cb = fig.colorbar(im, ax=[a1, a2], fraction=0.04, pad=0.03)
    cb.set_label("Spells with a qualifying earlier onset (%)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7, length=2, width=0.5); cb.outline.set_linewidth(0.5)
    C.panel_label(a1, "(a)", x=-0.12); C.panel_label(a2, "(b)", x=-0.06)
    fig.savefig(os.path.join(FIG, "figS5_envelope_los.png"))
    plt.close(fig)


# ------------------------------------------------------------------------------ main ---
def main():
    C.apply_figure_style()
    at, off, pings = C.load_processed()
    onsets = off.reset_index(drop=True)
    rd = lambda n: pd.read_csv(os.path.join(TAB, n))
    PI, PC = rd("result_cells_100m.csv"), rd("result_concentration_variants.csv")
    per_day, summ = rd("result_heldout_per_day.csv"), rd("result_heldout_summary.csv")
    LL, EN = rd("result_leadlag_profile.csv"), rd("table4_envelope.csv")

    C.hr("CONCENTRATION")
    PI[PI.events > 0].to_csv(os.path.join(TAB, "tableS1_cells.csv"), index=False)
    PC[PC.k == 5].to_csv(os.path.join(TAB, "tableS2_robustness.csv"), index=False)
    pv = per_day.pivot(index="shipday", columns="rule",
                       values=["n_test", "k", "capture", "time_share", "lift"])
    pv.columns = [f"{rule}_{v}" for v, rule in pv.columns]
    pv.reset_index().to_csv(os.path.join(TAB, "tableS3_heldout_per_day.csv"), index=False)
    by_arm(at).to_csv(os.path.join(TAB, "tableS_pledge_arm.csv"), index=False)
    duration_fixed_cells(off, PI).to_csv(os.path.join(TAB, "tableS_duration_fixed_cells.csv"),
                                         index=False)
    print("  trail definition and buffer:")
    trail_definition_sensitivity(pings).to_csv(
        os.path.join(TAB, "tableS_trail_definition.csv"), index=False)

    C.hr("TEMPORAL")
    BC = burst_grid(onsets)
    BC.to_csv(os.path.join(TAB, "tableS4_bursts.csv"), index=False)
    LL.to_csv(os.path.join(TAB, "tableS5_leadlag.csv"), index=False)
    sc = state_cue(at)
    sc.to_csv(os.path.join(TAB, "tableS_state_cue.csv"), index=False)
    print(sc.round(1).to_string(index=False))

    C.hr("SI FIGURES")
    TR = pd.concat([
        summ[summ.check == "k sensitivity"][["check", "k", "mean_capture", "mean_time_share",
                                             "mean_lift"]],
        per_day[per_day.rule != "primary"].rename(columns={
            "rule": "check", "capture": "mean_capture", "time_share": "mean_time_share",
            "lift": "mean_lift"})[["check", "k", "mean_capture", "mean_time_share",
                                   "mean_lift", "shipday"]]], ignore_index=True)
    SR = per_day[per_day.rule == "primary"].rename(columns={"lift": "ratio"})
    d = dict(PC=PC, TR=TR, SR=SR[["shipday", "n_test", "capture", "time_share", "ratio"]],
             BC=BC, EN=EN, onsets=onsets)
    for f in (figS1_robustness, figS2_heldout_days, figS3_k_sensitivity, figS4_bursts,
              figS5_envelope_los):
        f(d)
        print(f"  wrote {f.__name__}.png")


if __name__ == "__main__":
    main()
