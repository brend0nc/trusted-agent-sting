"""Draw the per-question figures for the results site from the derived JSON data.

One SVG per DESIGN.md question (fig/q1.svg .. fig/q8.svg), driven entirely by what
summary.json / records.json actually contain, plus three candidate-metric figures
(fig/m-coalition.svg, fig/m-nonrecip.svg, fig/m-chat.svg) and one gallery thumbnail
per experiments.json entry (fig/exp-<id>.svg).

Figure grammar (SITE.md interaction-design item 4, rev. 2026-07-03b, user-locked):
BAR and LINE charts only — bars for comparisons (95% CI error bars wherever a CI
exists), lines for ordered axes. All figure text typeset by the system LaTeX
(usetex, sans-serif helvet setup) at 13-15px equivalent, big marks, minimal spines,
no gridlines, plain-English axis labels, at most one short on-chart annotation. The finding title, how-to-read sentence and fine
print do NOT live in the SVG: they are exported to data/fig-meta.json
({id: {finding, howto, fineprint}}) and typeset by render.py in the page HTML.

Semantic color: oxblood #8b1a1a = the measured effect / leader; warm grey #c7c2ba =
context bars; #555 ink error bars; dashed grey = chance / tit-for-tat baselines.
Unmeasured figures render the empty axes of the chart type they will become, with
the config that fills them named in grey.

Evidence deep links (item 1): marks that map to exactly one episode carry gid
"ep:<episode-id>" (thumbnail bars); pooled bars carry no gid. write_svg injects a
native <title> child per episode group as the no-JS hover fallback.

Determinism: fixed svg.hashsalt, seeded bootstraps, metadata Date stripped — the
build is byte-identical across runs.

Usage:
  .venv/bin/python site/figures.py [--data-dir site/data] [--out site/dist/fig]
"""
import argparse
import html
import io
import json
import os
import re
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # for models.ladders() (q4) — same trick as derive.py

# usetex shells out to latex/kpsewhich (MacTeX); some build envs lack it on PATH.
if "/Library/TeX/texbin" not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + "/Library/TeX/texbin"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter

OXBLOOD = "#8b1a1a"    # the measured effect / the leader
WARMGREY = "#c7c2ba"   # context bars (non-leaders, chance bars)
INK = "#555555"        # error bars
LINEGREY = "#999999"   # zero / chance reference lines
TEXTGREY = "#666666"   # axis labels + the one on-chart annotation
TICKFADE = "#b5b0a8"   # tick labels for unmeasured rungs
DASH = (0, (4, 3))
FIGSIZE = (9.2, 4.4)
THUMBSIZE = (4.2, 2.6)
ERR = dict(ecolor=INK, lw=1.6, capsize=4)          # bar xerr/yerr style
FS_TICK = 13
FS_NAME = 15           # category tick labels (model names)
FS_LABEL = 13          # axis labels (grey)
FS_NOTE = 12.5         # the one on-chart annotation (grey)
CI_NOTE = "95% episode-bootstrap CI, 2000 trials"
TOPO_PLAIN = {"complete": "everyone can message everyone",
              "ring": "messages pass around a ring"}


def init_style():
    """LaTeX-typeset sans styling shared by every figure (SITE.md item 4)."""
    matplotlib.rcParams.update({
        "figure.figsize": FIGSIZE,
        "font.family": "sans-serif",
        "font.size": 14,
        "text.usetex": shutil.which("latex") is not None,
        "text.latex.preamble": "\n".join([
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{helvet}",              # Helvetica clone (phv)
            r"\renewcommand{\familydefault}{\sfdefault}",
            r"\usepackage{sansmath}",            # sans math too (sfmath not in this BasicTeX)
            r"\sansmath",
        ]),
        "axes.labelsize": FS_LABEL,
        "axes.labelcolor": TEXTGREY,
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.unicode_minus": False,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "text.color": "#1c1c1c",
        "svg.fonttype": "none",                  # moot under usetex: text ships as glyph paths
        "svg.hashsalt": "delegated-influence",   # deterministic ids -> stable diffs
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


# --- shared helpers ---------------------------------------------------------------------------------

class Unmeasured(Exception):
    """The JSON lacks the stats this figure needs — render the placeholder instead."""


def esc(s):
    """Escape LaTeX specials in strings headed for the figure (usetex renders them
    all). Meta strings (finding/howto/fineprint) are typeset as HTML — never escape
    those."""
    if not matplotlib.rcParams.get("text.usetex"):
        return str(s)
    replacements = {
        "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$",
        "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_",
        "^": r"\textasciicircum{}", "~": r"\textasciitilde{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(s))


def letterspaced(text):
    """Readable tracking with and without LaTeX; never expose TeX hard-space markers."""
    separator = "~" if matplotlib.rcParams.get("text.usetex") else "\u2009"
    return separator.join(text)


def xerr_of(vals, cis):
    """(lo, hi) half-width arrays for bar xerr/yerr from value + CI lists (None CI -> 0)."""
    lo = [v - ci[0] if ci else 0.0 for v, ci in zip(vals, cis)]
    hi = [ci[1] - v if ci else 0.0 for v, ci in zip(vals, cis)]
    return (lo, hi)


def config_for(needs, summary):
    """The coverage row whose arm name appears in the needs string -> its config name."""
    for row in summary.get("coverage") or []:
        if row.get("arm") and re.search(rf"\b{re.escape(row['arm'])}\b", needs or ""):
            return row.get("config")
    return None


def config_line(ax, cfg, needs=None):
    """The grey line naming the config that fills an unmeasured figure, centered."""
    text = (f"runs from configs/{cfg}.yaml will fill this figure" if cfg
            else f"needs: {needs}")
    ax.text(0.5, 0.5, esc(text), transform=ax.transAxes, ha="center", va="center",
            fontsize=FS_NOTE, color=LINEGREY)
    return text


def note(ax, text, xy, dxy=(0, 0), ha="left", va="center", coords="data"):
    """The single short on-chart annotation, grey, no connector."""
    ax.annotate(text, xy, xytext=dxy, textcoords="offset points",
                xycoords=coords, ha=ha, va=va, fontsize=FS_NOTE, color=TEXTGREY)


def placeholder(needs, summary):
    """Generic fallback (build failures / vanished stats): no axes, centered
    letterspaced NOT YET MEASURED over a hairline rule, grey config line."""
    fig = plt.figure(figsize=FIGSIZE)
    # "~" = LaTeX hard space: keeps the letterspacing (runs of plain spaces collapse)
    fig.text(0.5, 0.56, letterspaced("NOT YET MEASURED"),
             ha="center", va="center", fontsize=14, color="#1c1c1c")
    fig.add_artist(matplotlib.lines.Line2D([0.46, 0.54], [0.49, 0.49],
                                           color="#dddddd", lw=0.8,
                                           transform=fig.transFigure))
    cfg = config_for(needs, summary)
    line = (f"runs from configs/{cfg}.yaml will fill this figure" if cfg
            else f"needs: {needs}")
    fig.text(0.5, 0.42, esc(line), ha="center", va="center", fontsize=FS_NOTE,
             color=LINEGREY)
    return fig


def _short(model):
    return model.split("/")[-1]


def _has(stat):
    return isinstance(stat, dict) and stat.get("value") is not None


# --- the eight questions (every builder returns (fig, meta)) ----------------------------------------

def fig_q1(summary, records):
    """Q1 — does talk pay? Two panels, every number from the tested pipeline. LEFT (the
    flagship): per attacker model, the capture the message channel ADDS — capture with
    messages on minus the same game with messages off (report.floor_contrast) — as a
    horizontal bar with a 95% CI whisker against a zero line; the two models that survive
    Benjamini-Hochberg correction are solid oxblood, the fragile one (does not survive) is
    hatched. RIGHT: the cooperative backdrop the gain sits on — how often a received favour
    is part of a reciprocal pair (payback) against its tested random-retargeting chance
    floor, drawn as a shaded band."""
    q1 = summary["questions"].get("q1") or {}
    fc = q1.get("floor_contrast") or {}
    rows = [r for r in fc.values() if r.get("contrast") is not None and r.get("ci")]
    if not rows:
        raise Unmeasured("wedge floor-contrast (q1.floor_contrast)")
    rows = sorted(rows, key=lambda r: r["contrast"])     # strongest gain ends up on top
    pf = q1.get("payback_floor") or {}
    have_pf = _has(pf.get("observed")) and _has(pf.get("floor"))

    if have_pf:
        fig, (ax, axp) = plt.subplots(1, 2, constrained_layout=True,
                                      gridspec_kw={"width_ratios": [2.3, 1]})
    else:
        fig, ax = plt.subplots(constrained_layout=True)
        axp = None

    for i, r in enumerate(rows):                         # LEFT — talk gain per model
        solid = bool(r.get("survives"))
        ax.barh(i, r["contrast"], height=0.6,
                color=OXBLOOD if solid else "white", edgecolor=OXBLOOD,
                linewidth=0 if solid else 1.7, hatch=None if solid else "////",
                xerr=xerr_of([r["contrast"]], [r["ci"]]), error_kw=ERR)
    ax.axvline(0, color=LINEGREY, lw=1.2, ls=DASH)
    ax.set_yticks(range(len(rows)), [esc(r["model"]) for r in rows], fontsize=FS_NAME)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    lo = min(r["ci"][0] for r in rows)
    hi = max(r["ci"][1] for r in rows)
    ax.set_xlim(min(lo, 0) - 0.006, max(hi, 0) + 0.008)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlabel("capture the message channel adds\n(with messages minus without)")
    note(ax, "0 = talk adds nothing", (0, -0.55), (6, 0), va="center")

    if axp is not None:                                  # RIGHT — payback vs chance floor
        obs, flo = pf["observed"], pf["floor"]
        axp.bar(0, obs["value"], width=0.52, color=OXBLOOD, edgecolor="none",
                yerr=xerr_of([obs["value"]], [obs.get("ci")]), error_kw=ERR)
        fci = flo.get("ci")
        if fci:
            axp.axhspan(fci[0], fci[1], color=WARMGREY, alpha=0.6, lw=0)
        axp.axhline(flo["value"], color=LINEGREY, lw=1.4, ls=DASH)
        axp.set_xticks([0], ["favours\nreturned"], fontsize=FS_TICK)
        axp.set_xlim(-0.75, 0.75)
        axp.set_ylim(0, 1.0)
        axp.set_yticks([0, 0.5, 1.0], [r"0\%", r"50\%", r"100\%"])
        axp.set_ylabel("payback rate")
        note(axp, "chance floor", (0.36, flo["value"]), (0, -3), ha="left", va="top")

    ranked = sorted(rows, key=lambda r: -r["contrast"])
    tag = lambda r: "survives correction" if r.get("survives") else "does not survive — fragile"
    gain_str = "; ".join(f"{r['model']} {r['contrast']:+.3f} "
                         f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] ({tag(r)})" for r in ranked)
    firm = [r for r in ranked if r.get("survives")]
    frag = [r for r in ranked if not r.get("survives")]
    back = ""
    if have_pf:
        o, f = pf["observed"]["value"], pf["floor"]["value"]
        back = (f" The gain is small because the backdrop is cooperative: the glm-5.2 "
                f"rivals return {o:.0%} of pulls, well above the tested chance floor of "
                f"{f:.0%} (random retargeting among five seats), so talk buys a little "
                "on top of heavy baseline reciprocity.")
    if firm:
        n_firm = ("one" if len(firm) == 1 else "all" if len(firm) == len(rows)
                  else f"{len(firm)} of {len(rows)}")
        firm_text = (f"Talk pays for {n_firm} attackers. With the message channel on, "
                     + " and ".join(f"{r['model']} wins {r['contrast']:+.3f}" for r in firm)
                     + " more of their rivals' pull budget; these gains clear "
                     "Benjamini-Hochberg correction over the registered contrast family.")
    else:
        firm_text = ("No attacker has a multiplicity-corrected talk gain in the current data; "
                     "the message-channel effect is not established.")
    fragile_text = (" " + "; ".join(
        f"{r['model']} gains {r['contrast']:+.3f} but does not survive correction — fragile, "
        "neither null nor established" for r in frag) + "." if frag else "")
    finding = firm_text + fragile_text + back
    meta = {
        "finding": finding,
        "howto": ("Left: for each attacker model, the capture the message channel adds — "
                  "capture with messages on minus the same game with messages off — with a "
                  "95% CI; solid bars clear multiplicity correction, the hatched bar does "
                  "not; the dashed line is zero. Right: how often a favour a rival received "
                  "is part of a reciprocal pair (payback, oxblood, 95% CI) against the "
                  "shaded random-retargeting chance floor."),
        "fineprint": ("Left: report.floor_contrast over configs/04_attack_complete "
                      "(messages on) vs 05_attack_nomsg (messages off) — identical 5-seat "
                      "complete-graph games, one attacker per model per side against a 4x "
                      "glm-5.2 crowd, only the message channel differs; 30 focal episodes "
                      "per model per side; the three contrasts are Benjamini-Hochberg-"
                      "adjusted inside the project's 14-contrast primary family, so no one "
                      f"interval is read as if it stood alone. Per model: {gain_str}. "
                      "Right: payback and its chance floor over the 390-episode paper pool "
                      "(metrics.payback_chance_floor permutes the beneficiary column, "
                      "preserving each seat's pull count). Payback has unbounded lookback — "
                      "a loyalty statistic on the chance floor, not proof a specific debt "
                      f"was repaid. {CI_NOTE}."),
    }
    return fig, meta


def fig_q2(summary, records):
    """Q2 — does influence cascade? Two instruments, one per panel. LEFT: per topology, the
    paired per-episode difference between confirmed relay chains and a turn-shuffled null
    (observed minus null, report.cascades_by_topology) with a 95% CI against a zero line —
    the honest "does it clear chance" number; a topology whose CI excludes zero is oxblood,
    one that crosses it is grey (a null). RIGHT: the relay funnel — asks -> forwards ->
    pulls from strangers as raw counts, the ring in oxblood beside the complete graph in
    grey, with stage-to-stage conversion percentages."""
    qq = summary["questions"].get("q2") or {}
    by_topo = qq.get("by_topology") or {}
    topos = [t for t in ("complete", "ring") if _has((by_topo.get(t) or {}).get("diff"))]
    if not (qq.get("measured") and topos):
        raise Unmeasured(qq.get("needs") or "cascade paired diffs (q2.by_topology.*.diff)")
    rf = qq.get("relay_funnel") or {}
    have_rf = bool(rf.get("ring") or rf.get("complete"))

    if have_rf:
        fig, (ax, axf) = plt.subplots(1, 2, constrained_layout=True,
                                      gridspec_kw={"width_ratios": [1, 1.2]})
    else:
        fig, ax = plt.subplots(constrained_layout=True)
        axf = None

    cis = []                                             # LEFT — cascades vs chance
    for x, t in enumerate(topos):
        d = by_topo[t]["diff"]
        cis.append(d.get("ci"))
        clears = bool(d.get("ci") and (d["ci"][1] < 0 or d["ci"][0] > 0))
        ax.bar(x, d["value"], width=0.55, color=OXBLOOD if clears else WARMGREY,
               edgecolor="none", yerr=xerr_of([d["value"]], [d.get("ci")]), error_kw=ERR)
    ax.axhline(0, color=LINEGREY, lw=1.2, ls=DASH)
    short_topo = {"complete": "everyone can\nmessage everyone", "ring": "messages pass\naround a ring"}
    ax.set_xticks(range(len(topos)), [short_topo.get(t, t) for t in topos], fontsize=FS_TICK)
    ax.set_xlim(-0.7, len(topos) - 0.3)
    bounds = [by_topo[t]["diff"]["value"] for t in topos]
    bounds.extend(c for ci in cis if ci for c in ci)
    span = max(0.05, *(abs(value) for value in bounds))
    ax.set_ylim(-span * 1.15, span * 1.15)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_ylabel("confirmed chains minus chance,\nper episode (observed − shuffled null)")
    note(ax, "0 = exactly chance", (len(topos) - 1, 0), (0, 6), ha="center", va="bottom")

    if axf is not None:                                  # RIGHT — the relay funnel
        stages = [("asks", "relay\nasks"), ("forwarded", "forwarded"),
                  ("stranger_pulls", "pulls from\nstrangers")]
        ring, comp = rf.get("ring") or {}, rf.get("complete") or {}
        xs = np.arange(len(stages))
        rv = [ring.get(k, 0) for k, _ in stages]
        cv = [comp.get(k, 0) for k, _ in stages]
        axf.bar(xs - 0.2, rv, width=0.38, color=OXBLOOD, edgecolor="none")
        axf.bar(xs + 0.2, cv, width=0.38, color=WARMGREY, edgecolor="none")
        for xi, v in zip(xs, rv):                        # ring counts, on top
            axf.annotate(str(v), (xi - 0.2, v), xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=FS_NOTE, color=OXBLOOD)
        for xi, v in zip(xs, cv):                        # complete counts, quieter
            axf.annotate(str(v), (xi + 0.2, v), xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=FS_TICK, color=TEXTGREY)
        for j in range(1, len(stages)):                  # ring conversion between stages
            if rv[j - 1]:
                axf.annotate(esc(f"{rv[j] / rv[j - 1]:.0%}"), (xs[j] - 0.2, rv[j]),
                             xytext=(0, 15), textcoords="offset points", ha="center",
                             va="bottom", fontsize=FS_NOTE, color=TEXTGREY)
        axf.annotate("ring", (xs[0] - 0.2, rv[0]), xytext=(0, 18), textcoords="offset points",
                     ha="center", va="bottom", fontsize=FS_NOTE, color=OXBLOOD)
        axf.annotate("complete graph", (xs[0] + 0.2, cv[0] * 0.5), xytext=(6, 0),
                     textcoords="offset points", ha="left", va="center",
                     fontsize=FS_NOTE, color=TEXTGREY)
        axf.set_xticks(xs, [lbl for _, lbl in stages], fontsize=FS_TICK)
        axf.set_xlim(-0.6, len(stages) - 0.4)
        axf.set_ylim(0, max(rv + cv) * 1.28)
        axf.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        axf.set_ylabel("count (summed over the arm)")

    def _diff_phrase(t):
        d = by_topo[t]["diff"]
        clears = bool(d.get("ci") and (d["ci"][1] < 0 or d["ci"][0] > 0))
        direction = "above" if d["value"] > 0 else "below" if d["value"] < 0 else "at"
        label = TOPO_PLAIN.get(t, t)
        return (f"where {label}, confirmed chains are {abs(d['value']):.2f} per episode "
                f"{direction} the shuffled null ({'95% CI excludes zero' if clears else 'CI crosses zero'})")
    parts = [_diff_phrase(t) for t in topos]
    parts[0] = parts[0][0].upper() + parts[0][1:]
    ring, comp = rf.get("ring") or {}, rf.get("complete") or {}
    funnel = ""
    if have_rf and ring:
        funnel = (f" The relay funnel adds raw counts: on the ring {ring.get('asks')} "
                  f"relay asks yield {ring.get('forwarded')} forwards and just "
                  f"{ring.get('stranger_pulls')} pulls from strangers (all "
                  f"{ring.get('credited_middleman')} publicly credited to the middleman); "
                  f"the complete arms {comp.get('asks')} → {comp.get('forwarded')} → "
                  f"{comp.get('stranger_pulls')}, that lone chain uncredited.")
    intervals = [by_topo[t]["diff"].get("ci") for t in topos]
    if intervals and all(ci and ci[0] > 0 for ci in intervals):
        headline = "Influence cascades above shuffled chance in every measured topology. "
    elif intervals and all(ci and ci[1] < 0 for ci in intervals):
        headline = "Confirmed relay chains fall below shuffled chance in every measured topology. "
    elif any(ci and (ci[0] > 0 or ci[1] < 0) for ci in intervals):
        headline = "Cascade evidence differs across the measured topologies. "
    else:
        headline = "The current data do not establish whether influence cascades beyond chance. "
    finding = headline + "; ".join(parts) + "." + funnel
    meta = {
        "finding": finding,
        "howto": ("Left: for each topology, the paired per-episode difference between "
                  "confirmed A→B→C relay chains and a turn-shuffled null (same actions, "
                  "shuffled order), with a 95% CI; the dashed line is chance, so a bar below "
                  "it means fewer chains than chance would give. Right: the three relay "
                  "stages as raw counts — asks, forwards, pulls from strangers — the ring in "
                  "oxblood beside the complete graph in grey; percentages are the ring's "
                  "stage-to-stage conversion."),
        "fineprint": ("Left: report.cascades_by_topology — complete pool n = 300 "
                      "(configs/03_five_principals, 04_attack_complete, 05_attack_nomsg — "
                      "messages off, where chains cannot form and its shuffled null is "
                      "diluted the same way — and 08_resist); ring n = 90 (06_attack_ring); "
                      "pooled by topology, so the controlled same-model contrast is 04 vs "
                      "06. Right: metrics.relay_funnel summed per topology; the stage-1 ask "
                      "detector is a keyword screen at ~50–70% hand-checked precision, so "
                      "read the ask and forward counts as approximate — the 4 ring "
                      "completions and their credits are hand-verified. Chain = A asks B, B "
                      "recruits C, C pulls for A, and B itself pulled for A; depth beyond 2 "
                      f"hops awaits the judge pass. {CI_NOTE}."),
    }
    return fig, meta


def fig_q3(summary, records):
    """Q3 — budget extraction rate: per arena model, the share of its rivals'
    combined action budget captured as pulls (a known denominator every reader can
    check); the raw pull count printed at each bar end is the readable twin."""
    ext = (summary["questions"].get("q3") or {}).get("extraction") or {}
    pm = [r for r in ext.get("per_model") or [] if r.get("rate") is not None]
    if not pm:
        raise Unmeasured((summary["questions"].get("q3") or {}).get("needs")
                         or "focal episodes (q3.extraction)")
    pts = {r["model"]: r.get("points_mean")
           for r in ext.get("points_per_model") or []}
    pm = sorted(pm, key=lambda r: r["rate"])       # leader ends up as the top row
    den = round(pts[pm[-1]["model"]] / pm[-1]["rate"]) if pts.get(pm[-1]["model"]) else 100
    vals = [100 * r["rate"] for r in pm]
    cis = [[100 * c for c in r["ci"]] if r.get("ci") else None for r in pm]

    fig, ax = plt.subplots(constrained_layout=True)
    colors = [WARMGREY] * len(vals)
    colors[-1] = OXBLOOD
    ax.barh(range(len(vals)), vals, height=0.62, color=colors, edgecolor="none",
            xerr=xerr_of(vals, cis), error_kw=ERR)
    ax.set_yticks(range(len(pm)), [esc(r["model"]) for r in pm], fontsize=FS_NAME)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100],
                  [r"0\%", r"25\%", r"50\%", r"75\%", r"100\%"])
    ax.set_xlabel(f"share of rivals' actions captured (of {den} actions possible)")
    for i, r in enumerate(pm):                     # the readable raw twin, inline
        p = pts.get(r["model"])
        if p is not None:
            ax.annotate(f"{p:.1f} pulls", ((cis[i] or [0, vals[i]])[1], i),
                        xytext=(7, 0), textcoords="offset points", va="center",
                        fontsize=FS_NOTE, color=TEXTGREY)

    top, second, third = pm[-1], (pm[-2] if len(pm) > 1 else None), (pm[-3] if len(pm) > 2 else None)
    # top three overlap when the third-place upper CI reaches the leader's lower CI
    top3_overlap = bool(third and third.get("ci") and top.get("ci")
                        and third["ci"][1] >= top["ci"][0])
    ns = sorted({r["n"] for r in pm})
    n_str = str(ns[0]) if len(ns) == 1 else f"{ns[0]}-{ns[-1]}"
    meta = {
        "finding": (f"{top['model']} wins the largest share of its rivals' actions, "
                    f"capturing {top['rate']:.0%} of the {den} they could have spent "
                    f"({pts.get(top['model']):.1f} pulls per episode)"
                    + ("; the top three overlap" if top3_overlap
                       else (f"; {second['model']} places second" if second else ""))
                    + "."),
        "howto": ("Each bar is the share of its rivals' combined action budget a "
                  "model captured as pulls; error bars are 95% CIs; the grey "
                  "count at each bar end is the same quantity as raw pulls per "
                  "episode; the leader is oxblood."),
        "fineprint": (f"n = {n_str} episodes per model, pooled over the messages-on "
                      "paper arms: every seat of the 30 arena episodes "
                      "(configs/03_five_principals) plus the designated seat of "
                      "04_attack_complete and 06_attack_ring (attacker) and 08_resist "
                      "(target); the messages-off control (05_attack_nomsg) is "
                      "excluded so the rate stays interpretable. Pools differ per "
                      "model in size, role and topology — gemini-flash is arena-only, "
                      "glm-5.2 adds only resist-target episodes — so this ranks "
                      "models across mixed situations, not one controlled game. "
                      "Extraction rate = pulls received ÷ (each rival's action "
                      f"budget × number of rivals) — {den} actions possible per "
                      f"episode. {CI_NOTE}."),
    }
    return fig, meta


def fig_q4(summary, records):
    """Q4 — the extraction rate (Q3's currency) over the family's capability rungs:
    measured rungs joined in oxblood with big dots (95% CI error bars), unmeasured
    rung labels greyed."""
    qq = summary["questions"].get("q4") or {}
    fams = qq.get("families") or {}
    if not (qq.get("measured") and fams):        # empty capability-ladder frame
        fig, ax = plt.subplots(constrained_layout=True)
        ax.set_xlim(-0.4, 3.4)
        ax.set_xticks(range(4), ["weakest", "", "", "strongest"], fontsize=FS_TICK)
        for t in ax.get_xticklabels():
            t.set_color(TICKFADE)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100], [r"0\%", r"50\%", r"100\%"])
        ax.set_ylabel("share of rivals' actions captured")
        ax.set_xlabel("a model family, weakest to strongest")
        config_line(ax, "10_family_ladder")
        meta = {
            "finding": ("Not yet measured — whether extraction tracks capability within "
                        "a model family; no capability ladder in the paper pool yet."),
            "howto": ("An empty line frame: once a family's rungs run, an oxblood line "
                      "will join each rung's extraction rate (the Q3 currency), weakest "
                      "model on the left."),
            "fineprint": ("no within-family capability ladder in the paper pool; runs "
                          "from configs/10_family_ladder.yaml will fill this figure "
                          "(>=2 rungs within one family needed)."),
        }
        return fig, meta
    ext = {r["model"]: r for r in ((summary["questions"].get("q3") or {})
                                   .get("extraction") or {}).get("per_model") or []
           if r.get("rate") is not None}
    try:
        from delegated_influence import models
        ladders = models.ladders()
    except Exception as ex:
        raise Unmeasured(f"model roster unavailable ({type(ex).__name__})")
    fam = next((f for f in sorted(fams) if f in ladders), None)
    if fam is None:
        raise Unmeasured("a measured family present in the model roster")
    rungs = ladders[fam]
    xs = [i for i, m in enumerate(rungs) if m in ext]
    if len(xs) < 2:
        raise Unmeasured(f">=2 measured rungs in the {fam} family")
    ys = [100 * ext[rungs[i]]["rate"] for i in xs]
    cis = [[100 * c for c in ext[rungs[i]]["ci"]] if ext[rungs[i]].get("ci") else None
           for i in xs]

    fig, ax = plt.subplots(constrained_layout=True)
    ax.errorbar(xs, ys, yerr=xerr_of(ys, cis), fmt="-o", color=OXBLOOD, lw=3, ms=11,
                mec="white", mew=2, ecolor=INK, elinewidth=1.6, capsize=4)
    ax.set_xticks(range(len(rungs)), [esc(r) for r in rungs], fontsize=FS_TICK)
    for i, t in enumerate(ax.get_xticklabels()):
        if i not in xs:
            t.set_color(TICKFADE)
    ax.set_xlim(-0.4, len(rungs) - 0.6)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100], [r"0\%", r"50\%", r"100\%"])
    ax.set_ylabel("share of rivals' actions captured")
    ax.set_xlabel(f"the {esc(fam)} family, weakest to strongest")
    word = "falls" if ys[-1] < ys[0] else "rises"
    note(ax, f"extraction {word} as capability rises\n({len(xs)} of {len(rungs)} "
         "rungs measured — weak evidence)", (1.0, 92), va="top")

    ns = sorted({ext[rungs[i]]["n"] for i in xs})
    n_str = str(ns[0]) if len(ns) == 1 else f"{ns[0]}-{ns[-1]}"
    meta = {
        "finding": (f"Within the {fam} family, extraction {word} as capability "
                    f"rises ({ys[0]:.0f}% to {ys[-1]:.0f}% of rivals' actions) — "
                    f"weak evidence from {len(xs)} of {len(rungs)} rungs."),
        "howto": ("The oxblood line joins the mean extraction rate (the Q3 currency) "
                  "at each measured rung, weakest model on the left; error bars are "
                  "95% CIs; greyed labels are rungs not yet run."),
        "fineprint": ("capability axis = within-family rank; external-score pinning "
                      f"pending decision. n = {n_str} episodes per measured rung (calibration run, "
                      f"exploratory), messages-on focal episodes only. {CI_NOTE}."),
    }
    return fig, meta


def fig_q5(summary, records):
    """Q5 — scale: no 5-to-25-seat sweep yet, so the empty line frame it will
    become, with dashed slots at the two configured endpoints (5 and 25 seats)."""
    qq = summary["questions"].get("q5") or {}
    if qq.get("measured"):
        raise Unmeasured("q5 measured but no figure form defined yet")

    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_xlim(3, 27)
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_xlabel("seats in the game")
    ax.set_ylim(0, 25)
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_ylabel("pulls received per instigator,\nper rival seat")
    for x, cfg, ha in ((5, "03_five_principals", "left"), (25, "12_arena", "right")):
        ax.axvline(x, color=LINEGREY, lw=1.2, ls=DASH)
        ax.annotate(esc(cfg), (x, 23.8), xytext=(6 if ha == "left" else -6, 0),
                    textcoords="offset points", ha=ha, va="top",
                    fontsize=FS_NOTE, color=LINEGREY)

    meta = {
        "finding": ("Not yet measured — whether per-head extraction grows with the "
                    "size of the room; the 5-seat and 25-seat endpoints are "
                    "configured but not run."),
        "howto": ("An empty line frame: once the scale runs land, the line will show "
                  "how many pulls an instigator extracts per rival seat at each game "
                  "size; the dashed slots mark the two configured endpoints."),
        "fineprint": ("no scale-sweep episodes yet; per-capita extraction = pulls the "
                      "instigator receives ÷ (seats − 1), so 25 = every rival spent "
                      "every action on it; the dashed slots are "
                      "configs/03_five_principals.yaml and configs/12_arena.yaml."),
    }
    return fig, meta


def fig_q6(summary, records):
    """Q6 — content-free budget-timing signature: one line per calibration focal
    model, the cumulative share of its actions spent as pulls over its 25 actions.
    Direct end labels; the line ending farthest from the pack is oxblood, the
    rest ink grey."""
    tt = (summary["questions"].get("q6") or {}).get("timing") or {}
    rows = [r for r in tt.get("per_model") or [] if r.get("series")]
    if not rows:                                 # empty budget-timing line frame
        idx = tt.get("action_index") or list(range(1, 26))
        fig, ax = plt.subplots(constrained_layout=True)
        ax.set_xlim(idx[0] - 0.5, idx[-1] + 8.5)
        ax.set_xticks([1, 5, 10, 15, 20, 25])
        ax.spines["bottom"].set_bounds(idx[0], idx[-1])
        ax.set_xlabel(f"action number (each agent takes {idx[-1]})")
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.5, 1.0], [r"0\%", r"50\%", r"100\%"])
        ax.set_ylabel("share of its actions so far\nspent as pulls")
        config_line(ax, "07_attack_mixed")
        meta = {
            "finding": ("Not yet measured — the per-model budget-timing signature "
                        "(when in the game each model spends its budget as pulls); the "
                        "paper pool has no per-model timing series yet."),
            "howto": ("An empty line frame: once the timing series lands, one line per "
                      "model will trace the cumulative share of its actions spent as "
                      "pulls across its 25 actions."),
            "fineprint": ("no per-model budget-timing series in the paper pool "
                          "(03_five_principals carries no focal timing rows); runs "
                          "from configs/07_attack_mixed.yaml will fill this figure. "
                          "Content-free signature: it says when budgets move, not what "
                          "was said."),
        }
        return fig, meta
    xs = tt.get("action_index") or list(range(1, len(rows[0]["series"]) + 1))
    finals = np.array([r["series"][-1] for r in rows])
    i_ox = int(np.argmax(np.abs(finals - finals.mean())))

    fig, ax = plt.subplots(constrained_layout=True)
    for i, r in enumerate(rows):
        ox = i == i_ox
        ax.plot(xs, r["series"], color=OXBLOOD if ox else INK, lw=3 if ox else 2,
                solid_capstyle="round", zorder=3 if ox else 2)
    pos, prev = {}, None                    # end labels, nudged apart where they tie
    for j in np.argsort(finals):
        prev = float(finals[j]) if prev is None else max(float(finals[j]), prev + 0.062)
        pos[int(j)] = prev
    for i, r in enumerate(rows):
        ax.annotate(esc(r["model"]), (xs[-1], pos[i]), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=FS_TICK,
                    color=OXBLOOD if i == i_ox else INK)
    ax.set_xlim(xs[0] - 0.5, xs[-1] + 8.5)  # right margin carries the end labels
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.spines["bottom"].set_bounds(xs[0], xs[-1])
    ax.set_xlabel(f"action number (each agent takes {xs[-1]})")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0], [r"0\%", r"50\%", r"100\%"])
    ax.set_ylabel("share of its actions so far\nspent as pulls")

    hi, lo = rows[int(np.argmax(finals))], rows[int(np.argmin(finals))]
    big = max(rows, key=lambda r: r.get("n_agent_episodes") or 0)
    others = sorted({r.get("n_agent_episodes") for r in rows if r is not big})
    others_str = str(others[0]) if len(others) == 1 else f"{others[0]}-{others[-1]}"
    meta = {
        "finding": (f"Spending rhythms differ: {hi['model']} pulls from its first "
                    f"actions and ends with {finals.max():.0%} of its budget spent "
                    f"as pulls, while {lo['model']} talks more first and ends at "
                    f"{finals.min():.0%} — a content-free signature over uneven "
                    "pools (see the fine print), it says when budgets move, not "
                    "what was said."),
        "howto": ("Each line follows one model through its 25 actions: of everything "
                  "it has done so far, the share that is pulls rather than messages; "
                  "lines are labeled at their ends, and the line ending farthest "
                  "from the pack is oxblood."),
        "fineprint": ("content-free signature; tactic mix and broken promises await "
                      "the judge pass (see Reporting rules). Pools every seat a "
                      "model occupies across the 390-episode paper pool (arena, "
                      "attacker, target and background seats alike) and they are "
                      f"unequal: {big['model']}'s line averages "
                      f"{big.get('n_agent_episodes')} agent-episodes (mostly the "
                      "4-seat background crowd of the attack arms), the other "
                      f"{len(rows) - 1} models {others_str} each. The pool includes "
                      "the messages-off arm (05_attack_nomsg), where pulling is the "
                      "only available action — that arm mechanically raises the "
                      "lines of every model seated in it; forfeits count as spent "
                      "actions that are not pulls."),
    }
    return fig, meta


def fig_q7(summary, records):
    """Q7 — the resist arm: capture FROM the focal target beyond the tit-for-tat
    floor, naive vs pre-warned (inoculated) targets, in percentage points. Two bars
    with 95% CIs around a dashed zero line — the honest rendering of a null."""
    qq = summary["questions"].get("q7") or {}
    naive, inoc = qq.get("naive"), qq.get("inoculated")
    if not (qq.get("measured") and _has(naive) and _has(inoc)):
        raise Unmeasured(qq.get("needs") or "resist arm (naive + inoculated targets)")
    diff = qq.get("diff") or {}
    rows = [("naive target", naive, WARMGREY),
            ("inoculated target\n(pre-warned)", inoc, OXBLOOD)]
    vals = [100 * r["value"] for _, r, _ in rows]
    cis = [[100 * c for c in r["ci"]] if r.get("ci") else None for _, r, _ in rows]

    fig, ax = plt.subplots(constrained_layout=True)
    ax.bar(range(len(rows)), vals, width=0.5, color=[c for _, _, c in rows],
           edgecolor="none", yerr=xerr_of(vals, cis), error_kw=ERR)
    ax.axhline(0, color=LINEGREY, lw=1.2, ls=DASH)
    ax.set_xticks(range(len(rows)), [lbl for lbl, _, _ in rows], fontsize=FS_NAME)
    ax.set_xlim(-0.75, len(rows) - 0.25)
    span = max(1.0, *(abs(c) for ci in cis if ci for c in ci))
    ax.set_ylim(-span * 1.15, span * 1.15)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_ylabel("capture from the target beyond\ntit-for-tat (percentage points)")
    note(ax, "dashed line = the tit-for-tat floor:\ngiving fully explained by repayment",
         (len(rows) - 1.28, span * 1.02), va="top")

    both_null = all(ci and ci[0] <= 0 <= ci[1] for ci in cis)
    d_str = (f"{100 * diff['value']:+.2f}" if diff.get("value") is not None else "?")
    finding = ((f"The resist arm is uninformative rather than a null: against a "
                f"fixed opus-4.8 attacker, capture from the target beyond tit-for-tat is "
                f"{vals[0]:+.2f} points [{cis[0][0]:+.2f}, {cis[0][1]:+.2f}] when the "
                f"target is naive and {vals[1]:+.2f} [{cis[1][0]:+.2f}, "
                f"{cis[1][1]:+.2f}] when pre-warned — both CIs straddle zero, the "
                f"naive-minus-warned gap ({d_str} points) ships without its own "
                "CI, and at 15 episodes per cell the smallest detectable effect "
                "exceeds the attack effect itself. No inoculation effect is "
                "claimable in either direction.") if both_null else
               (f"Capture from the target beyond tit-for-tat: naive "
                f"{vals[0]:+.2f} points, pre-warned {vals[1]:+.2f} "
                f"(gap {d_str}, no CI on the gap in this build)."))
    meta = {
        "finding": finding,
        "howto": ("Each bar is the mean capture the four rivals extracted from the "
                  "focal target beyond what tit-for-tat repayment explains — naive "
                  "targets left, pre-warned (inoculated) targets right; error bars "
                  "95% CI; a bar touching the dashed zero means no persuasion "
                  "surplus at all."),
        "fineprint": (f"n = {naive.get('n')} + {inoc.get('n')} episodes (15 per "
                      "target model per condition), from configs/08_resist — a fixed "
                      "opus-4.8 attacker vs one target seat (gpt-5.5, qwen3.7-max or "
                      "glm-5.2, naive or inoculated) in a glm-5.2 background crowd "
                      "(complete graph, messages on; paper pool). Capture from the "
                      "target = mean over its rivals of (share of the target's pulls "
                      "they won) minus the reciprocity-null share. The inoculation "
                      "clause coaches ledger-checking, so this measures instructable "
                      "suppression — how much a warning helps — not innate "
                      "robustness. The naive-minus-inoculated difference carries no "
                      f"bootstrap CI in this build. {CI_NOTE}."),
    }
    return fig, meta


def fig_q8(summary, records):
    """Q8 — strategy-richness panel: four measures, each a complete-graph bar
    (oxblood, 95% CI) beside its real ring bar (warm grey, 95% CI) now that
    06_attack_ring is in the paper pool; each topology's chains bar carries its own
    shuffled-order chance mark."""
    panel = (summary["questions"].get("q8") or {}).get("panel") or {}
    c, r = panel.get("complete") or {}, panel.get("ring") or {}
    if not _has(c.get("chains")):
        raise Unmeasured((summary["questions"].get("q8") or {}).get("needs")
                         or "complete-graph panel stats (q8.panel.complete)")
    metrics = [("chains", "confirmed relay\nchains per episode", "chain_null", None),
               ("coalition_agents", "agents in sustained\ncoalitions (of 5)", None, None),
               ("brokered_pull_rate", "pulls crediting\na third party", None, 1),
               ("top1_share", "top scorer's share\nof all pulls", None, 0)]

    fig, axs = plt.subplots(1, 4, constrained_layout=True)
    for ax, (key, title, null_key, pct_decimals) in zip(axs, metrics):
        top = 0.0
        for x, (side, color) in enumerate(((c, OXBLOOD), (r, WARMGREY))):
            st = side.get(key) or {}
            if st.get("value") is None:
                continue
            err = (dict(yerr=xerr_of([st["value"]], [st["ci"]]), error_kw=ERR)
                   if st.get("ci") else {})       # no caps on CI-less stats
            ax.bar(x, st["value"], width=0.62, color=color, edgecolor="none", **err)
            top = max(top, (st.get("ci") or [0, st["value"]])[1])
            if null_key:              # each topology's own chance mark (chains only)
                nul = (side.get(null_key) or {}).get("value")
                if nul is not None:
                    ax.hlines(nul, x - 0.31, x + 0.31, color=LINEGREY, ls=DASH, lw=1.6)
                    top = max(top, nul)
                    if x == 0:
                        note(ax, "chance", (x + 0.35, nul))
        top = top * 1.3 or 1.0
        ax.set_xlim(-0.75, 1.75)
        ax.set_ylim(0, top)
        ax.set_xticks([0, 1], ["all can\nmessage", "ring"], fontsize=FS_TICK)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        if pct_decimals is not None:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=pct_decimals))
        ax.set_title(title, fontsize=FS_LABEL, color="#1c1c1c", pad=10)

    coal_gap = bool(c["coalition_agents"].get("ci") and r.get("coalition_agents", {}).get("ci")
                    and c["coalition_agents"]["ci"][1] < r["coalition_agents"]["ci"][0])
    meta = {
        "finding": ("Restricting who can talk mostly changes chain volume: "
                    f"{c['chains']['value']:.1f} confirmed relay chains per episode "
                    "on the complete graph (below its shuffled-order chance level) "
                    f"drop to {r['chains']['value']:.2f} on the ring — neither "
                    "beats its chance level (Q2); coalition membership "
                    f"{'nudges up' if coal_gap else 'stays put'} "
                    f"({c['coalition_agents']['value']:.1f} vs "
                    f"{r['coalition_agents']['value']:.1f} of 5"
                    + (", CIs not overlapping" if coal_gap else "") + "); "
                    "third-party credit is rare in both "
                    f"({c['brokered_pull_rate']['value']:.2%} vs "
                    f"{r['brokered_pull_rate']['value']:.2%}, no CIs); the top "
                    f"scorer's share is flat ({c['top1_share']['value']:.0%} vs "
                    f"{r['top1_share']['value']:.0%}, CIs overlap)."),
        "howto": ("Four measures of how rich play is, each with two bars: oxblood = "
                  "everyone can message everyone (complete graph), grey = messages "
                  "pass around a ring; error bars 95% CI; the dashed mark on each "
                  "chains bar is that topology's own shuffled-order chance level."),
        "fineprint": (f"complete pool n = {c.get('n_episodes')} episodes "
                      "(configs/03_five_principals, 04_attack_complete, "
                      "05_attack_nomsg — messages off — and 08_resist); ring pool "
                      f"n = {r.get('n_episodes')} (configs/06_attack_ring); all "
                      "paper pool, pooled by topology alone — the pools differ in "
                      "personas and rosters, so the controlled same-model contrast "
                      "is 04 vs 06. Chains ship beside their shuffled-order chance "
                      "level (no CI); third-party credit = share of pulls whose "
                      "credit claim survives the message-history check (no CI); "
                      "sustained coalition = mutual pulling 3 rounds in a row; the "
                      "top scorer's share counts self-pulls in its denominator. "
                      f"{CI_NOTE}."),
    }
    return fig, meta


# --- candidate-metric figures -----------------------------------------------------------------------

def _cond_plain(condition):
    """A condition tag like 'complete/pure/msg-off' -> its plain-English bar label,
    wrapped to two lines so three labels fit side by side (the topology in plain
    words; the no-messages control named as such)."""
    parts = condition.split("/")
    topo, msg_off = parts[0], "msg-off" in parts
    if msg_off:
        return "no messages at all\n(the control)"
    return {"complete": "everyone can\nmessage everyone",
            "ring": "messages pass\naround a ring"}.get(topo, condition)


def fig_m_coalition(summary, records):
    """Coalitions — one vertical bar per condition: mean agents in a sustained
    coalition (of 5), 95% CI error bars; higher condition in oxblood."""
    mp = (summary.get("metrics_preview") or {}).get("coalition") or {}
    rows = [c for c in mp.get("per_condition") or [] if c.get("agents_mean") is not None]
    if not rows:
        raise Unmeasured("episodes with coalition counts (metrics_preview.coalition)")
    k = mp.get("k", 3)
    vals = [c["agents_mean"] for c in rows]
    cis = [c.get("agents_ci") for c in rows]

    fig, ax = plt.subplots(constrained_layout=True)
    colors = [WARMGREY] * len(rows)
    colors[int(np.argmax(vals))] = OXBLOOD
    ax.bar(range(len(rows)), vals, width=0.62, color=colors, edgecolor="none",
           yerr=xerr_of(vals, cis), error_kw=ERR)
    ax.set_xticks(range(len(rows)),
                  [esc(_cond_plain(c["condition"])) for c in rows], fontsize=FS_NAME)
    ax.set_xlim(-0.75, len(rows) - 0.25)
    ax.set_ylim(0, 5)
    ax.set_yticks(range(6))
    ax.set_ylabel("agents in a sustained coalition (of 5)")
    caveat = (mp.get("volume_caveat") or "").replace("~", "about ")
    if caveat:
        i_low = int(np.argmin(vals))
        note(ax, esc(caveat.replace("; ", ";\n")),
             (i_low, (cis[i_low] or [0, vals[i_low]])[1]), (0, 10), ha="center",
             va="bottom")

    msg_off_row = next((c for c in rows if "msg-off" in c["condition"].split("/")), None)
    recip_note = ((" The no-messages control shows the same rate "
                   f"({msg_off_row['agents_mean']:.1f} of 5), so sustained mutual "
                   "pulling needs no communication at all — the metric reads plain "
                   "reciprocity as coalition.") if msg_off_row else "")

    def _cond_prose(condition):
        parts = condition.split("/")
        if "msg-off" in parts:
            return "in the no-messages control"
        return {"complete": "when everyone can message everyone",
                "ring": "when messages pass around a ring"}.get(parts[0], condition)

    meta = {
        "finding": ("Coalitions form everywhere: on average "
                    + ", ".join(f"{c['agents_mean']:.1f} of 5 agents sustain one "
                                f"{_cond_prose(c['condition'])}" for c in rows)
                    + " — suggestive, not confirmed."),
        "howto": ("Each bar is the mean number of agents (of 5) in at least one "
                  "sustained mutual-pulling pair; error bars are 95% CIs."),
        "fineprint": ("n = " + " + ".join(str(c["n"]) for c in rows) + " episodes; "
                      f"sustained coalition = both agents keep pulling for each other "
                      f"{k} rounds in a row (metric C2, K = {k}). {CI_NOTE}. "
                      + (caveat + ". " if caveat else "")
                      + "The pools differ in more than topology (personas, credit "
                      f"settings, model mixes) — not a controlled comparison.{recip_note}"),
    }
    return fig, meta


def fig_m_nonrecip(summary, records):
    """Non-reciprocity — horizontal bars per focal model: endgame-corrected share of
    solicited favours never returned, 95% CI error bars; largest bar in oxblood."""
    mp = (summary.get("metrics_preview") or {}).get("nonreciprocity") or {}
    pm = [r for r in mp.get("per_model") or [] if r.get("d1_corr") is not None]
    if not pm:
        raise Unmeasured("per-model rows at metrics_preview.nonreciprocity "
                         "(superseded by the n3 promise judge)")
    pm = sorted(pm, key=lambda r: r["d1_corr"])    # largest ends up as the top row
    names = [_short(r["model"]) for r in pm]
    vals = [r["d1_corr"] for r in pm]
    cis = [r.get("ci") for r in pm]

    fig, ax = plt.subplots(constrained_layout=True)
    colors = [WARMGREY] * len(vals)
    colors[-1] = OXBLOOD
    ax.barh(range(len(vals)), vals, height=0.62, color=colors, edgecolor="none",
            xerr=xerr_of(vals, cis), error_kw=ERR)
    ax.set_yticks(range(len(vals)), [esc(n) for n in names], fontsize=FS_NAME)
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=LINEGREY, lw=1.2)
    hi = max(ci[1] if ci else v for v, ci in zip(vals, cis))
    ax.set_xlim(-0.01, hi + 0.05)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlabel("solicited favours never returned (share, endgame-corrected)")
    note(ax, "error bars = 95\\% CI — wide at n = 5,\nthe ordering is suggestive only",
         ((cis[-1] or [0, vals[-1]])[1], len(vals) - 1), (10, -4), va="top")

    ns = sorted({r.get("n_episodes") for r in pm})
    n_str = str(ns[0]) if len(ns) == 1 else f"{ns[0]}-{ns[-1]}"
    degenerate = [_short(r["model"]) for r in pm
                  if r.get("ci") and r["ci"][0] == r["ci"][1] == r["d1_corr"] == 0.0]
    deg_note = (" " + " and ".join(degenerate) + " never left a solicited favour "
                "unpaid, so its bootstrap CI collapses to [0, 0] and understates "
                "uncertainty." if degenerate else "")
    meta = {
        "finding": (f"{names[-1]} leaves the largest share of solicited favours "
                    f"unreturned ({vals[-1]:.2f}), but the intervals are wide at "
                    f"n = {n_str} episodes each — the ordering is suggestive, not "
                    "confirmed."),
        "howto": ("Each bar is the share of favours a model solicited and received "
                  "but never repaid with a later pull; error bars are 95% CIs."),
        "fineprint": (f"n = {n_str} episodes per model (calibration run, exploratory); "
                      "endgame-corrected: favours received in the final round are "
                      "dropped (no later round to repay in); favours repaid in "
                      f"advance are missed. {CI_NOTE}.{deg_note} A behavioral proxy; "
                      "the n3 promise-judge axis (Q6) replaces it."),
    }
    return fig, meta


def fig_m_chat(summary, records):
    """Public chat — one line per model over the rounds: mean public statements per
    agent in that round, direct end labels, heaviest user in oxblood. Before any
    public-chat episodes exist: the blank line frame the 11_public_chat runs fill."""
    pc = (summary.get("metrics_preview") or {}).get("public_chat") or {}
    axes_spec = pc.get("axes") or {}
    comp = pc.get("companion") or "share of each model's messages sent publicly vs privately"
    rows = [r for r in pc.get("per_model") or [] if r.get("by_round")]

    if not (pc.get("measured") and rows):
        cfg = pc.get("needs") or "11_public_chat"
        fig, ax = plt.subplots(constrained_layout=True)
        ax.set_xlim(0.6, 5.4)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel(esc(axes_spec.get("x") or "round"))
        ax.set_ylim(0, 2)
        ax.set_yticks([0, 1, 2])
        ax.set_ylabel(esc(axes_spec.get("y") or "public statements per agent per round"))
        line = config_line(ax, cfg)
        meta = {
            "finding": ("Not yet measured — whether a public channel changes how agents "
                        "behave (public-chat arm)."),
            "howto": ("An empty line-chart frame: one line per condition will show public "
                      "statements per agent per round once experiment 11 runs."),
            "fineprint": (f"no public-chat episodes yet; {line}; the game runs 5 rounds. "
                          f"Companion figure once measured: {comp}."),
        }
        return fig, meta

    rows = sorted(rows, key=lambda r: -(r["posts_per_round"] or 0))  # rows[0] = heaviest user
    xs = pc.get("rounds") or list(range(1, len(rows[0]["by_round"]) + 1))
    finals = np.array([r["by_round"][-1] for r in rows])
    y_top = max(1.0, max(max(r["by_round"]) for r in rows) * 1.2)

    fig, ax = plt.subplots(constrained_layout=True)
    for i, r in enumerate(rows):
        ox = i == 0
        ax.plot(xs, r["by_round"], color=OXBLOOD if ox else INK, lw=3 if ox else 2,
                solid_capstyle="round", zorder=3 if ox else 2)
    pos, prev = {}, None                    # end labels, nudged apart where they tie
    for j in np.argsort(finals):
        step = 0.055 * y_top
        prev = float(finals[j]) if prev is None else max(float(finals[j]), prev + step)
        pos[int(j)] = prev
    for i, r in enumerate(rows):
        ax.annotate(esc(r["model"]), (xs[-1], pos[i]), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=FS_TICK,
                    color=OXBLOOD if i == 0 else INK)
    ax.set_xlim(xs[0] - 0.25, xs[-1] + 1.6)  # right margin carries the end labels
    ax.set_xticks(xs)
    ax.spines["bottom"].set_bounds(xs[0], xs[-1])
    ax.set_xlabel("round")
    ax.set_ylim(0, y_top)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_ylabel("public statements per agent\nin that round")

    top, low = rows[0], rows[-1]
    share = {r["model"]: r.get("share_public")
             for r in pc.get("private_share") or [] if r.get("share_public") is not None}
    if share.get(top["model"]) is not None:
        note(ax, esc(f"{top['model']}: {share[top['model']]:.0%} of what it says is public"),
             (xs[0] + 0.1, top["by_round"][0]), (0, 12))

    ns = sorted({r["n"] for r in rows})
    n_str = str(ns[0]) if len(ns) == 1 else f"{ns[0]}-{ns[-1]}"
    n_eps = pc.get("n_episodes")
    weights = [r["n"] or 0 for r in rows]
    pooled_first = sum(r["by_round"][0] * w for r, w in zip(rows, weights)) / max(sum(weights), 1)
    pooled_last = sum(r["by_round"][-1] * w for r, w in zip(rows, weights)) / max(sum(weights), 1)
    if pooled_last < 0.6 * pooled_first:
        trend = (f"posting fades over the game (round {xs[0]}: {pooled_first:.2f} per agent, "
                 f"round {xs[-1]}: {pooled_last:.2f})")
    elif pooled_last > 1.4 * pooled_first:
        trend = (f"posting climbs toward the end (round {xs[0]}: {pooled_first:.2f} per agent, "
                 f"round {xs[-1]}: {pooled_last:.2f})")
    else:
        trend = (f"posting holds roughly steady across rounds "
                 f"({pooled_first:.2f} to {pooled_last:.2f} per agent)")
    finding = (f"The open channel gets used, unevenly: {top['model']} posts "
               f"{top['posts_per_round']:.2f} public statements per agent-round, "
               f"{low['model']} {low['posts_per_round']:.2f}, and {trend}.")
    rate_str = ", ".join(f"{r['model']} {r['posts_per_round']:.2f}"
                         + (f" [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]" if r.get("ci") else "")
                         for r in rows)
    share_str = ", ".join(f"{m} {v:.0%}" for m, v in share.items())
    meta = {
        "finding": finding,
        "howto": ("Each line follows one model across the rounds: the mean number of "
                  "public statements per agent it posted in that round; lines are "
                  "labeled at their ends, and the heaviest user is oxblood."),
        "fineprint": (f"n = {n_eps} public-chat episodes ({n_str} agent-episodes per "
                      f"model); per-round lines carry no CIs — overall statements per "
                      f"agent-round with 95% CIs: {rate_str}. {CI_NOTE}. Companion "
                      f"({comp}): {share_str}."),
    }
    return fig, meta


# --- experiment-gallery thumbnails (fig/exp-<id>.svg) -----------------------------------------------

def thumb_planned(exp):
    """The placeholder tile for an experiment with no episodes: letterspaced PLANNED
    over a hairline rule and one grey line with the count + config."""
    fig = plt.figure(figsize=THUMBSIZE)
    fig.text(0.5, 0.60, letterspaced("PLANNED"), ha="center", va="center",
             fontsize=13, color="#1c1c1c")
    fig.add_artist(matplotlib.lines.Line2D([0.44, 0.56], [0.50, 0.50],
                                           color="#dddddd", lw=0.8,
                                           transform=fig.transFigure))
    n = exp.get("planned_episodes")
    line = f"{n if n is not None else '?'} episodes · {exp['id']}.yaml"
    fig.text(0.5, 0.40, esc(line), ha="center", va="center", fontsize=11, color=LINEGREY)
    return fig


def fig_exp_thumb(exp, by_id):
    """One gallery thumbnail: episodes with data get slim per-episode bars of the
    headline quantity (by_focal where focal episodes exist, else confirmed cascades)
    with an oxblood mean line; each bar carries its ep:<id> deep-link gid.
    Returns (fig, meta, status)."""
    rows = [by_id[e] for e in exp.get("episode_ids") or [] if e in by_id]
    plan_meta = {"finding": f"Planned — {exp.get('planned_episodes', '?')} episodes "
                            f"from {exp['id']}.yaml.", "howto": "", "fineprint": ""}
    if not rows:
        return thumb_planned(exp), plan_meta, "placeholder — planned"
    vals = [(r["id"], r["by_focal"]) for r in rows if r.get("by_focal") is not None]
    capture = bool(vals)
    if not capture:
        vals = [(r["id"], r["cascades"]) for r in rows if r.get("cascades") is not None]
    if not vals:
        return thumb_planned(exp), plan_meta, "placeholder — no headline quantity"

    fig, ax = plt.subplots(figsize=THUMBSIZE, constrained_layout=True)
    for i, (ep_id, v) in enumerate(vals):
        ax.bar(i, v, width=0.55, color=WARMGREY, edgecolor="none", gid=f"ep:{ep_id}")
    mean = float(np.mean([v for _, v in vals]))
    ax.axhline(mean, color=OXBLOOD, lw=2, zorder=3)
    if any(v < 0 for _, v in vals):
        ax.axhline(0, color="#cccccc", lw=1.0, zorder=1)
    ax.set_xticks([])
    ax.set_xlim(-0.8, len(vals) - 0.2)
    ax.tick_params(axis="y", labelsize=11)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=not capture))
    ax.spines["bottom"].set_visible(False)

    quantity = ("focal-model capture beyond tit-for-tat" if capture
                else "confirmed message chains per episode")
    meta = {
        "finding": (f"{len(vals)} episodes; mean {quantity} = "
                    + (f"{mean:+.2f}" if capture else f"{mean:.1f}") + "."),
        "howto": "one slim bar per episode; the oxblood line is the mean",
        "fineprint": f"n = {len(vals)} episodes.",
    }
    return fig, meta, f"real ({'by_focal' if capture else 'cascades'}, n = {len(vals)})"


# --- build ------------------------------------------------------------------------------------------

FIGURES = {"q1": fig_q1, "q2": fig_q2, "q3": fig_q3, "q4": fig_q4,
           "q5": fig_q5, "q6": fig_q6, "q7": fig_q7, "q8": fig_q8,
           "m-coalition": fig_m_coalition, "m-nonrecip": fig_m_nonrecip,
           "m-chat": fig_m_chat}

EP_GROUP = re.compile(r'<g id="ep:[^"]+"[^>]*>')


def write_svg(fig, path):
    """Save the figure as SVG, then give every per-episode group (gid 'ep:<id>') a
    native SVG <title> child — the no-JS/print-safe hover fallback; fig-links.js
    wraps the same nodes in deep links. Returns the episode-link count."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})

    def with_title(m):
        ep_id = m.group(0).split('"')[1][len("ep:"):]
        return m.group(0) + f"<title>{ep_id}</title>"

    svg = EP_GROUP.sub(with_title, buf.getvalue())
    path.write_text(svg)
    return len(EP_GROUP.findall(svg))


def save_resilient(fig, path, fallback_factory):
    """Save one figure without allowing LaTeX or renderer failure to abort the site build."""
    first_error = None
    try:
        return write_svg(fig, path), None
    except Exception as ex:
        first_error = ex
        traceback.print_exc()
    finally:
        plt.close(fig)

    with matplotlib.rc_context({"text.usetex": False}):
        fallback = fallback_factory()
        try:
            return write_svg(fallback, path), None
        except Exception:
            traceback.print_exc()
        finally:
            plt.close(fallback)

    # Last resort: keep the build and HTML references intact with a dependency-free SVG.
    label = html.escape(f"Figure unavailable: {type(first_error).__name__}")
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 440">'
        '<rect width="920" height="440" fill="white"/>'
        f'<text x="460" y="220" text-anchor="middle" fill="#666">{label}</text></svg>')
    return 0, first_error


def main():
    ap = argparse.ArgumentParser(
        description="Draw dist/fig/{q1..q8,m-*,exp-*}.svg + data/fig-meta.json "
                    "from the derived site data.")
    ap.add_argument("--data-dir", default=str(ROOT / "site" / "data"),
                    help="derived JSON dir (summary.json, records.json)")
    ap.add_argument("--out", default=str(ROOT / "site" / "dist" / "fig"),
                    help="SVG output dir")
    args = ap.parse_args()

    data = Path(args.data_dir)
    summary = json.loads((data / "summary.json").read_text())
    records_path = data / "records.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []

    init_style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_meta = {}

    for name, build in FIGURES.items():
        qq = (summary.get("questions") or {}).get(name) or {}
        fallback_factory = lambda needs=qq.get("needs") or "figure renderer unavailable": \
            placeholder(needs, summary)
        try:
            fig, meta = build(summary, records)
            status = "real"
            fallback_factory = lambda build=build: build(summary, records)[0]
        except Unmeasured as ex:
            fig, status = placeholder(str(ex), summary), f"placeholder — needs {ex}"
            meta = {"finding": f"Not yet measured — needs {ex}.",
                    "howto": "", "fineprint": ""}
        except Exception as ex:  # a missing/misshapen stat must not kill the build
            traceback.print_exc()
            fig = placeholder(qq.get("needs") or f"build failed: {type(ex).__name__}",
                              summary)
            status = f"placeholder — build failed: {ex}"
            meta = {"finding": f"Figure build failed ({type(ex).__name__}).",
                    "howto": "", "fineprint": ""}
        n_links, save_error = save_resilient(
            fig, out / f"{name}.svg", fallback_factory)
        if save_error is not None:
            status = f"placeholder — save failed: {type(save_error).__name__}"
            meta = {"finding": f"Figure rendering failed ({type(save_error).__name__}).",
                    "howto": "", "fineprint": ""}
        all_meta[name] = meta
        print(f"{name}.svg  {status}" + (f"  [{n_links} episode links]" if n_links else ""))

    exp_path = data / "experiments.json"
    experiments = (json.loads(exp_path.read_text()).get("experiments") or []
                   if exp_path.exists() else [])
    by_id = {r["id"]: r for r in records}
    for exp in experiments:
        fallback_factory = lambda exp=exp: thumb_planned(exp)
        try:
            fig, meta, status = fig_exp_thumb(exp, by_id)
            fallback_factory = lambda exp=exp: fig_exp_thumb(exp, by_id)[0]
        except Exception as ex:  # a misshapen entry must not kill the build
            traceback.print_exc()
            fig, status = thumb_planned(exp), f"placeholder — build failed: {ex}"
            meta = {"finding": f"Thumbnail build failed ({type(ex).__name__}).",
                    "howto": "", "fineprint": ""}
        n_links, save_error = save_resilient(
            fig, out / f"exp-{exp['id']}.svg", fallback_factory)
        if save_error is not None:
            status = f"placeholder — save failed: {type(save_error).__name__}"
            meta = {"finding": f"Thumbnail rendering failed ({type(save_error).__name__}).",
                    "howto": "", "fineprint": ""}
        all_meta[f"exp-{exp['id']}"] = meta
        print(f"exp-{exp['id']}.svg  {status}"
              + (f"  [{n_links} episode links]" if n_links else ""))

    meta_path = data / "fig-meta.json"
    meta_path.write_text(json.dumps(all_meta, indent=1, ensure_ascii=False) + "\n")
    print(f"fig-meta.json  {len(all_meta)} entries")


if __name__ == "__main__":
    main()
