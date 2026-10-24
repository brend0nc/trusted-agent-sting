"""Render the results site: templates/ + content/*.md + data/ -> dist/.

Reads data-dir/summary.json (defensively — any field may be missing or null; missing
values render as an em-dash, never crash), resolves {{dotted.path}} or
{{dotted.path:.3f}} placeholders in the content markdown against it, converts the
markdown to HTML, and writes index.html (+ episodes.html when its template exists),
one figure detail page fig/qN.html per question (figure, how-to-read, prose,
stats, per-episode table, downloads — all server-rendered), one experiment detail
page experiments/<id>.html per experiments.json entry, and prompts.html from
prompts.json. index.html additionally carries the findings-at-a-glance tile grid,
the experiments gallery, and the advance-metrics preview (m-*.svg figures paired
with content/metrics.md subsections). Figure SVGs that are not built yet are
skipped cleanly and appear on the next render.
Per-figure text (SITE.md item 4, rev. 2026-07-03b) comes from data-dir/fig-meta.json
({id: {finding, howto, fineprint}}, written by figures.py) and is typeset in page
HTML wherever a figure is placed: the finding above the SVG, the how-to-read
sentence directly under it, the fine print merged into the figcaption. Missing
meta degrades to the figure alone — never a crash.
Copies static/ and the JSON that the transcript reader and download links fetch.

Usage:
  python site/render.py [--data-dir site/data] [--out site/dist]
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import quote

import markdown
from jinja2 import Environment, FileSystemLoader

SITE = Path(__file__).parent
DASH = "—"  # em-dash: the one honest rendering of a missing number


def warn(msg):
    print(f"render.py: warning: {msg}", file=sys.stderr)


# --- formatting -----------------------------------------------------------------------------------

def fmt(v):
    """One value -> display string: floats to 3 significant figures, missing -> em-dash."""
    if v is None or v == "":
        return DASH
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return DASH if v != v else f"{v:.3g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(fmt(x) for x in v) or DASH
    return str(v)


def fmt_ci(ci):
    """[lo, hi] -> "[lo, hi]"; anything malformed or missing -> em-dash."""
    if not isinstance(ci, (list, tuple)) or len(ci) != 2 or ci[0] is None or ci[1] is None:
        return DASH
    return f"[{fmt(ci[0])}, {fmt(ci[1])}]"


def short_sha(commit):
    return str(commit)[:7] if commit else DASH


def short_date(v, n=10):
    """ISO-ish timestamp -> its first n characters ('2026-07-02'); em-dash when missing."""
    return str(v)[:n].replace("T", " ") if v else DASH


# --- placeholder resolution against summary.json ---------------------------------------------------

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_][A-Za-z0-9_.]*)\s*(?::([^{}]+))?\}\}")


def resolve(path, data):
    """Walk a dotted path ('questions.q1.net_capture_mean') through dicts/lists; None if absent."""
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, (list, tuple)) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
    return cur


def fill_placeholders(text, summary, source):
    """Replace {{dotted.path}} / {{dotted.path:.3f}} with summary values; unresolvable -> em-dash."""
    def sub(m):
        path, spec = m.group(1), m.group(2)
        val = resolve(path, summary)
        if val is None:
            warn(f"{source}: unresolved placeholder {{{{{path}}}}}")
            return DASH
        if spec:
            try:
                return format(val, spec)
            except (ValueError, TypeError):
                warn(f"{source}: bad format spec {spec!r} for {{{{{path}}}}}")
        return fmt(val)
    return PLACEHOLDER.sub(sub, text)


# --- content markdown ------------------------------------------------------------------------------

CAPTION = re.compile(r"<!--\s*caption:\s*(.*?)\s*-->", re.S)


def load_content(path, summary):
    """One content .md -> {title, html, caption}; None if the file is missing."""
    if not path.exists():
        warn(f"missing content file: {path}")
        return None
    text = path.read_text()
    m = CAPTION.search(text)
    caption = fill_placeholders(m.group(1), summary, path.name) if m else None
    text = CAPTION.sub("", text)
    title, lines = None, text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("# "):
            title = line[2:].strip()
            lines = lines[:i] + lines[i + 1:]
            break
    body = fill_placeholders("\n".join(lines), summary, path.name)
    return {"title": title or path.stem, "html": markdown.markdown(body, extensions=["tables"]),
            "caption": caption}


def fig_text(fmeta, fig_id):
    """fig-meta.json entry for one figure id -> {finding, howto, fineprint}, each a
    stripped string or None. Defensive: a missing file (fmeta {}), unknown id, or
    malformed entry yields three Nones — the figure then renders without the text
    block."""
    entry = fmeta.get(fig_id) if isinstance(fmeta, dict) else None
    if not isinstance(entry, dict):
        entry = {}
    out = {}
    for key in ("finding", "howto", "fineprint"):
        v = entry.get(key)
        out[key] = (str(v).strip() or None) if v is not None else None
    return out


def merge_caption(fineprint, caption):
    """Fine print + the hand-written decode caption -> one figcaption string, fine
    print first, ' · ' separated. None when both are absent (figcaption collapses)."""
    return " · ".join(x for x in (fineprint, caption) if x) or None


def inline_svg(path):
    """Figure SVG -> markup inlined into the page (so gid-tagged marks are addressable
    by fig-links.js): everything before the root <svg> tag (xml prolog, doctype) is
    stripped. Our own build artifact — not sanitized beyond that. None on any problem."""
    try:
        text = path.read_text()
    except OSError as e:
        warn(f"unreadable figure {path}: {e}")
        return None
    i = text.find("<svg")
    if i == -1:
        warn(f"{path}: no <svg> root found")
        return None
    return text[i:]


def question_sections(content_dir, summary, out_dir, fmeta):
    """content/questions/q*.md (sorted) -> section dicts; the figure only when its
    SVG exists. Finding / how-to-read come from fig-meta.json (None when absent);
    the fine print is merged into the caption, ahead of the decode text."""
    sections = []
    for path in sorted((content_dir / "questions").glob("q*.md")):
        m = re.match(r"q(\d+)", path.stem)
        number = int(m.group(1)) if m else None
        doc = load_content(path, summary)
        if doc is None:
            continue
        fig = None
        if number is not None and (out_dir / "fig" / f"q{number}.svg").exists():
            fig = inline_svg(out_dir / "fig" / f"q{number}.svg")
        ft = fig_text(fmeta, f"q{number}" if number is not None else path.stem)
        sections.append({"id": f"q{number}" if number is not None else path.stem,
                         "number": number if number is not None else "?",
                         "title": doc["title"], "html": doc["html"],
                         "finding": ft["finding"], "howto": ft["howto"],
                         "caption": merge_caption(ft["fineprint"], doc["caption"]),
                         "figure_svg": fig,
                         "detail_href": f"fig/q{number}.html" if number is not None else None})
    return sections


# --- figure detail pages (fig/qN.html) --------------------------------------------------------------

# site-root-relative targets that appear in content HTML; detail pages live one
# level down (fig/), so these get the page's root prefix prepended.
ROOT_LINK = re.compile(r'\b(href|src)="(episodes\.html|index\.html|data/|fig/|static/|#)')


def relocate_html(html, prefix):
    """Rewrite site-root-relative links in rendered content HTML for a page under fig/.
    Bare '#anchor' links point at index.html sections, so they become index.html#anchor."""
    def sub(m):
        target = m.group(2)
        if target == "#":
            return f'{m.group(1)}="{prefix}index.html#'
        return f'{m.group(1)}="{prefix}{target}'
    return ROOT_LINK.sub(sub, html)


def flatten_stats(obj, prefix=""):
    """summary.questions.qN (any shape) -> [{key, value}] rows, keys dotted, values
    pre-formatted (CIs as "[lo, hi]", missing as em-dash). Defensive: never raises."""
    if obj is None:
        return [] if not prefix else [{"key": prefix, "value": DASH}]
    if isinstance(obj, dict):
        rows = []
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if str(k) == "ci":
                rows.append({"key": key, "value": fmt_ci(v)})
            elif isinstance(v, (dict, list, tuple)):
                rows += flatten_stats(v, key)
            else:
                rows.append({"key": key, "value": fmt(v)})
        return rows
    if isinstance(obj, (list, tuple)):
        rows = []
        for i, value in enumerate(obj):
            key = f"{prefix}.{i}" if prefix else str(i)
            if isinstance(value, (dict, list, tuple)):
                rows += flatten_stats(value, key)
            else:
                rows.append({"key": key, "value": fmt(value)})
        return rows
    return [{"key": prefix or "value", "value": fmt(obj)}]


def episode_rows(records, ids=None):
    """Loaded records.json -> one pre-formatted row per episode for the detail-page
    tables, sorted by condition then id, links relative to a page one level below
    the site root (fig/ or experiments/). ids, when given, filters to those episode
    ids (an empty list means no rows). [] when no data."""
    if not isinstance(records, list):
        return []
    keep = None if ids is None else {str(i) for i in ids}
    rows = []
    for r in records:
        if not isinstance(r, dict):
            continue
        rid = r.get("id")
        if keep is not None and str(rid) not in keep:
            continue
        model = r.get("focal_model")
        rows.append({"id": fmt(rid),
                     "href": f"../episodes.html?e={quote(str(rid))}" if rid else None,
                     "condition": fmt(r.get("condition")),
                     "model": fmt(str(model).split("/")[-1] if model else model),
                     "by_focal": fmt(r.get("by_focal")),
                     "cascades": fmt(r.get("cascades")),
                     "gini": fmt(r.get("gini"))})
    rows.sort(key=lambda x: (x["condition"], x["id"]))
    return rows


# --- table rows (pre-formatted strings; templates stay dumb) ----------------------------------------

def coverage_rows(summary):
    rows = []
    for c in summary.get("coverage") or []:
        done, planned = c.get("done_episodes"), c.get("planned_episodes")
        bar_pct = None
        if isinstance(done, (int, float)) and isinstance(planned, (int, float)) and planned > 0:
            bar_pct = f"{max(0.0, min(1.0, done / planned)) * 100:.1f}"
        selected = c.get("selected_instrument")
        if c.get("served_model_drift"):
            instrument = "blocked: served-model drift"
        elif c.get("instrument_ambiguous"):
            instrument = "blocked: ambiguous"
        elif selected == "current":
            instrument = "current"
        elif isinstance(selected, str) and selected.startswith("legacy-"):
            instrument = "historical (recovered)"
        elif selected is None:
            instrument = "not selected"
        else:
            instrument = fmt(selected)
        served = c.get("served_instruments") or {}
        if c.get("served_model_drift"):
            provenance = "model-version mix (blocked)"
        elif c.get("provider_drift"):
            provenance = "provider mix (audit)"
        elif served:
            provenance = "recorded"
        else:
            provenance = "not recorded"
        rows.append({"arm": fmt(c.get("arm")), "task": fmt(c.get("task")),
                     "config": fmt(c.get("config")), "stage": c.get("stage") or "",
                     "conditions": fmt(c.get("conditions")),
                     "done": fmt(done), "planned": fmt(planned), "pct": bar_pct,
                     "instrument": instrument, "provenance": provenance,
                     "analysis": fmt(c.get("analysis_episodes")),
                     "duplicates": fmt(c.get("duplicate_episodes"))})
    return rows


# coverage rows split by planned stage: confirmatory (paper) experiments first, then
# exploratory (test/calibration); unknown stages trail. (config, label) per group.
COVERAGE_STAGES = (("confirmatory", "Confirmatory (paper)"),
                   ("exploratory", "Test & calibration"))


def coverage_groups(rows):
    """coverage_rows -> [{stage, label, rows}] in fixed order; empty groups dropped,
    unknown stages land in a trailing 'Other' group."""
    groups = []
    for stage, label in COVERAGE_STAGES:
        stage_rows = [r for r in rows if r["stage"] == stage]
        if stage_rows:
            groups.append({"stage": stage, "label": label, "rows": stage_rows})
    known = {s for s, _ in COVERAGE_STAGES}
    leftover = [r for r in rows if r["stage"] not in known]
    if leftover:
        groups.append({"stage": "other", "label": "Other", "rows": leftover})
    return groups


# --- data status: paper vs test/exploratory pools (SITE.md — the whole point) -----------------------

def data_status(summary):
    """summary.pools -> the data-status strip context: paper episode count + its run
    names, the test/exploratory count, and the in-progress count. None when pools is
    absent (strip is skipped). Also returns paper_note — the one-line paper-pool
    provenance stamped once in the results area."""
    pools = summary.get("pools")
    if not isinstance(pools, dict):
        return None

    def pool(key):
        p = pools.get(key)
        return p if isinstance(p, dict) else {}

    paper, in_prog, expl, held = (pool("paper"), pool("in_progress"),
                                  pool("exploratory"), pool("held_out"))
    paper_runs = [str(r) for r in paper.get("runs") or []]
    paper_n = paper.get("n_episodes")
    # test = exploratory + in-progress + held-out: everything not a confirmed paper result
    test_n = sum(x for x in (expl.get("n_episodes"), in_prog.get("n_episodes"),
                             held.get("n_episodes"))
                 if isinstance(x, (int, float)))
    held_n = held.get("n_episodes")
    runs_str = ", ".join(paper_runs) or DASH
    note = f"Paper pool: {runs_str}, n={fmt(paper_n)}" if paper_runs else None
    return {"paper_n": fmt(paper_n), "paper_runs": runs_str,
            "in_progress_n": fmt(in_prog.get("n_episodes")),
            "exploratory_n": fmt(expl.get("n_episodes")),
            "held_out_n": fmt(held_n),
            "has_held_out": isinstance(held_n, (int, float)) and held_n > 0,
            "test_n": fmt(test_n), "paper_note": note}


def run_rows(runs):
    rows = []
    for r in runs or []:
        eval_file = r.get("eval_file")
        rows.append({"name": fmt(r.get("name")), "task": fmt(r.get("task")),
                     "arm": fmt(r.get("arm")), "conditions": fmt(r.get("conditions")),
                     "n": fmt(r.get("n_episodes")),
                     "commit": short_sha(r.get("commit")),
                     "started": short_date(r.get("started"), 16),
                     "eval_file": Path(str(eval_file)).name if eval_file else DASH})
    return rows


def pct(v):
    """A 0-1 rate -> "24.3%"; missing -> em-dash. For the extraction leaderboard."""
    if not isinstance(v, (int, float)) or v != v:
        return DASH
    return f"{v * 100:.1f}%"


def pct_ci(ci):
    """[lo, hi] rates -> "[20.7%, 25.8%]"; malformed/missing -> em-dash."""
    if not isinstance(ci, (list, tuple)) or len(ci) != 2 or ci[0] is None or ci[1] is None:
        return DASH
    return f"[{pct(ci[0])}, {pct(ci[1])}]"


def leaderboard_rows(summary, key="leaderboard"):
    """summary[key] (arena schema) -> pre-formatted rows: extraction is the
    headline (share of rivals' actions won, as a %), with rank / model / n / CI /
    raw points. The focal-only columns (by_focal etc.) are null in the arena and
    dropped from this table entirely. key="pilot_leaderboard" renders the same
    table over the non-paper pool for the pilot page."""
    rows = []
    for r in summary.get(key) or []:
        model = r.get("model")  # old calibration seat maps carry full "openrouter/..." specs
        rows.append({"rank": fmt(r.get("rank_ub")),
                     "model": fmt(str(model).split("/")[-1] if model else model),
                     "n": fmt(r.get("n")),
                     "extraction": pct(r.get("extraction")),
                     "ci": pct_ci(r.get("ci")),
                     "points": fmt(r.get("points"))})
    return rows


def load_json(path):
    """Parse a JSON file; None (with a warning) when missing or unreadable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        warn(f"unreadable {path}: {e}")
        return None


def featured_rows(data_dir):
    """records.json -> the 3 most cascade-rich PAPER-pool episodes (ties: more events,
    then id), each with a reader deep link and a one-line why-look note. [] when no
    data. Paper pool only: this is a headline display, so smoke/calibration/held-out
    episodes are barred here even though the reader can browse them."""
    records = load_json(data_dir / "records.json")
    if not isinstance(records, list):
        return []
    n_events = {}
    for e in load_json(data_dir / "episodes_index.json") or []:
        if isinstance(e, dict) and e.get("id"):
            n_events[e["id"]] = e.get("n_events") or 0
    rows = [r for r in records
            if isinstance(r, dict) and r.get("id") and isinstance(r.get("cascades"), (int, float))
            and r["cascades"] > 0 and r.get("pool") == "paper"]
    rows.sort(key=lambda r: (-r["cascades"], -n_events.get(r["id"], 0), r["id"]))
    ordinal = ("most", "second-most", "third-most")
    out = []
    for i, r in enumerate(rows[:3]):
        why = f"{ordinal[i]} cascade-rich episode so far"
        if n_events.get(r["id"]):
            why += f", {n_events[r['id']]} events"
        out.append({"id": r["id"], "href": f"episodes.html?e={quote(r['id'])}",
                    "condition": fmt(r.get("condition")), "cascades": fmt(r.get("cascades")),
                    "why": why})
    return out


def condition_rows(summary):
    keys = ("condition", "n", "cascades", "cascade_null", "brokered_pull_rate",
            "credit_payback", "payback_rate", "gini")
    return [{k: fmt(r.get(k)) for k in keys} for r in summary.get("by_condition") or []]


# --- tile grids: findings at a glance + experiments gallery (SITE.md item 4b) -----------------------

def question_tiles(summary, out_dir, fmeta):
    """summary.questions -> findings-at-a-glance tiles: the qN figure inlined as a
    thumbnail, a Question N label, the finding (fig-meta.json) as the one-line
    subtitle, and a measured/needs status. Thumbnails carry no how-to-read or fine
    print. Tiles link to the fig/qN.html detail page."""
    questions = summary.get("questions")
    if not isinstance(questions, dict):
        return []
    tiles = []
    for key in sorted(questions, key=lambda k: (len(k), k)):
        m = re.fullmatch(r"q(\d+)", key)
        if not m:
            continue
        q = questions[key] if isinstance(questions[key], dict) else {}
        svg_path = out_dir / "fig" / f"{key}.svg"
        if q.get("measured"):
            status = "measured"
        elif q.get("needs"):
            status = f"not yet measured — needs {q['needs']}"
        else:
            status = "not yet measured"
        tiles.append({"id": key, "number": int(m.group(1)), "href": f"fig/{key}.html",
                      "svg": inline_svg(svg_path) if svg_path.exists() else None,
                      "finding": fig_text(fmeta, key)["finding"],
                      "status": status})
    return tiles


EXPERIMENT_GROUPS = (("Exploratory runs", ("exploratory",)),
                     ("Completed", ("done",)),
                     ("In progress", ("partial",)),
                     ("Planned", ("planned",)))
# gallery tiles repeat the experiment name as their title, so the blurb's
# "Experiment N — name:" prefix is dropped for the one-line subtitle
BLURB_PREFIX = re.compile(r"^Experiment\s+\d+\s+—\s+[^:]*:\s*")


def experiment_entries(data_dir, out_dir, fmeta):
    """experiments.json -> one entry per experiment: gallery-tile fields plus
    everything the experiments/<id>.html detail page needs. Thumbnail SVGs
    (fig/exp-<id>.svg) that are not built yet are skipped cleanly. fig-meta.json
    text (id exp-<id>): the finding stands in for a missing blurb on the tile and
    heads the detail-page figure, where howto/fineprint also render."""
    data = load_json(data_dir / "experiments.json")
    entries = []
    for e in (data or {}).get("experiments") or []:
        if not isinstance(e, dict) or not e.get("id"):
            continue
        eid = str(e["id"])
        svg_path = out_dir / "fig" / f"exp-{eid}.svg"
        ft = fig_text(fmeta, f"exp-{eid}")
        done, planned = e.get("done_episodes"), e.get("planned_episodes")
        if isinstance(planned, (int, float)):
            status_line = f"{fmt(done)} / {fmt(planned)} episodes"
        else:
            status_line = f"{fmt(done)} episodes"
        blurb = str(e.get("blurb") or "")
        sub = BLURB_PREFIX.sub("", blurb)
        sub = (sub[:1].upper() + sub[1:]) if sub else blurb
        if not sub:
            sub = ft["finding"] or ""
        config = f"configs/{eid}.yaml"
        entries.append({
            "id": eid,
            "name": fmt(e.get("name") or eid),
            "blurb": sub,
            "finding": ft["finding"], "howto": ft["howto"], "fineprint": ft["fineprint"],
            "description": str(e.get("description") or blurb),
            "status": str(e.get("status") or "planned"),
            "status_line": status_line,
            "questions": [str(q) for q in e.get("questions") or []
                          if re.fullmatch(r"q\d+", str(q))],
            "conditions": fmt(e.get("conditions")),
            "episode_ids": [str(i) for i in e.get("episode_ids") or []],
            "href": f"experiments/{eid}.html",
            "svg": inline_svg(svg_path) if svg_path.exists() else None,
            "config": config if (SITE.parent / config).exists() else None,
        })
    return entries


def experiment_groups(entries):
    """Entries -> gallery groups in fixed order (exploratory / completed / in progress / planned);
    empty groups are dropped, unknown statuses land in a trailing group."""
    groups = []
    for label, statuses in EXPERIMENT_GROUPS:
        tiles = [e for e in entries if e["status"] in statuses]
        if tiles:
            groups.append({"label": label, "tiles": tiles})
    known = {s for _, statuses in EXPERIMENT_GROUPS for s in statuses}
    leftover = [e for e in entries if e["status"] not in known]
    if leftover:
        groups.append({"label": "Other", "tiles": leftover})
    return groups


# --- advance-metrics preview (index #metrics): m-*.svg + content/metrics.md ------------------------

# (figure id, fallback heading, keywords that match a metrics.md "## " heading)
METRIC_FIGURES = (
    ("m-coalition", "Coalitions", ("coalition",)),
    ("m-nonrecip", "Non-reciprocity", ("recip", "repaid", "repay", "unpaid", "favour", "favor")),
    ("m-chat", "Public chat", ("chat", "public")),
)
# figure id -> summary.metrics_preview key, for the data-driven fallback note
METRIC_PREVIEW_KEYS = {"m-coalition": "coalition", "m-nonrecip": "nonreciprocity",
                       "m-chat": "public_chat"}
H2_LINE = re.compile(r"^##\s+(.*)$")


def split_subsections(text):
    """Markdown body -> (preamble, [(heading, body)]) split at '## ' headings."""
    pre, subs, cur = [], [], None
    for line in text.splitlines():
        m = H2_LINE.match(line)
        if m:
            cur = []
            subs.append((m.group(1).strip(), cur))
        elif cur is None:
            pre.append(line)
        else:
            cur.append(line)
    return "\n".join(pre).strip(), [(h, "\n".join(body).strip()) for h, body in subs]


def metric_fallback(summary, fig_id):
    """summary.metrics_preview -> one grey plain-text note for a metric whose
    hand-written reading (content/metrics.md subsection) does not exist yet:
    the definition plus every caveat, verbatim from the derived data."""
    mp = resolve(f"metrics_preview.{METRIC_PREVIEW_KEYS.get(fig_id, '')}", summary)
    if not isinstance(mp, dict):
        return None
    parts = []
    if mp.get("definition"):
        parts.append(str(mp["definition"]))
    for c in mp.get("caveats") or []:
        parts.append(str(c))
    if mp.get("volume_caveat"):
        note = str(mp["volume_caveat"])
        parts.append(note[:1].upper() + note[1:] + ".")
    if mp.get("measured") is False and mp.get("needs"):
        parts.append(f"Not yet measured — waiting on {mp['needs']}.")
    return " ".join(parts) or None


def metrics_section(content_dir, summary, out_dir, fmeta):
    """content/metrics.md (subsections at '## ' headings, matched to the m-* figures
    by heading keyword) + dist/fig/m-*.svg -> the #metrics section context. Each
    figure also carries its fig-meta.json text (finding / howto / fineprint, None
    when absent). Missing pieces degrade: no SVG -> prose only; no matching
    subsection -> the figure with a data-driven fallback note; nothing at all ->
    None (section and TOC entry skipped)."""
    path = content_dir / "metrics.md"
    title, preamble_html, subs = "Advance metrics", None, []
    if path.exists():
        text = CAPTION.sub("", path.read_text())
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("# "):
                title = line[2:].strip()
                lines = lines[:i] + lines[i + 1:]
                break
        pre, subs = split_subsections("\n".join(lines))
        if pre:
            preamble_html = markdown.markdown(fill_placeholders(pre, summary, path.name),
                                              extensions=["tables"])
    else:
        warn(f"missing content file: {path}")

    taken = [False] * len(subs)
    items = []
    for fig_id, fallback_title, keywords in METRIC_FIGURES:
        svg_path = out_dir / "fig" / f"{fig_id}.svg"
        svg = inline_svg(svg_path) if svg_path.exists() else None
        heading, html = fallback_title, None
        for i, (sub_heading, body) in enumerate(subs):
            if not taken[i] and any(k in sub_heading.lower() for k in keywords):
                taken[i] = True
                heading = sub_heading
                if body:
                    html = markdown.markdown(fill_placeholders(body, summary, path.name),
                                             extensions=["tables"])
                break
        note = None if html else metric_fallback(summary, fig_id)
        if svg is None and html is None and note is None:
            continue
        ft = fig_text(fmeta, fig_id)
        items.append({"id": fig_id, "title": heading, "svg": svg, "html": html, "note": note,
                      "finding": ft["finding"], "howto": ft["howto"],
                      "fineprint": ft["fineprint"]})
    # subsections that matched no figure still render, in file order, after the figures
    extra = [{"title": h, "html": markdown.markdown(fill_placeholders(b, summary, path.name),
                                                    extensions=["tables"])}
             for (h, b), t in zip(subs, taken) if not t]

    if not items and not extra and preamble_html is None:
        return None
    # key is "figures", not "items": metrics.items in Jinja would hit dict.items
    return {"title": title, "preamble": preamble_html, "figures": items, "extra": extra,
            "pending": not path.exists()}


# --- prompts page (prompts.html) --------------------------------------------------------------------

PROMPT_GROUP_LABELS = {"system": "System prompts", "turn": "The turn message",
                       "judge": "Judges"}


def prompts_context(data_dir):
    """prompts.json -> grouped sections for prompts.html (group order = first
    appearance; known groups get their display label) + provenance. None when the
    file is missing or unreadable — the page still renders, with an empty note."""
    data = load_json(data_dir / "prompts.json")
    if not isinstance(data, dict):
        return None
    groups, by_group = [], {}
    for s in data.get("sections") or []:
        if not isinstance(s, dict) or not s.get("text"):
            continue
        g = str(s.get("group") or "other")
        if g not in by_group:
            by_group[g] = {"label": PROMPT_GROUP_LABELS.get(g, g.capitalize()),
                           "sections": []}
            groups.append(by_group[g])
        by_group[g]["sections"].append({"id": str(s.get("id") or ""),
                                        "title": s.get("title") or s.get("id"),
                                        "note": s.get("note"), "text": s.get("text")})
    return {"groups": groups, "commit": short_sha(data.get("git_commit")),
            "dirty": bool(data.get("git_dirty"))}


# --- build ------------------------------------------------------------------------------------------

def build_context(summary, content_dir, data_dir, out_dir):
    fmeta = load_json(data_dir / "fig-meta.json")
    if not isinstance(fmeta, dict):
        if fmeta is not None:
            warn(f"{data_dir / 'fig-meta.json'} is not a JSON object")
        else:
            warn(f"missing {data_dir / 'fig-meta.json'} — figures render without "
                 "finding/how-to-read text")
        fmeta = {}
    intro = load_content(content_dir / "intro.md", summary)
    methods = load_content(content_dir / "methods.md", summary)
    next_up = load_content(content_dir / "next.md", summary)
    status = data_status(summary)
    paper_note = status["paper_note"] if status else None
    questions = question_sections(content_dir, summary, out_dir, fmeta)
    featured = featured_rows(data_dir)
    findings = question_tiles(summary, out_dir, fmeta)
    experiments = experiment_entries(data_dir, out_dir, fmeta)
    exp_groups = experiment_groups(experiments)
    metrics = metrics_section(content_dir, summary, out_dir, fmeta)

    toc = []
    if intro:
        toc.append({"id": "intro", "label": intro["title"]})
    if status:
        toc.append({"id": "data-status", "label": "Results status"})
    if findings:
        toc.append({"id": "findings", "label": "Findings at a glance"})
    if exp_groups or featured:
        toc.append({"id": "experiments", "label": "Experiments"})
    toc.append({"id": "project-state", "label": "Project state"})
    if methods:
        toc.append({"id": "methods", "label": methods["title"]})
    toc += [{"id": q["id"], "label": f"Q{q['number']} · {q['title']}"} for q in questions]
    if metrics:
        toc.append({"id": "metrics", "label": "Advance metrics"})
    toc.append({"id": "leaderboard", "label": "Leaderboard"})
    toc.append({"id": "conditions", "label": "Conditions"})
    if next_up:
        toc.append({"id": "next", "label": next_up["title"]})

    return {
        "meta": {"commit": short_sha(summary.get("git_commit")),
                 "generated": short_date(summary.get("generated_at")),
                 "n_episodes": fmt(summary.get("n_episodes"))},
        "intro": intro, "methods": methods, "next_up": next_up,
        "data_status": status, "paper_note": paper_note,
        "questions": questions, "toc": toc, "featured": featured,
        "findings": findings, "experiments": experiments,
        "experiment_groups": exp_groups, "metrics": metrics,
        "coverage_groups": coverage_groups(coverage_rows(summary)),
        "exploratory_runs": run_rows(summary.get("exploratory_runs")),
        "runs": run_rows(summary.get("runs")),
        "leaderboard": leaderboard_rows(summary),
        "conditions": condition_rows(summary),
        "summary": summary,
    }


def copy_assets(data_dir, out_dir):
    """static/ -> out/static; the reader's episode JSON and the detail pages'
    summary.json download -> out/data. out/fig SVGs are left in place."""
    static = SITE / "static"
    if static.exists():
        shutil.copytree(static, out_dir / "static", dirs_exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    for name in ("episodes_index.json", "summary.json"):
        src = data_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / "data" / name)
    episodes = data_dir / "episodes"
    if episodes.is_dir():
        shutil.copytree(episodes, out_dir / "data" / "episodes", dirs_exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the results site to static HTML.")
    parser.add_argument("--data-dir", default=str(SITE / "data"), help="derived JSON dir")
    parser.add_argument("--content-dir", default=str(SITE / "content"), help="hand-written markdown dir")
    parser.add_argument("--out", default=str(SITE / "dist"), help="output dir")
    args = parser.parse_args()

    data_dir, content_dir, out_dir = Path(args.data_dir), Path(args.content_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, summary_path = {}, data_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            warn(f"unreadable {summary_path}: {e}")
    else:
        warn(f"missing {summary_path}")
    if not isinstance(summary, dict):
        warn(f"{summary_path} is not a JSON object")
        summary = {}

    context = build_context(summary, content_dir, data_dir, out_dir)
    env = Environment(loader=FileSystemLoader(SITE / "templates"), autoescape=True)
    (out_dir / "index.html").write_text(env.get_template("index.html.j2").render(**context))
    print(f"render.py: wrote {out_dir / 'index.html'}")
    if (SITE / "templates" / "episodes.html.j2").exists():
        (out_dir / "episodes.html").write_text(env.get_template("episodes.html.j2").render(**context))
        print(f"render.py: wrote {out_dir / 'episodes.html'}")

    # one detail page per question: fig/qN.html, next to the qN.svg it inlines
    fig_dir = out_dir / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_template = env.get_template("figure.html.j2")
    records = load_json(data_dir / "records.json")
    episodes = episode_rows(records)
    n_pages = 0
    for q in context["questions"]:
        if not isinstance(q.get("number"), int):
            continue
        page_q = dict(q, html=relocate_html(q["html"], "../"))
        stats = flatten_stats(resolve(f"questions.q{q['number']}", summary))
        (fig_dir / f"q{q['number']}.html").write_text(
            fig_template.render(q=page_q, stats=stats, episodes=episodes,
                                meta=context["meta"], root="../"))
        n_pages += 1
    print(f"render.py: wrote {n_pages} figure detail pages in {fig_dir}")

    # one detail page per experiment: experiments/<id>.html, episodes joined by id
    exp_dir = out_dir / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_template = env.get_template("experiment.html.j2")
    for e in context["experiments"]:
        (exp_dir / f"{e['id']}.html").write_text(
            exp_template.render(e=e, episodes=episode_rows(records, e["episode_ids"]),
                                meta=context["meta"], root="../"))
    print(f"render.py: wrote {len(context['experiments'])} experiment pages in {exp_dir}")

    # prompts.html: what the agents see, from prompts.json (always written — the
    # top nav links here from every page; missing data renders as an empty note)
    (out_dir / "prompts.html").write_text(
        env.get_template("prompts.html.j2").render(prompts=prompts_context(data_dir),
                                                   meta=context["meta"], root=""))
    print(f"render.py: wrote {out_dir / 'prompts.html'}")

    # pilot.html: every non-paper run — quarantined pilot/smoke/calibration data with
    # its own leaderboard, coverage, and the full eval-file provenance index
    (out_dir / "pilot.html").write_text(
        env.get_template("pilot.html.j2").render(
            pilot_content=load_content(content_dir / "pilot.md", summary),
            pilot_leaderboard=leaderboard_rows(summary, key="pilot_leaderboard"),
            coverage_groups=context["coverage_groups"],
            exploratory_runs=context["exploratory_runs"],
            runs=context["runs"], data_status=context["data_status"],
            meta=context["meta"], root=""))
    print(f"render.py: wrote {out_dir / 'pilot.html'}")

    copy_assets(data_dir, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
