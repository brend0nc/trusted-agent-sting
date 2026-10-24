"""Run the real reader in headless Chrome. No paid calls or browser packages are needed."""
import html
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import jinja2
import pytest

ROOT = Path(__file__).resolve().parents[1]
CHROME = os.environ.get("CHROME_BIN") or shutil.which("chromium") or shutil.which("google-chrome")
if not CHROME:
    cached = sorted((Path.home() / "Library/Caches/ms-playwright").glob(
        "chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell"))
    CHROME = str(cached[-1]) if cached else None


@pytest.mark.parametrize("free_reports", [False, True])
def test_agent_reader_shows_actions_evidence_and_unknown_costs(tmp_path, free_reports):
    if not CHROME:
        pytest.skip("Chrome is required for the reader browser test")
    env = jinja2.Environment(loader=jinja2.ChoiceLoader([
        jinja2.DictLoader({"base.html.j2": "<!doctype html><html><head><meta charset='utf-8'>{% block head %}{% endblock %}</head>"
                          "<body>{% block content %}{% endblock %}</body></html>"}),
        jinja2.FileSystemLoader(ROOT / "site/templates"),
    ]))
    page = env.get_template("episodes.html.j2").render()
    (tmp_path / "static").mkdir()
    for name in ("reader.js", "reader.css"):
        shutil.copy(ROOT / "site/static" / name, tmp_path / "static" / name)
    (tmp_path / "data/episodes").mkdir(parents=True)
    rows = [
        {"actor": "P1", "type": "pull", "beneficiary": "P2", "scores_after": {"P1": 0, "P2": 1, "P3": 0},
         "calls": [{"call_id": "paid-1", "input": [{"role": "user", "content": "Exact saved observation <safe>"}],
                    "response": "Exact saved reply", "usage": {"input_tokens": 42, "total_cost": 0.012,
                    "served_provider": "fixture-provider"}}]},
        {"actor": "P3", "type": "pull", "beneficiary": "P1", "scores_after": {"P1": 1, "P2": 1, "P3": 0}},
        {"actor": "P1", "type": "message", "to": "P2", "text": "Private deal"},
        {"actor": "P2", "type": "message", "to": "P3", "text": "Private between other agents"},
        {"actor": "P2", "type": "say", "text": "Agent board post"},
        {"actor": "P3", "type": "say_human", "text": "Human board post"},
        {"actor": "P2", "type": "punish", "target": "P1", "scores_after": {"P1": 0, "P2": 0, "P3": 0},
         "score_changes": {"P1": -1, "P2": -1}},
        {"actor": "P1", "type": "intent", "planned": {"action": "pull", "beneficiary": "P2"},
         "reason": "P2 kept the agreement."},
        {"actor": "P3", "type": "message", "to": "P1", "text": "Incoming warning"},
        {"actor": "P2", "type": "pull", "beneficiary": "P3", "forced": True,
         "scripted": True, "control_source": "scripted", "scores_after": {"P1": 0, "P2": 0, "P3": 1}},
        {"actor": "P2", "type": "pull", "beneficiary": "P3", "forced": True,
         "scores_after": {"P1": 0, "P2": 0, "P3": 2}},
        {"actor": "P2", "type": "pull", "beneficiary": "P3", "forced": True,
         "scripted": True, "scores_after": {"P1": 0, "P2": 0, "P3": 3}},
    ]
    if free_reports:
        for i, attached in [
            (0, {"action": "message", "to": "P3", "text": "Private attached report <safe-report>"}),
            (2, {"action": "say_human", "text": "Attached human report"}),
            (3, {"action": "message", "to": "P1", "text": "Attached incoming report"}),
            (8, {"action": "say", "text": "Attached agent report"}),
        ]:
            rows[i]["report"] = {**attached, "free_report": True, "action_cost": 0}
    for i, row in enumerate(rows):
        row.update({"i": i, "round": 1})
    episode = {"id": "fixture", "run": "fixture", "seat_map": {"P1": "alpha", "P2": "beta", "P3": "gamma"},
               "sampling_instruments": {"alpha": {"reasoning_effort": "none", "max_tokens": 256}},
               "events": rows, "final_scores": {"P1": 0, "P2": 0, "P3": 3}, "agent_totals": {
                   "P1": {"actions": 2, "calls": 3, "parse_retries": 0, "reported_cost_usd": 0.012,
                          "free_reports": 2 if free_reports else 0,
                          "unknown_cost_calls": 2, "input_tokens": 42, "unknown_input_calls": 2,
                          "output_tokens": 0, "unknown_output_calls": 3, "points_gained": 1, "points_lost": 1}}}
    (tmp_path / "data/episodes_index.json").write_text(json.dumps([{"id": "fixture", "run": "fixture"}]))
    (tmp_path / "data/episodes/fixture.json").write_text(json.dumps(episode))
    fixtures = json.dumps({"data/episodes_index.json": [{"id": "fixture", "run": "fixture"}],
                           "data/episodes/fixture.json": episode}).replace("<", "\\u003c")
    transport = ("<script>const fixtures = " + fixtures + "; window.fetch = async function(path) {"
                 "if (!(path in fixtures)) throw new Error('Unexpected fixture URL: ' + path);"
                 "return {ok: true, json: async function() {return fixtures[path];}};};</script>")
    page = page.replace("<head>", "<head>" + transport)
    # These checks use rendered content and controls. They do not inspect implementation text.
    checks = r"""
<script>
window.addEventListener('load', async function () {
  const failures = [];
  function check(ok, message) { if (!ok) failures.push(message); }
  for (let attempt = 0; attempt < 100 && !document.querySelector('#feed [data-i]'); attempt++) {
    await new Promise(resolve => setTimeout(resolve, 20));
  }
  const event = i => document.querySelector('#feed [data-i="' + i + '"]');
  const visible = () => Array.from(document.querySelectorAll('#feed [data-i]'))
    .filter(node => !node.classList.contains('hidden')).map(node => +node.dataset.i).join(',');
  const hasReports = !!fixtures['data/episodes/fixture.json'].events[0].report;
  check(document.querySelectorAll('#feed [data-i]').length === 12, 'attached reports added event rows');
  if (hasReports) {
    check(event(0).textContent.includes('Private attached report <safe-report>'), 'attached report missing');
    check(!event(0).querySelector('safe-report'), 'report text interpreted as HTML');
    check(/action cost: 0/.test(event(0).textContent), 'free report action cost missing');
    check(/human board/.test(event(2).querySelector('.ev-report').textContent), 'report audience missing');
  }
  check(!!event(6) && /punish/i.test(event(6).textContent), 'punishment missing');
  check(!!event(6) && event(6).classList.contains('linked'), 'punishment citation missing');
  check(!!event(5) && /human/i.test(event(5).textContent), 'human board missing');
  check(!!event(7) && /P2 kept the agreement/.test(event(7).textContent), 'intent reason missing');
  check(!!event(9) && /scripted.*not chosen/.test(event(9).textContent)
        && /scripted event/.test(event(9).textContent) && !/random/.test(event(9).textContent),
        'scripted control labeled as random');
  for (const i of [10, 11]) {
    check(!!event(i) && /random.*not chosen/.test(event(i).textContent)
          && /random event/.test(event(i).textContent), 'historical random label changed: ' + i);
  }
  for (const i of [0, 1]) {
    const detail = event(i).querySelector('details');
    if (detail) { detail.open = true; detail.dispatchEvent(new Event('toggle')); }
  }
  check(event(0).textContent.includes('Exact saved observation <safe>'), 'exact input missing or unsafe');
  check(!event(0).querySelector('safe'), 'model text interpreted as HTML');
  check(event(0).textContent.includes('Exact saved reply'), 'exact reply missing');
  check(event(0).textContent.includes('fixture-provider'), 'provider missing');
  check(event(0).textContent.includes('reasoning_effort') && event(0).textContent.includes('none'),
        'recorded model settings missing');
  check(/unknown|not recorded/i.test(event(1).textContent), 'missing historical evidence shown as known');
  const chip = document.querySelector('.seat-chip[data-seat="P1"]');
  if (chip) chip.click();
  const mode = document.querySelector('#agent-view-mode');
  check(!!mode, 'agent view control missing');
  if (mode) {
    mode.value = 'own'; mode.dispatchEvent(new Event('change'));
    check(visible() === '0,2,7', 'own actions filter is wrong: ' + visible());
    mode.value = 'received'; mode.dispatchEvent(new Event('change'));
    check(visible() === (hasReports ? '0,1,2,3,4,5,6,8,9,10,11' : '0,1,4,5,6,8,9,10,11'),
          'received/public filter is wrong: ' + visible());
    if (hasReports) {
      check(event(0).querySelector('.ev-report').classList.contains('hidden'), 'private report visible to wrong recipient');
      check(event(2).querySelector('.ev-action').classList.contains('hidden'), 'outgoing parent message leaked into received view');
      check(!event(2).querySelector('.ev-report').classList.contains('hidden'), 'public report hidden with private parent');
    }
    mode.value = 'involved'; mode.dispatchEvent(new Event('change'));
    check(visible() === (hasReports ? '0,1,2,3,6,7,8' : '0,1,2,6,7,8'),
          'involvement filter misses target or report recipient: ' + visible());
  }
  const totals = document.querySelector('#agent-details');
  check(!!totals && /0\.012/.test(totals.textContent) && /2.*unknown/i.test(totals.textContent),
        'agent cost coverage missing');
  if (hasReports) check(/Free reports2/.test(totals.textContent), 'free report total missing');
  const humanToggle = document.querySelector('#filter-human-board');
  if (mode && humanToggle) {
    mode.value = 'received'; mode.dispatchEvent(new Event('change'));
    humanToggle.checked = false; humanToggle.dispatchEvent(new Event('change'));
    check(!visible().split(',').includes('5'), 'human board filter failed');
    check(visible().split(',').includes('4'), 'human filter hid agent board');
    if (hasReports) check(!visible().split(',').includes('2'), 'human report filter failed');
  } else { failures.push('human board filter missing'); }
  document.querySelectorAll('#feed details').forEach(detail => { detail.open = false; });
  if (humanToggle) { humanToggle.checked = true; humanToggle.dispatchEvent(new Event('change')); }
  const result = document.createElement('pre'); result.id = 'test-result';
  result.hidden = true;
  result.textContent = JSON.stringify(failures); document.body.append(result);
});
</script>"""
    (tmp_path / "episodes.html").write_text(page.replace("</body>", checks + "</body>"))

    screenshot = ["--screenshot=" + os.environ["READER_SCREENSHOT"]] if os.environ.get("READER_SCREENSHOT") else []
    result = subprocess.run([
        CHROME, "--headless", "--no-sandbox", "--disable-gpu", "--no-first-run",
        "--disable-background-networking", f"--user-data-dir={tmp_path / 'chrome'}",
        "--allow-file-access-from-files", "--dump-dom", "--virtual-time-budget=5000", "--window-size=1440,1200",
        *screenshot,
        (tmp_path / "episodes.html").as_uri() + "?e=fixture&t=6",
    ], capture_output=True, text=True, timeout=35)
    assert result.returncode == 0, result.stderr[-2000:]
    match = re.search(r'<pre id="test-result"[^>]*>(.*?)</pre>', result.stdout, re.S)
    assert match, result.stdout[-2000:] + result.stderr[-1000:]
    assert json.loads(html.unescape(match[1])) == []
