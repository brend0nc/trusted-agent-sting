/* Transcript reader for episodes.html. Vanilla JS, no build step.
 *
 * Data contract (site/data/):
 *   episodes_index.json                  -> picker
 *   episodes/<id>.json                   -> one episode (events, seat_map, scores,
 *                                           promises, cascades)
 *
 * All model/message text goes through textContent — never innerHTML.
 */
(function () {
  "use strict";

  var PALETTE = ["#8b1a1a", "#1a3a6b", "#3a6b1a", "#6b4b1a", "#444", "#7a2a5a", "#2a6b6b"];

  var state = {
    index: [],
    episode: null,
    seats: [],
    letters: {},      // seat id -> display letter ("P1" -> "A"); meeting convention: Player A (model)
    seatColors: {},
    seatFilter: null,
    blocks: [],       // [{el, ev, k}] feed order; k = index into episode.events
    dividers: [],     // [{el, blocks: [...]}]
    scoresAt: [],     // scoresAt[k] = {seat: score} after event k
    zeroScores: {},
    pullEvents: [],   // [{k, actor, beneficiary}]
    promByEvent: {},  // event_i -> [promise, ...]
    blockByI: {},     // event "i" -> block rec (cascade indices refer to "i")
    cascades: [],     // ep.cascades, validated
    arcNodes: [],     // [{ci, el}] visible gutter marks per cascade index
    chainPinned: null,
    chainHover: null,
    stripGeom: null,  // {x0, step, n} of the advantage strip, px
    linkedEl: null,   // block carrying the &t= highlight
    navBlock: null,   // j/k cursor; cleared by organic scroll
    navUntil: 0,
    flashEl: null,
    flashT: 0,
    railScheduled: false,
    layoutScheduled: false,
  };

  function $(id) { return document.getElementById(id); }

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  function mono(text) { return el("span", "mono", text); }

  function svgEl(tag, attrs, text) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var key in attrs) node.setAttribute(key, attrs[key]);
    if (text != null) node.textContent = text;
    return node;
  }

  function setStatus(text) {
    var s = $("reader-status");
    if (s) { s.textContent = text; s.hidden = false; }
  }

  function fmtNum(x) {
    if (typeof x !== "number" || !isFinite(x)) return "—";
    return String(Math.round(x * 100) / 100);
  }

  function letterFor(i) {
    var s = "";
    do { s = String.fromCharCode(65 + (i % 26)) + s; i = Math.floor(i / 26) - 1; } while (i >= 0);
    return s;
  }

  function disp(seat) { return state.letters[seat] || seat || "?"; }

  function seatKeys(map) {
    return Object.keys(map || {}).sort(function (a, b) {
      var na = parseInt(String(a).replace(/\D+/g, ""), 10);
      var nb = parseInt(String(b).replace(/\D+/g, ""), 10);
      if (!isNaN(na) && !isNaN(nb) && na !== nb) return na - nb;
      return a < b ? -1 : a > b ? 1 : 0;
    });
  }

  /* ---- picker ------------------------------------------------------- */

  function renderPicker(index) {
    var select = $("picker-select");
    var tbody = $("picker-table").querySelector("tbody");
    select.textContent = "";
    tbody.textContent = "";

    var placeholder = el("option", null, "— choose an episode —");
    placeholder.value = "";
    select.appendChild(placeholder);

    var groups = {};   // run -> optgroup
    var runOrder = [];
    index.forEach(function (row) {
      if (!row || typeof row.id !== "string") return;
      var run = typeof row.run === "string" ? row.run : "(unknown run)";
      if (!groups[run]) {
        groups[run] = document.createElement("optgroup");
        groups[run].label = run;
        runOrder.push(run);
      }
      var opt = el("option", null,
        row.id + "  ·  " + (row.condition || "?") + "  ·  seed " + (row.seed != null ? row.seed : "?"));
      opt.value = row.id;
      groups[run].appendChild(opt);

      var tr = document.createElement("tr");
      tr.appendChild(el("td", "id", row.id));
      tr.appendChild(el("td", null, run));
      tr.appendChild(el("td", "cond", row.condition || ""));
      tr.appendChild(el("td", null, row.topology || ""));
      tr.appendChild(el("td", "num", row.seed != null ? String(row.seed) : ""));
      tr.appendChild(el("td", null, row.focal_model || "—"));
      tr.appendChild(el("td", "num", row.n_events != null ? String(row.n_events) : ""));
      tr.addEventListener("click", function () { loadEpisode(row.id); });
      tbody.appendChild(tr);
    });
    runOrder.forEach(function (run) { select.appendChild(groups[run]); });

    select.addEventListener("change", function () {
      if (select.value) loadEpisode(select.value);
    });
  }

  /* ---- episode load -------------------------------------------------- */

  function loadEpisode(id, t) {
    setStatus("Loading episode " + id + " …");
    fetch("data/episodes/" + encodeURIComponent(id) + ".json")
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (ep) {
        if (!ep || typeof ep !== "object") throw new Error("bad episode JSON");
        state.episode = ep;
        renderEpisode(ep);
        history.replaceState(null, "", "?e=" + encodeURIComponent(id));
        var ti = t != null ? parseInt(t, 10) : NaN;
        if (!isNaN(ti)) setLinked(ti, true);
        var sel = $("picker-select");
        if (sel) sel.value = id;
        var det = $("picker-details");
        if (det) det.open = false;
        var s = $("reader-status");
        if (s) s.hidden = true;
      })
      .catch(function (err) {
        setStatus("Could not load episode " + id + ": " + err.message);
      });
  }

  function renderEpisode(ep) {
    var seats = seatKeys(ep.seat_map);
    state.seats = seats;
    state.seatColors = {};
    state.letters = {};
    seats.forEach(function (seat, i) {
      state.seatColors[seat] = PALETTE[i % PALETTE.length];
      state.letters[seat] = letterFor(i);
    });
    state.seatFilter = null;
    state.zeroScores = {};
    seats.forEach(function (seat) { state.zeroScores[seat] = 0; });

    state.chainPinned = null;
    state.chainHover = null;
    state.navBlock = null;
    state.linkedEl = null;
    state.cascades = Array.isArray(ep.cascades) ? ep.cascades.filter(function (c) {
      return c && typeof c.ask_i === "number" && typeof c.relay_i === "number" &&
        typeof c.pull_i === "number";
    }) : [];

    // promises: null until the judge has run; then a list keyed by event_i
    state.promByEvent = {};
    if (Array.isArray(ep.promises)) {
      ep.promises.forEach(function (p) {
        if (!p || typeof p.event_i !== "number") return;
        (state.promByEvent[p.event_i] = state.promByEvent[p.event_i] || []).push(p);
      });
    }

    var events = Array.isArray(ep.events) ? ep.events : [];

    // running scores: carry the last known scores across events missing scores_after
    state.scoresAt = [];
    state.pullEvents = [];
    var prev = state.zeroScores;
    events.forEach(function (ev, k) {
      var after = (ev && ev.scores_after && typeof ev.scores_after === "object") ? ev.scores_after : prev;
      state.scoresAt.push(after);
      prev = after;
      if (ev && ev.type === "pull" && ev.actor) {
        state.pullEvents.push({ k: k, actor: ev.actor, beneficiary: ev.beneficiary,
          forced: !!ev.forced });
      }
    });

    $("ep-title").textContent = ep.id || "(episode)";
    $("ep-cond").textContent = condLine(ep);

    renderSeatChips(seats);
    renderSeatsTable(ep, seats);
    renderFeed(ep, events);

    $("episode").hidden = false;
    renderStrip();
    applyFilters();
  }

  function condLine(ep) {
    var parts = [];
    if (ep.condition) parts.push(ep.condition);
    if (ep.seed != null) parts.push("seed " + ep.seed);
    if (ep.rounds != null && ep.actions_per_round != null) {
      parts.push(ep.rounds + " rounds × " + ep.actions_per_round + " actions");
    }
    return parts.join("  ·  ");
  }

  /* ---- filters -------------------------------------------------------- */

  function renderSeatChips(seats) {
    var wrap = $("seat-chips");
    wrap.textContent = "";
    seats.forEach(function (seat) {
      var chip = el("button", "seat-chip", disp(seat));
      chip.type = "button";
      chip.dataset.seat = seat;
      chip.title = seat;
      chip.style.borderLeftColor = state.seatColors[seat];
      chip.setAttribute("aria-pressed", "false");
      chip.addEventListener("click", function () {
        state.seatFilter = state.seatFilter === seat ? null : seat;
        var chips = wrap.querySelectorAll(".seat-chip");
        for (var i = 0; i < chips.length; i++) {
          chips[i].setAttribute("aria-pressed",
            chips[i].dataset.seat === state.seatFilter ? "true" : "false");
        }
        applyFilters();
      });
      wrap.appendChild(chip);
    });
  }

  function involves(ev, seat) {
    return ev.actor === seat || ev.to === seat ||
      ev.beneficiary === seat || ev.credit === seat || ev.target === seat;
  }

  function agentViewIncludes(ev, seat, mode) {
    if (mode === "own") return ev.actor === seat && !ev.forced;
    if (mode === "received") {
      return ev.type === "say" || ev.type === "say_human" || ev.type === "pull" ||
        ev.type === "punish" || (ev.type === "message" && ev.to === seat);
    }
    return involves(ev, seat);
  }

  function applyFilters() {
    var showMsg = $("filter-messages") ? $("filter-messages").checked : true;
    var showPull = $("filter-pulls") ? $("filter-pulls").checked : true;
    var showSay = $("filter-broadcasts") ? $("filter-broadcasts").checked : true;
    var showIntent = $("filter-intents") ? $("filter-intents").checked : true;
    var showHuman = $("filter-human-board") ? $("filter-human-board").checked : true;
    var showPunish = $("filter-punishment") ? $("filter-punishment").checked : true;
    var mode = $("agent-view-mode");
    if (mode) mode.disabled = !state.seatFilter;
    var channels = {message: showMsg, pull: showPull, say: showSay, intent: showIntent,
      say_human: showHuman, punish: showPunish};
    function included(ev) {
      return channels[ev.type] !== false && (!state.seatFilter ||
        agentViewIncludes(ev, state.seatFilter, mode ? mode.value : "involved"));
    }
    state.blocks.forEach(function (b) {
      var showAction = included(b.ev);
      var report = b.ev.report;
      var showReport = !!report && included({actor: b.ev.actor, type: report.action, to: report.to});
      var actionNode = b.el.querySelector('.ev-action');
      var reportNode = b.el.querySelector('.ev-report');
      if (actionNode) actionNode.classList.toggle("hidden", !showAction);
      if (reportNode) reportNode.classList.toggle("hidden", !showReport);
      b.el.classList.toggle("hidden", !showAction && !showReport);
    });
    state.dividers.forEach(function (d) {
      var any = false;
      for (var i = 0; i < d.blocks.length; i++) {
        if (!d.blocks[i].el.classList.contains("hidden")) { any = true; break; }
      }
      d.el.classList.toggle("hidden", !any);
    });
    var note = $("agent-view-note");
    if (note) note.textContent = !state.seatFilter ? "Select an agent. Select it again to show all agents." :
      (mode && mode.value === "received" ?
        "Public events and messages sent to this agent. Open call details to see the exact saved input for a decision." :
        "Agent " + disp(state.seatFilter) + ". Totals cover the whole episode, regardless of filters.");
    renderAgentDetails();
    layoutArcs();
    scheduleRail();
  }

  /* ---- feed ------------------------------------------------------------ */

  function renderFeed(ep, events) {
    var feed = $("feed");
    feed.textContent = "";
    state.blocks = [];
    state.dividers = [];
    state.blockByI = {};

    var lastRound = null;
    var divider = null;
    events.forEach(function (ev, k) {
      if (!ev || typeof ev !== "object") return;
      if (typeof ev.round === "number" && ev.round !== lastRound) {
        lastRound = ev.round;
        var div = el("div", "round-divider", "round " + ev.round);
        var entering = k > 0 ? state.scoresAt[k - 1] : state.zeroScores;
        if (entering) {
          div.appendChild(el("span", "round-scores mono",
            state.seats.map(function (s) {
              return disp(s) + " " + fmtNum(typeof entering[s] === "number" ? entering[s] : 0);
            }).join("  ")));
        }
        feed.appendChild(div);
        divider = { el: div, blocks: [] };
        state.dividers.push(divider);
      }
      var block = buildEvent(ev, k);
      if (!block) return;
      feed.appendChild(block);
      var rec = { el: block, ev: ev, k: k, i: typeof ev.i === "number" ? ev.i : k };
      state.blocks.push(rec);
      state.blockByI[rec.i] = rec;
      if (divider) divider.blocks.push(rec);
    });

    if (!state.blocks.length) {
      feed.appendChild(el("p", "reader-status", "No events in this episode."));
    }
  }

  function pullDelta(k, beneficiary) {
    var after = state.scoresAt[k];
    var before = k > 0 ? state.scoresAt[k - 1] : state.zeroScores;
    if (after && before &&
        typeof after[beneficiary] === "number" && typeof before[beneficiary] === "number") {
      var d = after[beneficiary] - before[beneficiary];
      if (d > 0) return "+" + fmtNum(d);
    }
    return "+1";
  }

  function kindBadge(text) { return el("span", "ev-kind", text); }

  // one-line summary of an intent event's declared plan (or why it is missing)
  function planSummary(ev) {
    var p = ev.planned;
    if (p && typeof p === "object") {
      if (p.action === "pull") {
        return p.beneficiary === ev.actor ? "pull for self" : "pull for " + disp(p.beneficiary);
      }
      if (p.action === "message") return "message " + disp(p.to);
      if (p.action === "say") return "public statement";
      if (p.action === "say_human") return "human-board post";
      if (p.action === "punish") return "punish " + disp(p.target);
      if (p.action) return String(p.action);
    }
    return ev.intent_error ? "no plan (call failed)" : "undecided";
  }

  function buildEvent(ev, k) {
    var block = el("div", "ev");
    block.style.borderLeftColor = state.seatColors[ev.actor] || "#ddd";
    var key = typeof ev.i === "number" ? ev.i : k;
    block.dataset.i = key;

    if (ev.type === "pull" && ev.forced) {
      // Engine controls score like pulls but are not agent choices. Missing source keeps legacy wording.
      block.className = "ev ev-forced";
      block.style.borderLeftColor = "#bbb";
      var fline = el("p", "forced-line");
      var scripted = ev.control_source === "scripted";
      fline.appendChild(kindBadge(scripted ? "scripted · not chosen" : "random · not chosen"));
      fline.appendChild(document.createTextNode(scripted ? "◇ scripted event — no agent chose this: " :
        "◇ random event — no one chose this: "));
      fline.appendChild(mono(disp(ev.actor)));
      fline.appendChild(document.createTextNode("'s lever pulled for "));
      fline.appendChild(mono(disp(ev.beneficiary)));
      fline.appendChild(document.createTextNode(" "));
      fline.appendChild(mono("(" + pullDelta(k, ev.beneficiary) + ")"));
      block.appendChild(fline);
    } else if (ev.type === "pull") {
      block.className = "ev ev-pull";
      var line = el("p", "ev-line");
      line.appendChild(kindBadge("public · pull"));
      line.appendChild(document.createTextNode("■ "));
      line.appendChild(mono(disp(ev.actor)));
      if (ev.beneficiary === ev.actor) {
        line.appendChild(document.createTextNode(" pulls for self "));
      } else {
        line.appendChild(document.createTextNode(" pulls for "));
        line.appendChild(mono(disp(ev.beneficiary)));
        line.appendChild(document.createTextNode(" "));
      }
      line.appendChild(mono("(" + pullDelta(k, ev.beneficiary) + ")"));
      block.appendChild(line);
      if (ev.credit) {
        var credit = el("p", "ev-credit");
        credit.appendChild(document.createTextNode("credit: "));
        credit.appendChild(mono(disp(ev.credit)));
        block.appendChild(credit);
      }
    } else if (ev.type === "intent") {
      // pre-round declared plan (declared_intents control): meta, not a game action — kept subtle
      block.className = "ev ev-intent";
      block.style.borderLeftColor = "#ddd";
      if (ev.intent_error) block.title = String(ev.intent_error);
      var iline = el("p", "intent-line");
      iline.appendChild(kindBadge("private · intent"));
      iline.appendChild(mono(disp(ev.actor)));
      iline.appendChild(document.createTextNode(" plans: " + planSummary(ev)));
      block.appendChild(iline);
      if (ev.reason) block.appendChild(el("p", "intent-reason", "Stated reason: " + ev.reason));
    } else if (ev.type === "punish") {
      block.className = "ev ev-punish";
      var pline = el("p", "ev-line");
      pline.appendChild(kindBadge("public · punishment"));
      pline.appendChild(mono(disp(ev.actor) + " punishes " + disp(ev.target)));
      block.appendChild(pline);
    } else if (ev.type === "message") {
      block.className = "ev ev-msg";
      var mhead = el("p", "msg-head mono", disp(ev.actor) + " → " + disp(ev.to));
      mhead.appendChild(kindBadge("private · message"));
      block.appendChild(mhead);
      block.appendChild(el("p", "msg-text", ev.text != null ? String(ev.text) : ""));
    } else if (ev.type === "say" || ev.type === "say_human") {
      // public broadcast: no recipient field — everyone sees it
      block.className = "ev ev-say";
      var shead = el("p", "say-head mono", disp(ev.actor) + " → all");
      shead.appendChild(kindBadge(ev.type === "say_human" ? "public · human board" : "public · agent board"));
      block.appendChild(shead);
      block.appendChild(el("p", "say-text", ev.text != null ? String(ev.text) : ""));
    } else if (ev.type === "forfeit") {
      block.className = "ev ev-forfeit";
      var line2 = el("p", "forfeit-line");
      line2.appendChild(mono(disp(ev.actor)));
      var reason = ev.forfeit_reason ? " forfeits — " + ev.forfeit_reason : " forfeits";
      line2.appendChild(document.createTextNode(reason));
      block.appendChild(line2);
    } else {
      block.appendChild(el("p", "ev-line", disp(ev.actor) + " · " + (ev.type || "unrecognized event")));
      block.appendChild(el("p", "reader-note", "No display format exists for this event. Its saved fields appear below."));
    }

    var changes = ev.score_changes;
    if (!changes) {
      changes = {};
      var before = k > 0 ? state.scoresAt[k - 1] : state.zeroScores;
      state.seats.forEach(function (seat) {
        var delta = state.scoresAt[k][seat] - before[seat];
        if (delta) changes[seat] = delta;
      });
    }
    if (Object.keys(changes).length) block.appendChild(el("p", "ev-credit", "Score change: " +
      Object.keys(changes).map(function (seat) {
        return disp(seat) + " " + (changes[seat] > 0 ? "+" : "") + fmtNum(changes[seat]);
      }).join(" · ")));
    if (ev.report) {
      var actionBody = el("div", "ev-action");
      while (block.firstChild) actionBody.appendChild(block.firstChild);
      block.appendChild(actionBody);
      var report = ev.report;
      var reportBody = el("div", "ev-report");
      var privateReport = report.action === "message";
      var reportHead = el("p", privateReport ? "msg-head mono" : "say-head mono",
        disp(ev.actor) + " → " + (privateReport ? disp(report.to) : "all"));
      reportHead.appendChild(kindBadge(privateReport ? "private · report" :
        (report.action === "say_human" ? "public · human board" : "public · agent board")));
      reportBody.appendChild(reportHead);
      reportBody.appendChild(el("p", "ev-credit", "Optional report · action cost: 0 · point cost: 0"));
      reportBody.appendChild(el("p", privateReport ? "msg-text" : "say-text",
        report.text != null ? String(report.text) : ""));
      block.appendChild(reportBody);
    }
    block.appendChild(eventDetails(ev, key));

    var proms = state.promByEvent[key];
    if (proms && proms.length) {
      var tags = el("p", "ev-tags");
      proms.forEach(function (p) {
        tags.appendChild(el("span", "tag", "promise"));
        if (p.kept === true) tags.appendChild(el("span", "tag tag-kept", "kept"));
        else if (p.kept === false) tags.appendChild(el("span", "tag tag-broken", "broken"));
      });
      block.appendChild(tags);
    }
    return block;
  }

  function fieldValue(value) {
    if (value == null) return "Unknown: unavailable in this export";
    return typeof value === "string" ? value : JSON.stringify(value, null, 2);
  }

  function detailField(parent, title, value) {
    parent.appendChild(el("p", "detail-label", title));
    parent.appendChild(el("pre", "call-value", fieldValue(value)));
  }

  function firstRecorded(obj, keys) {
    for (var i = 0; i < keys.length; i++) if (obj[keys[i]] != null) return obj[keys[i]];
    return null;
  }

  function eventDetails(ev, key) {
    var details = el("details", "event-details");
    details.appendChild(el("summary", null, "Event " + key + " · saved details"));
    var rendered = false;
    details.addEventListener("toggle", function () {
      scheduleLayout();
      if (!details.open || rendered) return;
      rendered = true;
      var link = el("a", "event-link", "Link to this event");
      link.href = "?e=" + encodeURIComponent(state.episode.id) + "&t=" + key;
      details.appendChild(link);
      details.appendChild(el("p", "reader-note", "Parse retries: " + fieldValue(ev.parse_retries)));
      var savedEvent = {};
      Object.keys(ev).forEach(function (name) { if (name !== "calls") savedEvent[name] = ev[name]; });
      detailField(details, "Saved event fields", savedEvent);
      var model = (state.episode.seat_map || {})[ev.actor];
      detailField(details, "Recorded model settings for this episode", (state.episode.sampling_instruments || {})[model]);
      detailField(details, "Observation schema", state.episode.observation_schema);
      detailField(details, "Recorded runtime settings", state.episode.runtime_instrument);
      var calls = ev.calls;
      if (!Array.isArray(calls) || !calls.length) {
        details.appendChild(el("p", "reader-note", ev.forced ? "Engine event: no model call." :
          "Call input, output, usage, and cost are unknown: not recorded in this reader export."));
      }
      (calls || []).forEach(function (call, index) {
        var part = el("section", "call-detail");
        part.appendChild(el("p", "kicker", "Call " + (index + 1)));
        var usage = call.usage || {};
        var cost = firstRecorded(usage, ["reported_cost_usd", "cost_usd", "total_cost", "openrouter_cost", "reported_total_cost"]);
        var metadata = {};
        Object.keys(call).forEach(function (name) {
          if (["input", "response", "output", "usage"].indexOf(name) < 0) metadata[name] = call[name];
        });
        detailField(part, "Call ID", call.call_id || call.response_id);
        detailField(part, "Reported cost (USD)", cost);
        detailField(part, "Provider", call.provider || usage.served_provider);
        detailField(part, "Served model", call.served_model || usage.served_model);
        detailField(part, "Exact saved input", call.input);
        detailField(part, "Saved reply", call.response == null ? call.output : call.response);
        detailField(part, "Usage (missing fields are unknown)", usage);
        detailField(part, "Call metadata and errors", metadata);
        details.appendChild(part);
      });
    });
    return details;
  }

  function coveredTotal(row, field, unknownField) {
    if (row[field] == null) return "Unknown";
    var unknown = row[unknownField];
    if (unknown == null) return fmtNum(row[field]) + " recorded; coverage unknown";
    return fmtNum(row[field]) + (unknown ? " recorded; " + unknown + " calls unknown" : "");
  }

  function renderAgentDetails() {
    var panel = $("agent-details");
    if (!panel) return;
    panel.textContent = "";
    panel.hidden = !state.seatFilter;
    if (panel.hidden) return;
    var seat = state.seatFilter;
    panel.appendChild(el("p", "kicker", "Agent " + disp(seat) + " · whole episode"));
    var row = (state.episode.agent_totals || {})[seat];
    if (!row) {
      panel.appendChild(el("p", "reader-note", "Agent totals are not recorded in this export. Rebuild the reader data to add them."));
      return;
    }
    var cost = row.reported_cost_usd == null ? "Unknown" : "$" + Number(row.reported_cost_usd).toFixed(6);
    if (row.unknown_cost_calls) cost = (row.unknown_cost_calls === row.calls ? "" : cost + " recorded; ") +
      row.unknown_cost_calls + " calls unknown";
    var values = [
      ["Actions / calls", fmtNum(row.actions) + " / " + (row.unknown_call_count_events ? "at least " : "") + fmtNum(row.calls)],
      ["Parse retries", fmtNum(row.parse_retries) + (row.unknown_parse_retry_events ?
        " recorded; " + row.unknown_parse_retry_events + " events unknown" : "")],
      ["Points gained / lost", fmtNum(row.points_gained) + " / " + fmtNum(row.points_lost)],
      ["Private sent / received", fmtNum(row.private_messages) + " / " + fmtNum(row.received_messages)],
      ["Agent / human posts", fmtNum(row.agent_posts) + " / " + fmtNum(row.human_posts)],
      ["Free reports", fmtNum(row.free_reports == null ? 0 : row.free_reports)],
      ["Input tokens (incl. cache)", coveredTotal(row, "input_tokens", "unknown_input_calls")],
      ["Uncached input", coveredTotal(row, "uncached_input_tokens", "unknown_uncached_input_calls")],
      ["Output tokens", coveredTotal(row, "output_tokens", "unknown_output_calls")],
      ["Reasoning tokens", coveredTotal(row, "reasoning_tokens", "unknown_reasoning_calls")],
      ["Cache read", coveredTotal(row, "cache_read_tokens", "unknown_cache_read_calls")],
      ["Cache write", coveredTotal(row, "cache_write_tokens", "unknown_cache_write_calls")],
      ["Reported USD", cost],
      ["Estimated unreported USD", row.calls_with_cost_estimate ?
        "$" + Number(row.estimated_unreported_cost_usd).toFixed(6) + " for " + row.calls_with_cost_estimate + " calls" : "No saved estimates"]
    ];
    var table = el("table", "rail-table agent-totals");
    values.forEach(function (value) {
      var tr = el("tr");
      tr.appendChild(el("th", null, value[0]));
      tr.appendChild(el("td", "num", value[1]));
      table.appendChild(tr);
    });
    panel.appendChild(table);
  }

  /* ---- right rail ------------------------------------------------------- */

  function seatLabel(seat) {
    var lab = el("span", "seat-label", disp(seat));
    lab.title = seat;
    lab.style.borderLeftColor = state.seatColors[seat] || "#ddd";
    return lab;
  }

  function renderSeatsTable(ep, seats) {
    var table = $("rail-seats");
    table.textContent = "";
    var thead = document.createElement("thead");
    var hr = document.createElement("tr");
    hr.appendChild(el("th", null, "player"));
    hr.appendChild(el("th", null, "model"));
    hr.appendChild(el("th", "num", "final"));
    thead.appendChild(hr);
    table.appendChild(thead);

    var tbody = document.createElement("tbody");
    seats.forEach(function (seat) {
      var tr = document.createElement("tr");
      var td = document.createElement("td");
      td.appendChild(seatLabel(seat));
      if (ep.focal === seat) td.appendChild(el("span", "tag", "focal"));
      tr.appendChild(td);
      tr.appendChild(el("td", "mono", (ep.seat_map && ep.seat_map[seat]) || "?"));
      var score = ep.final_scores ? ep.final_scores[seat] : null;
      tr.appendChild(el("td", "num", score != null ? fmtNum(score) : "—"));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
  }

  function renderScores(k, round) {
    var cap = $("rail-scores-caption");
    if (cap) cap.textContent = round != null ? "running scores — round " + round : "running scores";
    var scores = k >= 0 && state.scoresAt[k] ? state.scoresAt[k] : state.zeroScores;
    var table = $("rail-scores");
    table.textContent = "";
    var tbody = document.createElement("tbody");
    state.seats.forEach(function (seat) {
      var tr = document.createElement("tr");
      tr.appendChild(el("td", "mono", disp(seat)));
      tr.appendChild(el("td", "num", typeof scores[seat] === "number" ? fmtNum(scores[seat]) : "0"));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
  }

  function renderLedger(k) {
    var counts = {};  // "actor|beneficiary" -> n
    state.pullEvents.forEach(function (p) {
      if (p.k > k) return;
      if (p.forced) return;  // a random favour is not a choice — this ledger is who CHOSE to pull
      var key = p.actor + "|" + p.beneficiary;
      counts[key] = (counts[key] || 0) + 1;
    });
    var table = $("rail-ledger");
    table.textContent = "";
    var thead = document.createElement("thead");
    var hr = document.createElement("tr");
    hr.appendChild(el("th", null, ""));
    state.seats.forEach(function (seat) { hr.appendChild(el("th", "num", disp(seat))); });
    thead.appendChild(hr);
    table.appendChild(thead);

    var tbody = document.createElement("tbody");
    state.seats.forEach(function (puller) {
      var tr = document.createElement("tr");
      tr.appendChild(el("td", "mono", disp(puller)));
      state.seats.forEach(function (ben) {
        var n = counts[puller + "|" + ben];
        tr.appendChild(el("td", "num", n ? String(n) : ""));
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
  }

  /* Topmost visible event: first non-hidden block whose bottom edge is below a
     small reading offset. If everything has scrolled past, use the last one. */
  function topVisibleBlock() {
    var top = null, lastAbove = null;
    for (var j = 0; j < state.blocks.length; j++) {
      var b = state.blocks[j];
      if (b.el.classList.contains("hidden")) continue;
      var r = b.el.getBoundingClientRect();
      if (r.bottom > 80) { top = b; break; }
      lastAbove = b;
    }
    return top || lastAbove;
  }

  function updateRail() {
    state.railScheduled = false;
    if (!state.episode) return;
    var cur = topVisibleBlock();
    renderScores(cur ? cur.k : -1, cur && typeof cur.ev.round === "number" ? cur.ev.round : null);
    renderLedger(cur ? cur.k : -1);
    stripCursor(cur ? cur.k : -1);
  }

  function scheduleRail() {
    if (state.railScheduled) return;
    state.railScheduled = true;
    // rAF throttles scroll updates, but is suppressed in hidden tabs — fall
    // back to a timeout so the rail still renders there.
    if (document.visibilityState === "visible" && window.requestAnimationFrame) {
      window.requestAnimationFrame(updateRail);
    } else {
      window.setTimeout(updateRail, 0);
    }
  }

  /* ---- advantage strip ----------------------------------------------------
     Stepped cumulative-score line per seat over event index, pull ticks on the
     baseline, letter end labels. Static-complete: always drawn from full data,
     never filtered. Redrawn on resize and print. */

  function renderStrip() {
    var wrap = $("adv-strip"), svg = $("strip-svg");
    if (!wrap || !svg) return;
    var n = state.scoresAt.length;
    if (!n) { wrap.hidden = true; state.stripGeom = null; return; }
    wrap.hidden = false;

    var w = wrap.clientWidth || 600, H = 48, padL = 4, padR = 30, padT = 4, baseY = 38;
    var innerW = Math.max(1, w - padL - padR);
    var step = n > 1 ? innerW / (n - 1) : 0;
    state.stripGeom = { x0: padL, step: step, n: n };

    var lo = 0, hi = 1;
    state.scoresAt.forEach(function (s) {
      state.seats.forEach(function (seat) {
        var v = s[seat];
        if (typeof v === "number") { if (v < lo) lo = v; if (v > hi) hi = v; }
      });
    });
    function r(v) { return Math.round(v * 10) / 10; }
    function x(k) { return r(padL + step * k); }
    function y(v) { return r(baseY - (v - lo) / (hi - lo) * (baseY - padT)); }

    svg.setAttribute("width", w);
    svg.setAttribute("height", H);
    svg.setAttribute("viewBox", "0 0 " + w + " " + H);
    svg.textContent = "";

    svg.appendChild(svgEl("line",
      { x1: padL, y1: baseY, x2: padL + innerW, y2: baseY, stroke: "#ddd" }));

    state.seats.forEach(function (seat) {
      var d = "M" + x(0) + " " + y(0), prevV = 0;
      for (var k = 0; k < n; k++) {
        var v = typeof state.scoresAt[k][seat] === "number" ? state.scoresAt[k][seat] : prevV;
        d += "H" + x(k);
        if (v !== prevV) d += "V" + y(v);
        prevV = v;
      }
      svg.appendChild(svgEl("path",
        { d: d, fill: "none", stroke: state.seatColors[seat], "stroke-width": 1.5 }));
    });

    state.pullEvents.forEach(function (p) {
      svg.appendChild(svgEl("line", {
        x1: x(p.k), y1: baseY, x2: x(p.k), y2: baseY + 5,
        // forced favours are gray: the score moved, but the actor did not choose it
        stroke: p.forced ? "#bbb" : (state.seatColors[p.actor] || "#555"),
      }));
    });

    // right-edge letter labels; nudge apart when finals tie
    var finals = state.scoresAt[n - 1] || state.zeroScores;
    var labs = state.seats.map(function (seat) {
      return { seat: seat, y: y(typeof finals[seat] === "number" ? finals[seat] : 0) };
    }).sort(function (a, b) { return a.y - b.y; });
    var i;
    for (i = 1; i < labs.length; i++) {
      if (labs[i].y - labs[i - 1].y < 9) labs[i].y = labs[i - 1].y + 9;
    }
    for (i = labs.length - 1; i >= 0; i--) {
      var cap = i === labs.length - 1 ? H - 3 : labs[i + 1].y - 9;
      if (labs[i].y > cap) labs[i].y = cap;
    }
    labs.forEach(function (L) {
      svg.appendChild(svgEl("text",
        { x: padL + innerW + 5, y: r(L.y + 3), fill: state.seatColors[L.seat] }, disp(L.seat)));
    });

    svg.appendChild(svgEl("line", {
      id: "strip-cursor", x1: -9, y1: 2, x2: -9, y2: H - 2,
      stroke: "#8b1a1a", visibility: "hidden",
    }));
  }

  function stripCursor(k) {
    var cur = document.getElementById("strip-cursor");
    if (!cur || !state.stripGeom) return;
    if (k < 0) { cur.setAttribute("visibility", "hidden"); return; }
    var g = state.stripGeom;
    var x = g.x0 + g.step * Math.min(k, g.n - 1);
    cur.setAttribute("x1", x);
    cur.setAttribute("x2", x);
    cur.removeAttribute("visibility");
  }

  function onStripClick(e) {
    var g = state.stripGeom;
    if (!g) return;
    var x = e.clientX - e.currentTarget.getBoundingClientRect().left;
    var k = g.step > 0 ? Math.round((x - g.x0) / g.step) : 0;
    seekToEvent(Math.max(0, Math.min(g.n - 1, k)));
  }

  /* ---- cascade arc gutter --------------------------------------------------
     Two chained arcs per cascade (ask→relay, relay→pull) in the eliciting
     seat's color, plus a small square at b's first pull for a. Endpoints are
     measured from live block geometry, so this reruns on filter change,
     resize, and print. */

  function arcPath(x0, y1, y2) {
    if (y2 < y1) { var t = y1; y1 = y2; y2 = t; }
    var ry = Math.max(4, (y2 - y1) / 2), rx = Math.min(24, ry);
    return "M" + x0 + " " + y1 + " A" + rx + " " + ry + " 0 0 0 " + x0 + " " + y2;
  }

  function chainHandlers(node, ci) {
    node.addEventListener("mouseenter", function () { state.chainHover = ci; applyChain(); });
    node.addEventListener("mouseleave", function () {
      if (state.chainHover === ci) { state.chainHover = null; applyChain(); }
    });
    node.addEventListener("click", function () {
      state.chainPinned = state.chainPinned === ci ? null : ci;
      applyChain();
    });
  }

  function layoutArcs() {
    var gut = $("gutter"), svg = $("cascade-svg");
    if (!gut || !svg) return;
    if (!state.cascades.length) {
      gut.hidden = true;
      svg.textContent = "";
      state.arcNodes = [];
      return;
    }
    gut.hidden = false;
    var count = $("cascade-count");
    if (count) {
      count.textContent = state.cascades.length +
        (state.cascades.length === 1 ? " cascade" : " cascades");
    }

    var feedR = $("feed").getBoundingClientRect();
    var svgR = svg.getBoundingClientRect();
    var h = Math.max(24, Math.ceil(feedR.bottom - svgR.top));
    svg.setAttribute("width", 56);
    svg.setAttribute("height", h);
    svg.setAttribute("viewBox", "0 0 56 " + h);
    svg.textContent = "";
    state.arcNodes = [];

    var x0 = 55;
    function visible(b) { return b && !b.el.classList.contains("hidden"); }
    function yOf(b) {
      var r = b.el.getBoundingClientRect();
      return Math.round(r.top - svgR.top + Math.min(r.height / 2, 14));
    }
    function addArc(y1, y2, col, ci) {
      var d = arcPath(x0, y1, y2);
      var p = svgEl("path", { d: d, "class": "arc", stroke: col });
      var hit = svgEl("path", { d: d, "class": "arc-hit" });   // wide invisible hover/click target
      chainHandlers(hit, ci);
      svg.appendChild(p);
      svg.appendChild(hit);
      state.arcNodes.push({ ci: ci, el: p });
    }

    state.cascades.forEach(function (c, ci) {
      var ask = state.blockByI[c.ask_i];
      var relay = state.blockByI[c.relay_i];
      var pull = state.blockByI[c.pull_i];
      var col = state.seatColors[c.a] || "#555";
      if (visible(ask) && visible(relay)) addArc(yOf(ask), yOf(relay), col, ci);
      if (visible(relay) && visible(pull)) addArc(yOf(relay), yOf(pull), col, ci);
      if (typeof c.b_pull_i === "number") {
        var bp = state.blockByI[c.b_pull_i];
        if (visible(bp)) {
          var m = svgEl("rect",
            { x: x0 - 5, y: yOf(bp) - 2, width: 5, height: 5, "class": "mark", fill: col });
          chainHandlers(m, ci);
          svg.appendChild(m);
          state.arcNodes.push({ ci: ci, el: m });
        }
      }
    });
    applyChain();
  }

  function cascadeEndingAt(i) {
    for (var ci = 0; ci < state.cascades.length; ci++) {
      if (state.cascades[ci].pull_i === i) return ci;
    }
    return null;
  }

  function applyChain() {
    var svg = $("cascade-svg");
    if (!svg) return;
    var sel = state.chainPinned != null ? state.chainPinned : state.chainHover;
    svg.classList.toggle("dimmed", sel != null);
    state.arcNodes.forEach(function (a) { a.el.classList.toggle("on", a.ci === sel); });
    var member = {};
    if (sel != null && state.cascades[sel]) {
      var c = state.cascades[sel];
      [c.ask_i, c.relay_i, c.pull_i, c.b_pull_i].forEach(function (i) {
        if (typeof i === "number") member[i] = true;
      });
    }
    state.blocks.forEach(function (b) { b.el.classList.toggle("chain", !!member[b.i]); });
  }

  /* ---- seek, keys, deep links ---------------------------------------------- */

  function visibleBlocks() {
    return state.blocks.filter(function (b) { return !b.el.classList.contains("hidden"); });
  }

  function flash(node) {
    if (state.flashEl) { state.flashEl.classList.remove("flash"); clearTimeout(state.flashT); }
    state.flashEl = node;
    node.classList.add("flash");
    state.flashT = window.setTimeout(function () {
      node.classList.remove("flash");
      state.flashEl = null;
    }, 1200);
  }

  function goTo(rec) {
    state.navBlock = rec;
    state.navUntil = Date.now() + 600;
    window.scrollBy(0, rec.el.getBoundingClientRect().top - 80);
    flash(rec.el);
  }

  function seekToEvent(k) {
    var vis = visibleBlocks();
    if (!vis.length) return;
    var target = null;
    for (var i = 0; i < vis.length; i++) {
      if (vis[i].k >= k) { target = vis[i]; break; }
    }
    goTo(target || vis[vis.length - 1]);
  }

  function stepEvent(dir) {
    var vis = visibleBlocks();
    if (!vis.length) return;
    var anchor = state.navBlock && vis.indexOf(state.navBlock) !== -1
      ? state.navBlock : topVisibleBlock();
    var at = anchor ? vis.indexOf(anchor) : (dir > 0 ? -1 : vis.length);
    goTo(vis[Math.max(0, Math.min(vis.length - 1, at + dir))]);
  }

  function stepRound(dir) {
    var target = null;
    for (var i = 0; i < state.dividers.length; i++) {
      var d = state.dividers[i];
      if (d.el.classList.contains("hidden")) continue;
      var top = d.el.getBoundingClientRect().top;
      if (dir > 0) {
        if (top > 82) { target = d.el; break; }
      } else if (top < 78) {
        target = d.el;
      } else {
        break;
      }
    }
    if (!target) return;
    state.navBlock = null;
    window.scrollBy(0, target.getBoundingClientRect().top - 80);
    flash(target);
  }

  function onKey(e) {
    if (!state.episode || e.metaKey || e.ctrlKey || e.altKey) return;
    var t = e.target;
    if (t && (/^(input|select|textarea)$/i.test(t.tagName) || t.isContentEditable)) return;
    if (e.key === "Escape") {
      if (state.chainPinned != null || state.chainHover != null) {
        state.chainPinned = null;
        state.chainHover = null;
        applyChain();
      }
      return;
    }
    if (e.key === "j") stepEvent(1);
    else if (e.key === "k") stepEvent(-1);
    else if (e.key === "J") stepRound(1);
    else if (e.key === "K") stepRound(-1);
    else return;
    e.preventDefault();
  }

  function setLinked(i, scroll) {
    var rec = state.blockByI[i];
    if (!rec || !state.episode) return;
    if (state.linkedEl) state.linkedEl.classList.remove("linked");
    state.linkedEl = rec.el;
    rec.el.classList.add("linked");
    history.replaceState(null, "",
      "?e=" + encodeURIComponent(state.episode.id) + "&t=" + encodeURIComponent(i));
    if (scroll) window.scrollBy(0, rec.el.getBoundingClientRect().top - 80);
  }

  function onFeedClick(e) {
    if (!state.episode || !e.target || !e.target.closest) return;
    if (e.target.closest(".event-details")) return;
    var node = e.target.closest(".ev");
    if (!node) return;
    var i = parseInt(node.dataset.i, 10);
    if (isNaN(i)) return;
    setLinked(i, false);
    var ci = cascadeEndingAt(i);   // a cascade-terminating pull also toggles its chain
    if (ci != null) {
      state.chainPinned = state.chainPinned === ci ? null : ci;
      applyChain();
    }
  }

  function relayout() {
    if (!state.episode) return;
    renderStrip();
    layoutArcs();
    scheduleRail();
  }

  function scheduleLayout() {
    if (state.layoutScheduled) return;
    state.layoutScheduled = true;
    var run = function () { state.layoutScheduled = false; relayout(); };
    if (document.visibilityState === "visible" && window.requestAnimationFrame) {
      window.requestAnimationFrame(run);
    } else {
      window.setTimeout(run, 0);
    }
  }

  /* ---- init --------------------------------------------------------------- */

  function init() {
    if (!$("reader-status")) return;  // not the episodes page

    ["filter-messages", "filter-pulls", "filter-broadcasts", "filter-intents", "filter-human-board",
      "filter-punishment", "agent-view-mode"].forEach(function (id) {
      var box = $(id);
      if (box) box.addEventListener("change", applyFilters);
    });
    window.addEventListener("scroll", function () {
      if (Date.now() > state.navUntil) state.navBlock = null;  // organic scroll resets j/k cursor
      scheduleRail();
    }, { passive: true });
    window.addEventListener("resize", scheduleRail);
    window.addEventListener("resize", scheduleLayout);
    window.addEventListener("beforeprint", relayout);
    window.addEventListener("afterprint", relayout);
    document.addEventListener("keydown", onKey);
    var strip = $("strip-svg");
    if (strip) strip.addEventListener("click", onStripClick);
    var feed = $("feed");
    if (feed) feed.addEventListener("click", onFeedClick);

    fetch("data/episodes_index.json")
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (index) {
        if (!Array.isArray(index)) throw new Error("index is not a list");
        state.index = index;
        renderPicker(index);
        $("picker").hidden = false;
        var s = $("reader-status");
        if (s) s.hidden = true;
        var params = new URLSearchParams(window.location.search);
        var want = params.get("e");
        if (want) loadEpisode(want, params.get("t"));
      })
      .catch(function (err) {
        setStatus("Could not load episode index: " + err.message);
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
