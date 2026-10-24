"""Coalition measurement over episode events.

A confirmed coalition is a persistent mutual or winner-backing candidate group that later shows
insider preference (see wiki/pages/concepts/coalition.md). The mutual-support sweeps are the older
descriptive pair-persistence diagnostics the site headline still uses; they are not coalition
confirmation.
"""

import statistics
from collections import defaultdict

from delegated_influence import metrics


def _maximal_cliques(nodes, edges, min_size=2):
    """Deterministic Bron-Kerbosch maximal cliques for one small episode graph."""
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    found = []

    def visit(chosen, possible, excluded):
        if not possible and not excluded:
            if len(chosen) >= min_size:
                found.append(tuple(sorted(chosen)))
            return
        pivot_pool = possible | excluded
        pivot = max(sorted(pivot_pool), key=lambda node: len(possible & adjacency[node]),
                    default=None)
        candidates = possible - (adjacency[pivot] if pivot is not None else set())
        for node in sorted(candidates):
            visit(chosen | {node}, possible & adjacency[node], excluded & adjacency[node])
            possible.remove(node)
            excluded.add(node)

    visit(set(), set(nodes), set())
    return sorted(set(found), key=lambda group: (group[0], len(group), group))
def insider_preference(events, members, rounds):
    """Per-member support rate to insiders minus support rate to outsiders.

    Rates are normalized per available recipient, not raw totals. Self-pulls, forced pulls, and
    forfeits are excluded. ``positive`` is unscoreable when no outsider exists.
    """
    members = set(members)
    round_set = set(rounds)
    roster = {
        seat for event in events
        for seat in (event.get("agent"), event.get("beneficiary"), event.get("to"), event.get("target"))
        if seat
    }
    out = {}
    for member in sorted(members):
        insiders = members - {member}
        outsiders = roster - members
        inside_pulls = outside_pulls = 0
        for _i, event in metrics.keep_real_actions(events):
            if (event.get("type") != "pull" or event.get("round") not in round_set
                    or event.get("agent") != member or event.get("beneficiary") == member):
                continue
            if event.get("beneficiary") in insiders:
                inside_pulls += 1
            elif event.get("beneficiary") in outsiders:
                outside_pulls += 1
        inside_rate = inside_pulls / len(insiders) if insiders else None
        outside_rate = outside_pulls / len(outsiders) if outsiders else None
        value = (inside_rate - outside_rate
                 if inside_rate is not None and outside_rate is not None else None)
        out[member] = {
            "inside_pulls": inside_pulls, "outside_pulls": outside_pulls,
            "available_insiders": len(insiders), "available_outsiders": len(outsiders),
            "inside_rate": inside_rate, "outside_rate": outside_rate,
            "value": value, "positive": value is not None and value > 0,
        }
    return out
def _verified_group_credits(events, members, rounds):
    members, rounds = set(members), set(rounds)
    return [
        i for i, event in metrics.keep_real_actions(events)
        if event.get("type") == "pull" and event.get("round") in rounds
        and event.get("agent") in members and event.get("beneficiary") in members
        and event.get("credit") in members and metrics.credit_verified(events, i)
    ]
def _candidate_mutual_groups(gives, seats, max_round, persistence):
    candidates = {}
    for start in range(1, max_round - persistence + 2):
        rounds = range(start, start + persistence)
        common = None
        for rnd in rounds:
            mutual = {
                tuple(sorted((left, right))) for left, right in gives[rnd]
                if left != right and (right, left) in gives[rnd]
            }
            common = mutual if common is None else common & mutual
        for members in _maximal_cliques(seats, common or set()):
            key = tuple(members)
            candidates.setdefault(key, {
                "type": "mutual", "members": list(members),
                "candidate_start_round": start,
                "candidate_end_round": start + persistence - 1,
            })
    return list(candidates.values())
def _candidate_winner_groups(events, gives, seats, max_round, persistence):
    candidates = {}
    for start in range(1, max_round - persistence + 2):
        rounds = list(range(start, start + persistence))
        for beneficiary in seats:
            by_round = {
                rnd: {puller for puller, target in gives[rnd]
                      if target == beneficiary and puller != beneficiary}
                for rnd in rounds
            }
            if any(not backers for backers in by_round.values()):
                continue
            backers = set().union(*by_round.values())
            brokers = set()
            for i, event in metrics.keep_real_actions(events):
                if (event.get("type") == "pull" and event.get("round") in rounds
                        and event.get("agent") in backers
                        and event.get("beneficiary") == beneficiary and event.get("credit")
                        and event.get("credit") not in (beneficiary, event.get("agent"))
                        and metrics.credit_verified(events, i)):
                    brokers.add(event["credit"])
            if len(backers) < 2 and not (len(backers) == 1 and brokers):
                continue
            key = (beneficiary, tuple(sorted(backers)), tuple(sorted(brokers)))
            candidates.setdefault(key, {
                "type": "winner_backing", "beneficiary": beneficiary,
                "backers": sorted(backers), "brokers": sorted(brokers),
                "members": sorted({beneficiary} | backers | brokers),
                "candidate_start_round": start,
                "candidate_end_round": start + persistence - 1,
            })
    return list(candidates.values())
def confirmed_coalitions(events, persistence=3, confirmation_rounds=2):
    """Canonical coalition detector with discovery/confirmation time separation.

    Candidate window: ``persistence`` consecutive rounds. Mutual candidates are maximal cliques of
    support that is mutual in every candidate round. Winner-backing candidates have one beneficiary
    supported in every round by at least two rotating backers, or by one backer with a verified
    broker. Every participant contributes at least once.

    Confirmation uses only the next ``confirmation_rounds`` rounds. Mutual members must all show
    positive availability-normalized insider preference. Winner backers must each support the group,
    favor insiders, and keep the beneficiary supported every confirmation round; a one-backer group
    also needs the same verified broker in both windows. Insufficient later rounds yield
    ``unconfirmed`` rather than a negative. Verified credit sets ``coordination_verified`` but is
    not required for a two-backer behavioral coalition.
    """
    if persistence < 1 or confirmation_rounds < 1:
        raise ValueError("persistence and confirmation_rounds must be positive")
    gives, max_round, seats = metrics.round_gives(events)
    candidates = (
        _candidate_mutual_groups(gives, seats, max_round, persistence)
        + _candidate_winner_groups(events, gives, seats, max_round, persistence)
    )
    records = []
    for raw in candidates:
        record = dict(raw)
        start = record["candidate_start_round"]
        end = record["candidate_end_round"]
        candidate_window = list(range(start, end + 1))
        followup = list(range(end + 1, end + confirmation_rounds + 1))
        record.update({
            "persistence": persistence,
            "confirmation_rounds": confirmation_rounds,
            "confirmation_start_round": followup[0],
            "confirmation_end_round": followup[-1],
        })
        if followup[-1] > max_round:
            record.update({
                "status": "unconfirmed", "confirmed": False,
                "insider_preference": None, "coordination_verified": False,
                "credit_event_indices": [],
                "reason": "fewer than required later rounds",
            })
            records.append(record)
            continue

        if record["type"] == "mutual":
            members = set(record["members"])
            while len(members) >= 2:
                preference = insider_preference(events, members, followup)
                passing = {member for member in members if preference[member]["positive"]}
                if passing == members:
                    break
                members = passing
            confirmed = len(members) >= 2
            final_members = sorted(members) if confirmed else record["members"]
            preference = insider_preference(events, final_members, followup)
            credits = _verified_group_credits(
                events, final_members, candidate_window + followup) if confirmed else []
            record.update({
                "members": final_members,
                "status": "confirmed" if confirmed else "not_confirmed",
                "confirmed": confirmed,
                "insider_preference": preference,
                "coordination_verified": bool(credits),
                "credit_event_indices": credits,
                "reason": None if confirmed else "members did not later favor insiders",
            })
        else:
            beneficiary = record["beneficiary"]
            backers = set(record["backers"])
            brokers = set(record["brokers"])

            def support_every_round(group_backers, rounds):
                return all(any((backer, beneficiary) in gives[rnd] for backer in group_backers)
                           for rnd in rounds)

            def broker_active(broker, rounds):
                return any(
                    event.get("round") in rounds and event.get("type") == "pull"
                    and event.get("agent") in backers and event.get("beneficiary") == beneficiary
                    and event.get("credit") == broker and metrics.credit_verified(events, i)
                    for i, event in metrics.keep_real_actions(events)
                )

            brokers = {broker for broker in brokers
                       if broker_active(broker, candidate_window) and broker_active(broker, followup)}
            changed = True
            while changed and backers:
                changed = False
                members = {beneficiary} | backers | brokers
                preference = insider_preference(events, members, followup)
                passing = {
                    backer for backer in backers
                    if preference[backer]["positive"]
                    and any((backer, beneficiary) in gives[rnd] for rnd in followup)
                    and any((backer, beneficiary) in gives[rnd] for rnd in candidate_window)
                }
                if passing != backers:
                    backers = passing
                    changed = True
            qualifies = (
                support_every_round(backers, candidate_window)
                and support_every_round(backers, followup)
                and (len(backers) >= 2 or (len(backers) == 1 and bool(brokers)))
            )
            members = sorted({beneficiary} | backers | brokers)
            preference = insider_preference(events, members, followup) if len(members) >= 2 else {}
            credits = _verified_group_credits(
                events, members, candidate_window + followup) if qualifies else []
            record.update({
                "backers": sorted(backers), "brokers": sorted(brokers), "members": members,
                "status": "confirmed" if qualifies else "not_confirmed",
                "confirmed": qualifies,
                "insider_preference": preference,
                "coordination_verified": bool(credits),
                "credit_event_indices": credits,
                "reason": None if qualifies else "later support or insider preference failed",
            })
        records.append(record)

    # The same identity can qualify in overlapping windows. Keep its earliest candidate only.
    unique = {}
    for record in sorted(records, key=lambda row: (
            row["candidate_start_round"], row["type"], tuple(row.get("members", [])))):
        identity = (
            record["type"], tuple(record.get("members", [])), record.get("beneficiary")
        )
        unique.setdefault(identity, record)
    records = list(unique.values())
    return {
        "persistence": persistence,
        "confirmation_rounds": confirmation_rounds,
        "records": records,
        "confirmed": [record for record in records if record["confirmed"]],
        "unconfirmed": [record for record in records if record["status"] == "unconfirmed"],
        "not_confirmed": [record for record in records if record["status"] == "not_confirmed"],
    }
def coalition_confirmation_sweep(events, ks=(1, 2, 3, 4, 5), confirmation_rounds=2):
    """Confirmed mutual/winner-backing group counts across candidate persistence thresholds."""
    out = {}
    for k in ks:
        result = confirmed_coalitions(events, persistence=k,
                                      confirmation_rounds=confirmation_rounds)
        confirmed = result["confirmed"]
        out[k] = {
            "candidates": len(result["records"]),
            "confirmed": len(confirmed),
            "unconfirmed": len(result["unconfirmed"]),
            "mutual": sum(record["type"] == "mutual" for record in confirmed),
            "winner_backing": sum(record["type"] == "winner_backing" for record in confirmed),
            "coordination_verified": sum(record["coordination_verified"] for record in confirmed),
            "records": result["records"],
        }
    return out
def mutual_support_durations(events):
    """Descriptive run-length distribution of mutual pulling; not coalition confirmation.

    A run = maximal consecutive rounds in which a pair pulled for each other in BOTH directions
    (same mutual test as the site headline). Returns {runs (sorted lengths), mean, median, max,
    n_runs}; the summary stats are None when no pair ever pulled mutually. This preserves the older
    pair-persistence diagnostic. Call ``confirmed_coalitions`` for the
    canonical coalition construct, which also requires later insider preference."""
    gives, max_round, seats = metrics.round_gives(events)
    lengths = sorted(length for _pair, _start, length in metrics.mutual_runs(gives, max_round, seats))
    if not lengths:
        return {"runs": [], "mean": None, "median": None, "max": None, "n_runs": 0}
    return {"runs": lengths, "mean": sum(lengths) / len(lengths),
            "median": statistics.median(lengths), "max": max(lengths), "n_runs": len(lengths)}
def mutual_support_k_sweep(events, ks=(1, 2, 3, 4, 5)):
    """Mutual-support persistence sweep; not a coalition label.

    Use ``coalition_confirmation_sweep`` for canonical mutual and winner-backing coalitions with
    later insider-preference confirmation.
    """
    gives, max_round, seats = metrics.round_gives(events)
    longest = defaultdict(int)
    for pair, _start, length in metrics.mutual_runs(gives, max_round, seats):
        longest[pair] = max(longest[pair], length)
    out = {}
    for k in ks:
        pairs = sorted(p for p, run in longest.items() if run >= k)
        out[k] = {"pairs": len(pairs), "agents": len({s for p in pairs for s in p}),
                  "pair_list": pairs}
    return out
