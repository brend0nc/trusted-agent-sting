"""Judge-vs-gold agreement over a hand-labeled calibration sample, so the numbers are
re-runnable instead of hand-computed.

Usage:  uv run python -m delegated_influence.calibrate_score
"""
import json


def load_records(path):
    """Records from a calibration sample file: either a bare list or {"records": [...]}."""
    doc = json.load(open(path))
    return doc["records"] if isinstance(doc, dict) else doc


def agreement(path, gold, pred):
    """Judge-vs-gold agreement on records where the gold field is present. A list-valued pred
    (e.g. judge_claims_live) is "positive" when non-empty; otherwise pred and gold are compared
    as booleans."""
    records = [r for r in load_records(path) if r.get(gold) is not None and pred in r]
    n = len(records)
    hit = fp = fn = 0
    for r in records:
        g = bool(r[gold])
        p = bool(r[pred])
        if p == g:
            hit += 1
        elif p and not g:
            fp += 1
        else:
            fn += 1
    return {"agreement": hit / n if n else None, "n": n, "fp": fp, "fn": fn}


def binary_metrics(records, gold, pred):
    """Full binary validation summary, including chance-corrected Cohen's kappa."""
    pairs = [(bool(r[gold]), bool(r[pred])) for r in records
             if r.get(gold) is not None and pred in r]
    tp = sum(g and p for g, p in pairs)
    tn = sum(not g and not p for g, p in pairs)
    fp = sum(not g and p for g, p in pairs)
    fn = sum(g and not p for g, p in pairs)
    n = len(pairs)

    def ratio(a, b):
        return a / b if b else None

    accuracy = ratio(tp + tn, n)
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    if n:
        pred_pos, gold_pos = tp + fp, tp + fn
        expected = (pred_pos * gold_pos + (n - pred_pos) * (n - gold_pos)) / (n * n)
        kappa = ratio(accuracy - expected, 1 - expected)
    else:
        kappa = None
    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "specificity": specificity, "f1": f1, "kappa": kappa,
    }


def format_agreement_line(label, result):
    """One CLI summary line; empty calibration slices report n/a instead of crashing."""
    value = result["agreement"]
    agreement_text = f"{value:.2%}" if value is not None else "n/a"
    return (f"{label:<24} agreement {agreement_text}  n={result['n']}  "
            f"fp={result['fp']}  fn={result['fn']}")


if __name__ == "__main__":
    lines = [
        ("lie claims", "data/lie_judge_calibration_sample.json", "gold_has_claims", "judge_claims_live"),
        ("promise (calibration)", "data/judge_calibration_sample.json", "gold_promise", "pred_promise"),
    ]
    for label, path, gold, pred in lines:
        r = agreement(path, gold, pred)
        print(format_agreement_line(label, r))
