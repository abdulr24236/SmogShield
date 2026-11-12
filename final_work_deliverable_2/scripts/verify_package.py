#!/usr/bin/env python
"""Check the final Deliverable 2 package against the frozen evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
EXPECTED_HASH = "47D8E0A89FC3F37EA6B74E807DCF566FBDD6D79F734D5A05CFD56124888EB2EB"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main() -> None:
    clf_pred = pd.read_csv(METRICS / "classification_predictions.csv")
    reg_pred = pd.read_csv(METRICS / "regression_predictions.csv")

    assert clf_pred.shape == (473, 12), clf_pred.shape
    assert reg_pred.shape[0] == 473, reg_pred.shape
    assert clf_pred["scorable"].sum() == 466, clf_pred["scorable"].sum()
    assert reg_pred["scorable"].sum() == 466, reg_pred["scorable"].sum()
    assert sha256(METRICS / "classification_predictions.csv") == EXPECTED_HASH

    clf = json.loads((METRICS / "classification_metrics.json").read_text(encoding="utf-8"))
    reg = json.loads((METRICS / "regression_metrics.json").read_text(encoding="utf-8"))
    bias = json.loads((METRICS / "bias_subgroup_metrics.json").read_text(encoding="utf-8"))

    assert abs(clf["selected_model_metrics"]["test"]["f1_macro"] - 0.477983418542242) < 1e-12
    assert abs(clf["baselines"]["persistence_test"]["f1_macro"] - 0.5157054289020233) < 1e-12
    assert abs(reg["hist_gb_test"]["rmse"] - 64.91978398176673) < 1e-9
    assert abs(reg["persistence_test"]["rmse"] - 64.81754811221737) < 1e-9
    assert abs(reg["ridge_test"]["rmse"] - 62.09558304180544) < 1e-9

    assert bias["join_counts"] == {"regression_rows": 466, "classification_rows": 466, "join_rows": 466}
    assert len(bias["results"]) == 47
    assert len({r["dimension"] for r in bias["results"]}) == 7

    charts = list((ROOT / "figures").glob("bias_*.png"))
    assert len(charts) == 14, len(charts)

    required = [
        ROOT / "report_d2.pdf",
        ROOT / "report_d2.html",
        ROOT / "report_d2.tex",
        ROOT / "README.md",
        ROOT / "CHECKLIST.md",
        ROOT / "tables" / "support_by_split.tex",
        ROOT / "tables" / "classification_metrics.tex",
        ROOT / "tables" / "regression_metrics.tex",
        ROOT / "tables" / "bias_test.tex",
        ROOT / "tables" / "git_authors.tex",
        ROOT / "figures" / "architecture.png",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    assert not missing, f"Missing files: {missing}"

    print("Package verification passed.")
    print("Classification rows:", clf_pred.shape, "scorable:", int(clf_pred["scorable"].sum()))
    print("Regression rows:", reg_pred.shape, "scorable:", int(reg_pred["scorable"].sum()))
    print("Test macro F1:", clf["selected_model_metrics"]["test"]["f1_macro"])
    print("Test persistence macro F1:", clf["baselines"]["persistence_test"]["f1_macro"])
    print("Test HGB RMSE:", reg["hist_gb_test"]["rmse"])
    print("Test persistence RMSE:", reg["persistence_test"]["rmse"])
    print("Test Ridge RMSE:", reg["ridge_test"]["rmse"])
    print("Bias rows:", len(bias["results"]), "charts:", len(charts))


if __name__ == "__main__":
    main()
