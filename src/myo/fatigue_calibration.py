"""Reproduce a literature comparison of the reduced elbow fatigue proxy.

This transfers published parameters, without fitting or claiming validation
against individual subjects. Reference values are calculated from a published
meta-analysis curve, not new measurements or held-out human participants.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .fatigue import (
    ELBOW_FATIGUE_RATE,
    ELBOW_RECOVERY_RATE,
    ELBOW_REST_MULTIPLIER,
    REST_ACTIVATION_THRESHOLD,
    advance_capacity,
    isometric_endurance,
)


SOURCES = {
    "endurance_curve": "https://pmc.ncbi.nlm.nih.gov/articles/PMC2891087/",
    "fatigue_rates": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3397684/",
    "rest_recovery": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6092960/",
    "independent_implementation": "https://myosuite.readthedocs.io/en/latest/fatigue_validation.html",
}


def reference_endurance(load: float) -> float:
    """Elbow power curve, Frey Law & Avin 2010 Table 2; load is a fraction."""
    return 17.98 * load ** -2.21


def calibration_report() -> dict:
    groups = {}
    for name, loads in (
        ("published_optimization_loads", np.arange(0.10, 0.91, 0.10)),
        ("published_test_loads", np.arange(0.15, 0.96, 0.10)),
    ):
        rows = []
        for load in loads:
            reference = reference_endurance(float(load))
            predicted = isometric_endurance(float(load))
            rows.append({
                "load_fraction_mvc": round(float(load), 2),
                "reference_seconds": reference,
                "model_seconds": predicted,
                "relative_error": (predicted - reference) / reference,
            })
        groups[name] = {
            "rows": rows,
            "rmse_seconds": math.sqrt(np.mean([
                (row["model_seconds"] - row["reference_seconds"]) ** 2
                for row in rows
            ])),
            "mean_absolute_relative_error": float(np.mean([
                abs(row["relative_error"]) for row in rows
            ])),
        }
    capacity_samples = []
    for activation in (0.10, 0.15, 0.20, 0.50, 1.0):
        for seconds in (30.0, 180.0, 300.0):
            capacity_samples.append({
                "constant_activation": activation,
                "time_seconds": seconds,
                "capacity": advance_capacity(
                    1.0, activation, seconds, ELBOW_FATIGUE_RATE,
                    ELBOW_RECOVERY_RATE, ELBOW_REST_MULTIPLIER,
                ),
            })
    return {
        "model": "activation_driven_reduced_compartment_proxy",
        "parameters": {
            "fatigue_rate_per_second": ELBOW_FATIGUE_RATE,
            "recovery_rate_per_second": ELBOW_RECOVERY_RATE,
            "rest_recovery_multiplier": ELBOW_REST_MULTIPLIER,
            "rest_activation_threshold": REST_ACTIVATION_THRESHOLD,
        },
        "sources": SOURCES,
        "parameter_selection": "Transferred published elbow parameters; no new fit.",
        "reference": "ET=17.98*load**(-2.21) seconds; load fraction of fresh MVC.",
        "comparison": groups,
        "implementation_crosscheck": {
            "reference": "Official MyoSuite validation, Frey-Law 2012 Elbow row; rounded published outputs.",
            "rows": [
                {
                    "load_fraction_mvc": load,
                    "myosuite_seconds": reference,
                    "reduced_model_seconds": isometric_endurance(load),
                    "relative_difference": isometric_endurance(load) / reference - 1.0,
                }
                for load, reference in ((0.20, 566.0), (0.50, 116.0), (0.80, 27.8))
            ],
            "interpretation": "Agreement with another implementation is not independent human validation.",
        },
        "constant_activation_capacity": capacity_samples,
        "rest_recovery": {
            "fatigue_half_time_seconds": math.log(2) / (
                ELBOW_RECOVERY_RATE * ELBOW_REST_MULTIPLIER
            ),
            "capacity_after_120_seconds_from_half_capacity": advance_capacity(
                0.5, 0.0, 120.0, ELBOW_FATIGUE_RATE,
                ELBOW_RECOVERY_RATE, ELBOW_REST_MULTIPLIER,
            ),
        },
        "limits": [
            "This is not full 3CC-r: its recruitment controller is replaced by the existing activation law.",
            "The test grid was excluded from the published rate fit, not a new independent participant dataset.",
            "Joint-level pooled adult parameters are not individual male or muscle-specific calibration.",
            "The 1% activation rest threshold is an engineering choice; its recovery transition is not experimentally fitted.",
            "No independent recovery-time or dynamic/eccentric-contraction validation has been performed.",
            "Errors are large near maximum load; exactly 100% MVC has zero strict endurance in this reduction.",
            "The model's nonzero sustainable-load asymptote is not evidence of unlimited human endurance.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = calibration_report()
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
