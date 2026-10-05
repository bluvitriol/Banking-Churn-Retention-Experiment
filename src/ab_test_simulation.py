"""
Design and simulate an A/B test of a retention intervention targeted at
the high-churn-risk segment identified by the propensity model.

note: The underlying dataset has no real experiment recorded. The
"retained" outcome below is SIMULATED to demonstrate experiment design and
analysis skills, not a real observed business result.
"""

import pandas as pd
import numpy as np
import os
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

SCORES_PATH = "outputs/churn_scores.csv"
AB_RESULTS_PATH = "outputs/ab_test_results.csv"

RANDOM_SEED = 42
HIGH_RISK_PERCENTILE = 0.80   # top 20% churn risk = target segment
BASELINE_RETENTION_RATE = 0.55   # assumed retention rate with no intervention
ASSUMED_UPLIFT = 0.07             # assumed +7pp retention from the offer (for simulation)
ALPHA = 0.05
POWER = 0.80


def calculate_required_sample_size(baseline_rate, mde, alpha=ALPHA, power=POWER):
    """Compute the per-group sample size needed to detect a given minimum
    detectable effect (MDE) at the specified alpha/power, BEFORE running
    the test — this is the 'sample sizing' step the JD explicitly names."""
    effect_size = proportion_effectsize(baseline_rate, baseline_rate + mde)
    analysis = NormalIndPower()
    n_per_group = analysis.solve_power(
        effect_size=effect_size, alpha=alpha, power=power, ratio=1.0
    )
    return int(np.ceil(n_per_group))


def select_high_risk_segment(df, percentile=HIGH_RISK_PERCENTILE):
    threshold = df["churn_probability"].quantile(percentile)
    segment = df[df["churn_probability"] >= threshold].copy()
    print(f"High-risk threshold (p{int(percentile*100)}): {threshold:.3f}")
    print(f"High-risk segment size: {len(segment)}")
    return segment


def randomize_treatment(segment, random_state=RANDOM_SEED):
    """Simple randomization — 50/50 split between treatment (retention
    offer) and control (no intervention)."""
    rng = np.random.RandomState(random_state)
    segment = segment.copy()
    segment["group"] = rng.choice(["treatment", "control"], size=len(segment), p=[0.5, 0.5])
    return segment


def simulate_outcomes(segment, random_state=RANDOM_SEED+1):
    """
    SIMULATED outcome generation. Control group retained at baseline rate;
    treatment group retained at baseline + assumed uplift. A small amount
    of individual-level noise (based on each customer's own churn
    propensity) is layered in so the simulation isn't a flat coin-flip
    per group.
    """
    rng = np.random.RandomState(random_state)
    segment = segment.copy()

    retention_prob = np.where(
        segment["group"] == "treatment",
        BASELINE_RETENTION_RATE + ASSUMED_UPLIFT,
        BASELINE_RETENTION_RATE
    )

    # Individual variation: customers with higher churn_probability are
    # somewhat less likely to be retained even within their assigned group
    adjustment = (1 - segment["churn_probability"]) * 0.10
    retention_prob = np.clip(retention_prob + adjustment - 0.05, 0.01, 0.99)


    segment["retained"] = rng.binomial(1, retention_prob)

    # Guardrail metric simulation: complaint rate should NOT rise due to
    # the intervention. Simulated as roughly flat across groups with noise.
    base_complaint_rate = 0.04
    complaint_prob = np.where(
        segment["group"] == "treatment", base_complaint_rate + 0.005, base_complaint_rate
    )
    segment["complained"] = rng.binomial(1, complaint_prob)

    return segment


def run_sample_size_check(segment):
    required_n = calculate_required_sample_size(BASELINE_RETENTION_RATE, ASSUMED_UPLIFT)
    available_n_per_group = len(segment) // 2
    print(f"\nRequired sample size per group (to detect {ASSUMED_UPLIFT*100:.0f}pp "
          f"uplift at alpha={ALPHA}, power={POWER}): {required_n}")
    print(f"Available sample size per group: {available_n_per_group}")
    if available_n_per_group < required_n:
        print("WARNING: Segment is UNDERPOWERED for the assumed effect size. "
              "Consider: widening the risk segment, running the test longer, "
              "or targeting a larger MDE expectation.")
    else:
        print("Segment size is sufficient for the assumed effect size.")
    return required_n, available_n_per_group


def run_ab_test_simulation():
    df = pd.read_csv(SCORES_PATH)
    segment = select_high_risk_segment(df)
    run_sample_size_check(segment)

    segment = randomize_treatment(segment)
    segment = simulate_outcomes(segment)

    os.makedirs("outputs", exist_ok=True)
    segment.to_csv(AB_RESULTS_PATH, index=False)
    print(f"\nSimulated A/B test results saved to {AB_RESULTS_PATH}")
    return segment


if __name__ == "__main__":
    run_ab_test_simulation()