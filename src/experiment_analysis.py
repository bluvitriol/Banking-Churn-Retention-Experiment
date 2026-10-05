"""
Analyze the A/B test: hypothesis test, confidence interval, effect size,
and guardrail metric check — with results communicated with appropriate
uncertainty rather than a bare "significant/not significant" verdict.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from statsmodels.stats.proportion import proportions_ztest, proportion_confint

AB_RESULTS_PATH = "outputs/ab_test_results.csv"
ALPHA = 0.05


def compute_group_rates(df, outcome_col):
    summary = df.groupby("group")[outcome_col].agg(["mean", "count", "sum"])
    return summary


def run_two_proportion_ztest(df, outcome_col="retained"):
    treatment = df[df["group"] == "treatment"]
    control = df[df["group"] == "control"]

    successes = np.array([treatment[outcome_col].sum(), control[outcome_col].sum()])
    n_obs = np.array([len(treatment), len(control)])

    z_stat, p_value = proportions_ztest(successes, n_obs)

    rate_treatment = successes[0] / n_obs[0]
    rate_control = successes[1] / n_obs[1]
    diff = rate_treatment - rate_control

    # CI for each group's rate
    ci_treatment = proportion_confint(successes[0], n_obs[0], alpha=ALPHA)
    ci_control = proportion_confint(successes[1], n_obs[1], alpha=ALPHA)

    # Approximate CI for the DIFFERENCE in proportions (normal approximation)
    se_diff = np.sqrt(
        rate_treatment * (1 - rate_treatment) / n_obs[0] +
        rate_control * (1 - rate_control) / n_obs[1]
    )
    z_crit = 1.96  # for alpha=0.05
    ci_diff = (diff - z_crit * se_diff, diff + z_crit * se_diff)

    print(f"\n{outcome_col.upper()} — Hypothesis Test")
    print(f"Treatment rate: {rate_treatment:.3%} (n={n_obs[0]}), 95% CI {ci_treatment}")
    print(f"Control rate:   {rate_control:.3%} (n={n_obs[1]}), 95% CI {ci_control}")
    print(f"Observed difference: {diff:.3%}, 95% CI for difference: "
          f"({ci_diff[0]:.3%}, {ci_diff[1]:.3%})")
    print(f"Z-statistic: {z_stat:.3f}, p-value: {p_value:.4f}")

    if p_value < ALPHA:
        print(f"Result is statistically significant at alpha={ALPHA}. "
              f"However, statistical significance alone doesn't guarantee practical/commercial significance")
    else:
        print(f"Result is NOT statistically significant at alpha={ALPHA}. "
              f"This could reflect a true null effect, an underpowered sample, or a smaller true effect than assumed in the design stage")

    return {
        "rate_treatment": rate_treatment,
        "rate_control": rate_control,
        "diff": diff,
        "ci_diff": ci_diff,
        "z_stat": z_stat,
        "p_value": p_value
    }


def check_guardrail_metric(df, outcome_col="complained"):
    print(f"\nGuardrail Check: {outcome_col}")
    result = run_two_proportion_ztest(df, outcome_col=outcome_col)
    if result["p_value"] < ALPHA and result["diff"] > 0:
        print("GUARDRAIL BREACH: complaint rate rose significantly in treatment. "
              "Recommend holding rollout pending investigation.")
    else:
        print("No evidence of guardrail breach so its safe to consider rollout "
              "on the primary metric result alone.")
    return result


def plot_results(df, outcome_col="retained", save_path="outputs/ab_test_result_plot.png"):
    summary = compute_group_rates(df, outcome_col)
    means = summary["mean"]
    cis = [proportion_confint(row["sum"], row["count"], alpha=ALPHA) for _, row in summary.iterrows()]
    errors = [[means.iloc[i] - ci[0], ci[1] - means.iloc[i]] for i, ci in enumerate(cis)]
    errors = np.array(errors).T

    plt.figure(figsize=(6, 5))
    plt.bar(means.index, means.values, yerr=errors, capsize=8, color=["#1f77b4", "#ff7f0e"])
    plt.ylabel(outcome_col.capitalize() + " rate")
    plt.title(f"{outcome_col.capitalize()} Rate by Group (95% CI)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Result plot saved to {save_path}")


def run_experiment_analysis():
    df = pd.read_csv(AB_RESULTS_PATH)

    
    primary_result = run_two_proportion_ztest(df, outcome_col="retained")
    plot_results(df, outcome_col="retained")

    guardrail_result = check_guardrail_metric(df, outcome_col="complained")


    return primary_result, guardrail_result


if __name__ == "__main__":
    run_experiment_analysis()