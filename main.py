"""
End-to-end pipeline: data prep -> churn modeling -> A/B test simulation
-> experiment analysis.
"""

from src.data_prep import prepare_data
from src.churn_model import run_churn_modeling
from src.ab_test_simulation import run_ab_test_simulation
from src.experiment_analysis import run_experiment_analysis


def main():
    print("STEP 1: Data preparation")
    prepare_data()

    print("\nSTEP 2: Churn propensity modeling")
    run_churn_modeling()

    print("\nSTEP 3: A/B test design and simulation on high-risk segment")
    run_ab_test_simulation()

    print("\nSTEP 4: Experiment analysis")
    run_experiment_analysis()


if __name__ == "__main__":
    main()