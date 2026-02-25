import pandas as pd
from datetime import datetime

FEEDBACK_FILE = "data/feedback.csv"


def compute_acceptance_rate(emp_id: str) -> float:
    """
    Returns acceptance rate with smoothing.
    Safe for:
    - Missing file
    - Empty file
    - Case inconsistencies
    """

    try:
        df = pd.read_csv(FEEDBACK_FILE)
    except FileNotFoundError:
        return 0.9  # Neutral prior for new system

    if df.empty:
        return 0.9

    # Filter employee
    emp_feedback = df[df["emp_id"] == emp_id]

    if emp_feedback.empty:
        return 0.9  # New employee neutral weight

    # Case insensitive comparison
    emp_feedback["decision"] = emp_feedback["decision"].str.lower()

    accepted = emp_feedback[emp_feedback["decision"] == "accepted"]

    total = len(emp_feedback)

    # --- Laplace Smoothing ---
    # prevents extreme 0 or 1 for small sample sizes
    smoothed_rate = (len(accepted) + 1) / (total + 2)

    return smoothed_rate
