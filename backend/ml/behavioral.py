"""Synthetic behavioural context for the ablation study.

The public text datasets contain no mood logs or interaction history, so Experiment 4 attaches
*simulated* behavioural features drawn from label-conditional distributions with deliberate overlap
and noise. Results from Experiment 4 therefore measure the pipeline's ability to use such signals,
not real-world predictive value, and the production model does not use these features for training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.nlp.features import BehavioralContext

# label -> (mood mean on 1..5, mood slope mean, p(prior high risk), msgs/day mean, neg ratio mean)
_PROFILES = {
    "low": (3.4, 0.0, 0.03, 3.0, 0.30),
    "moderate": (2.8, -0.08, 0.10, 4.0, 0.50),
    "high": (2.4, -0.15, 0.25, 5.0, 0.60),
}


def simulate(labels: pd.Series, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label in labels:
        mood_mu, slope_mu, p_prior, msgs_mu, neg_mu = _PROFILES[label]
        n = int(rng.integers(0, 8))
        start = rng.normal(mood_mu, 0.9)
        slope = rng.normal(slope_mu, 0.2)
        moods = np.clip(start + slope * np.arange(n) + rng.normal(0, 0.8, n), 1, 5).round().tolist()
        ctx = BehavioralContext(
            recent_moods=moods,
            msgs_24h=int(rng.poisson(msgs_mu)),
            prior_high_risk_7d=int(rng.random() < p_prior),
            recent_negative_ratio=float(np.clip(rng.normal(neg_mu, 0.2), 0, 1)),
        )
        rows.append(ctx.as_features())
    return pd.DataFrame(rows, index=labels.index)
