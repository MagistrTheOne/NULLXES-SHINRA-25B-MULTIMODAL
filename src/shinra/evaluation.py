"""Task-level metrics. Inputs are evaluated predictions, not generated claims."""

import math


def exact_match(predictions, targets):
    if len(predictions) != len(targets) or not targets:
        raise ValueError("Nonempty aligned predictions and targets required")
    return sum(prediction == target for prediction, target in zip(predictions, targets)) / len(targets)


def effective_context(results, threshold=0.9):
    """Require every task at a length, not an average hiding failed retrieval tasks."""
    accepted = []
    for length, tasks in sorted(results.items(), key=lambda item: int(item[0])):
        if not tasks or any(not math.isfinite(score) or score < threshold for score in tasks.values()):
            break
        accepted.append(int(length))
    return max(accepted, default=0)


def world_metrics(predicted, target, valid=None):
    error = (predicted.float() - target.float()).square().mean(-1)
    if valid is not None:
        error = error[valid.bool()]
    if error.numel() == 0:
        raise ValueError("No valid world states")
    return {"latent_mse": float(error.mean()), "latent_rmse": float(error.mean().sqrt())}
