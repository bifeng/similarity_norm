import numpy as np


def _predict_proba_lr(self, X):
    """Probability estimation for OvR logistic regression.

    Positive class probabilities are computed as
    1. / (1. + np.exp(-self.decision_function(X)));
    multiclass is handled by normalizing that over all classes.
    """
    prob = self.decision_function(X)
    prob *= -1
    np.exp(prob, prob)
    prob += 1
    np.reciprocal(prob, prob)
    if prob.ndim == 1:
        return np.vstack([1 - prob, prob]).T
    else:
        # OvR normalization, like LibLinear's predict_probability
        prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
        return prob
