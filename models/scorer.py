import numpy as np

from sklearn.metrics import make_scorer
import keras.backend as K


def home_credit_scoring_fn(y_true, y_pred):
    """
    Custom scoring function, designed to penalize more false negatives
    by a factor 10 (default). It can be modified in the fn/fp variables
    """
    fn_cost = 10
    fp_cost = 1

    y_pred_binary = (y_pred >= 0.5).astype(int)

    fp = ((y_pred_binary == 1) & (y_true == 0)).sum()
    fn = ((y_pred_binary == 0) & (y_true == 1)).sum()

    score = (fn * fn_cost + fp * fp_cost) / (fn + fp + 1e-7)  # Prevents zero division
    return score


def home_credit_fn_keras(y_true, y_pred):
    """
    Custom scoring function, designed to penalize more false negatives
    by a factor 10 (default). It can be modified in the fn/fp variables
    Designed to work with keras
    """
    fn_cost = 10
    fp_cost = 1

    y_pred_binary = K.cast(K.greater_equal(y_pred, 0.5), dtype='float32')

    fp = K.sum(K.cast(K.tf.logical_and(K.equal(y_pred_binary, 1), K.equal(y_true, 0)), dtype='float32'))
    fn = K.sum(K.cast(K.tf.logical_and(K.equal(y_pred_binary, 0), K.equal(y_true, 1)), dtype='float32'))

    score = (fn * fn_cost + fp * fp_cost) / (fn + fp + K.epsilon())
    return score


def hc_threshold_score(y_true, y_proba):
    """
    Custom scoring function to optimize threshold for classification models. Takes both true labels and
    predicted probabilities as input and returns the optimal decision threshold.
    """
    # Define the cost ratio for false negatives and false positives
    fn_cost = 10
    fp_cost = 1

    # Calculate the false negative and false positive rates for different thresholds
    thresholds = np.arange(0, 1, 0.01)
    fn_rates = []
    fp_rates = []
    for t in thresholds:
        y_pred = (y_proba[:, 1] >= t).astype(int)
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn_rates.append(fn / (fn + fp + 1e-7))
        fp_rates.append(fp / (fn + fp + 1e-7))

    # Calculate the total cost for different thresholds
    total_costs = [fn * fn_cost + fp * fp_cost for fn, fp in zip(fn_rates, fp_rates)]

    # Find the optimal threshold that minimizes total cost
    optimal_threshold_index = np.argmin(total_costs)
    optimal_threshold = thresholds[optimal_threshold_index]

    return optimal_threshold


# creating to scorer objects
home_credit_scorer = make_scorer(home_credit_scoring_fn, greater_is_better=False)
hc_threshold_scorer = make_scorer(hc_threshold_score, greater_is_better=True, needs_proba=True)
