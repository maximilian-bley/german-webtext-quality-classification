import numpy as np
from numpy.typing import NDArray

def sep_margin(
    probs: NDArray[np.float64],
    threshold: float = 0.5
) -> float:
    """
    Compute the separation margin for one multi‐label example.

    Parameters
    ----------
    probs : array of shape (n_classes,)
        Predicted probabilities for each class.
    threshold : float, default=0.5
        The cutoff: probs > threshold ⇒ positive; else negative.

    Returns
    -------
    margin : float
        = min(probs_pos) − max(probs_neg).
        If there are no positives or no negatives, returns 0.0.

    Example
    -------

    1. [0.9, 0.6, 0.2, 0.1],  # clear separation
    2. [0.55, 0.51, 0.49, 0.45],  # borderline
    3. [0.2, 0.3, 0.4, 0.1],   # no predicted positives
    4. [0.6, 0.7, 0.8, 0.85],   # no predicted negatives
    5. [0.8, 0.7, 0.6, 0.4],   # some positives & negativ
    6. [0.51, 0.49, 0.48, 0.47],  # very tight separation

    | row | pos scores    | neg scores       | margin            | inv_uncertainty  |
    | --- | ------------- | ---------------- | ----------------- | ---------------- |
    | 1   | {0.9,0.6}     | {0.2,0.1}        | 0.6−0.2 = 0.40    | 1/0.40  = 2.5    |
    | 2   | {0.55,0.51}   | {0.49,0.45}      | 0.51−0.49 = 0.02  | 1/0.02  = 50     |
    | 3   | ∅             | {…}              | 0.00              | 1/0.01 = 100     |
    | 4   | {…}           | ∅                | 0.00              | 1/0.01 = 100     |
    | 5   | {0.8,0.7,0.6} | {0.4}            | 0.6−0.4 = 0.20    | 1/0.20  = 5      |
    | 6   | {0.51}        | {0.49,0.48,0.47} | 0.51−0.49 = 0.02  | 1/0.02  = 50     |

    Reference
    ---------

    Xin Li and Yuhong Guo. 2013. Active learning with multi-label SVM classification. 
    In IJCAI 2013, Proceedings of the 23rd International Joint Conference on Artificial Intelligence, 
    Beijing, China, August 3-9, 2013, pages 1479–1485. IJCAI/AAAI.
    """
    
    pos = probs > threshold
    neg = ~pos

    # give instances with all labels pos or all labels neg max score
    if not pos.any() or not neg.any():
        return 0.0

    min_pos = probs[pos].min()
    max_neg = probs[neg].max()
    return float(min_pos - max_neg)
