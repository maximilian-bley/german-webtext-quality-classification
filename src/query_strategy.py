import numpy as np
import numpy.typing as npt

from typing import Union
from scipy.sparse import csr_matrix

from small_text.classifiers import Classifier
from small_text.data.datasets import Dataset
from small_text.query_strategies.strategies import QueryStrategy

from separation_margin import sep_margin


def _validate_bounds(parameter_name: str, parameter_value: float):
    if parameter_value < 0.0 or parameter_value > 1.0:
        raise ValueError(f'{parameter_name} must be in the interval [0, 1].')
    return parameter_value


def _label_cardinality_inconsistency(y_pred_proba_unlabeled: npt.NDArray[float],
                                     y_labeled: csr_matrix,
                                     prediction_threshold: float = 0.5) -> float:
    """Computes the label cardinality inconsistency per instance [LG13]_.

    The label cardinality inconsistency is defined by the ($L^2$ norm of the) difference between the number of labels
    of an unlabeled instance and the expected number of labels according to the labeled set.

    Parameters
    ----------
    y_pred_unlabeled : np.ndarray[float]
        Confidence score distribution over all classes of shape (num_samples, num_classes).
    y_labeled : csr_matrix
       Labels of the instances in the labeled pool.
    prediction_threshold : float, default=0.5
       Once the prediction confidence ("proba") exceeds this threshold, the label counts as predicted.

    Returns
    -------
    label_cardinality_inconsistency : np.ndarray[float]
        A numpy array with the label cardinality inconsistency score for every unlabeled instance (i.e. with size
        `y_pred_unlabeled.shape[0]`).


    .. versionadded:: 2.0.0

    ! Ported from: https://github.com/webis-de/small-text/blob/v2.0.0.dev2/small_text/query_strategies/multi_label.py
    """
    if y_labeled.shape[0] == 0:
        average_count_per_labeled_instance = 1
    else:
        average_count_per_labeled_instance = (y_labeled > 0).sum(axis=1).mean()

    count_per_unlabeled_instance = (y_pred_proba_unlabeled > prediction_threshold).sum(axis=1)
    count_per_unlabeled_instance = np.asarray(count_per_unlabeled_instance).ravel()

    label_cardinality_inconsistency = count_per_unlabeled_instance - average_count_per_labeled_instance
    l2_label_cardinality_inconsistency = np.sqrt(np.power(label_cardinality_inconsistency, 2))
    return l2_label_cardinality_inconsistency


def _uncertainty_weighted_label_cardinality_inconsistency(y_pred_proba_unlabeled: npt.NDArray[float],
                                                          y_labeled: csr_matrix,
                                                          uncertainty_weight: float = 0.5,
                                                          eps: float = 0.01) -> float:
    """Computes uncertainty-weighted label cardinality inconsistency per instance [LG13]_.

    The label cardinality inconsistency is defined by the ($L^2$ norm of the) difference between the number of labels
    of an unlabeled instance and the expected number of labels according to the labeled set.

    Parameters
    ----------
    y_pred_proba_unlabeled : npt.NDArray[float]
        Confidence score distribution over all classes of shape (num_samples, num_classes).
    y_labeled : csr_matrix
        Labels of the instances in the labeled pool.
    uncertainty_weight : float, default=0.5
        A weight between 0 and 1 that upweights the margin uncertainty score per label and downweights the
        label cardinality inconsistency per sample. Corresponds to the parameter $\beta$ in [LG13]_.
    eps : float, default=0.01
        A small value to be added to the denominator of the inverse uncertainty.

    Returns
    -------
    label_cardinality_inconsistency : np.ndarray[float]
        A numpy array with the label cardinality inconsistency score for every unlabeled instance (i.e. with size
        `y_pred_unlabeled.shape[0]`).


    .. versionadded:: 2.0.0

    ! Ported from: https://github.com/webis-de/small-text/blob/v2.0.0.dev2/small_text/query_strategies/multi_label.py
    """

    # +1 smoothing, so that in the case of a lci value of 0 the uncertainty is not nulled
    lci = _label_cardinality_inconsistency(y_pred_proba_unlabeled, y_labeled) + 1
    
    inverse_uncertainty = 1 / np.maximum(np.apply_along_axis(lambda x: sep_margin(x), 1, y_pred_proba_unlabeled), eps)

    return inverse_uncertainty ** uncertainty_weight * lci ** (1 - uncertainty_weight)


class AdaptiveActiveLearning(QueryStrategy):
    """Queries the instances which exhibit the maximum inverse margin uncertainty-weighted
    label cardinality inconsistency [LG13]_.

    This strategy is a combination of breaking ties and label cardinaly inconsistency.
    The keyword argument `uncertainty_weight` controls the weighting between those two.

    .. seealso::

        Function :py:func:`uncertainty_weighted_label_cardinality_inconsistency`.
            Function to compute uncertainty-weighted label cardinality inconsistency.

    .. versionadded:: 2.0.0

    ! Ported from: https://github.com/webis-de/small-text/blob/v2.0.0.dev2/small_text/query_strategies/multi_label.py
    """

    def __init__(self, uncertainty_weight: float = 0.5, prediction_threshold: float = 0.5):
        """
        Parameters
        ----------
        uncertainty_weight : float, default=0.5
            Weighting of the query strategy's uncertainty portion (between 0 and 1). A higher number
            prioritizes uncertainty, a lower number prioritizes label cardinality inconsistency.
        prediction_threshold : float, default=0.5
            Prediction threshold at which a confidence estimate is counted as a label.
        """
        self.prediction_threshold = _validate_bounds('Prediction threshold', prediction_threshold)
        self.uncertainty_weight = _validate_bounds('Uncertainty weight', uncertainty_weight)
        self.prediction_threshold = prediction_threshold

    def query(self,
              clf: Classifier,
              dataset: Dataset,
              indices_unlabeled: npt.NDArray[np.uint],
              indices_labeled: npt.NDArray[np.uint],
              y: Union[npt.NDArray[np.uint], csr_matrix],
              n: int = 10) -> np.ndarray:
        self._validate_query_input(indices_unlabeled, n)

        y_pred_proba_unlabeled = clf.predict_proba(dataset[indices_unlabeled])

        scores = _uncertainty_weighted_label_cardinality_inconsistency(y_pred_proba_unlabeled,
                                                                       y,
                                                                       uncertainty_weight=self.uncertainty_weight)

        if len(indices_unlabeled) == n:
            return np.array(indices_unlabeled)

        indices_queried = np.argpartition(-scores, n)[:n]

        return np.array([indices_unlabeled[i] for i in indices_queried])

    def __str__(self):
        return f'AdaptiveActiveLearning(uncertainty_weight={self.uncertainty_weight}, ' \
               f'prediction_threshold={self.prediction_threshold})'
