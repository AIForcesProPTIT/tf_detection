import tensorflow as tf
from tensorflow import keras

import numpy as np

class MatcherNumpy(object):
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLDS': int,
    }
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        # type: (float, float, bool) -> None
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self,match_quality_matrix:np.ndarray):
        """
            Args:
                match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N predicted elements.

            Returns:
                matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
                [0, M - 1] or a negative value indicating that prediction i could not
                be matched.
        """
        if match_quality_matrix.shape[0] ==0:
            raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
        matched_vals = np.max( match_quality_matrix, axis=0)
        matches = np.argmax(match_quality_matrix, axis=0)

        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = np.logical_and(matched_vals >= self.low_threshold, matched_vals < self.high_threshold)

        matches[below_low_threshold] = -1
        matches[between_thresholds] = -2

        if self.allow_low_quality_matches:
            raise Exception('not implement')
        return matches
class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLDS': int,
    }

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        # type: (float, float, bool) -> None
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def _set_values_using_indicator(self, x, indicator, val):
        """Set the indicated fields of x to val.
        Args:
        x: tensor.
        indicator: boolean with same shape as x.
        val: scalar with value to set.
        Returns:
        modified tensor.
        """
        indicator = tf.cast(indicator, x.dtype)
        return x * (1 - indicator) + val * indicator 

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if tf.shape(match_quality_matrix)[0] == 0:
            raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")


        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals = tf.reduce_max(match_quality_matrix, axis=0)
        matches = tf.argmax(match_quality_matrix, axis=0, output_type=tf.int32)

       

        # print(matched_vals,matches)
       

        # Assign candidate matches with low quality to negative (unassigned) values
        # below_low_threshold = matched_vals < self.low_threshold
        below_low_threshold = tf.less(matched_vals, self.low_threshold)

        between_thresholds = tf.logical_and(
            tf.greater_equal(matched_vals, self.low_threshold),
            tf.greater(self.high_threshold, matched_vals)
        )

        matches = self._set_values_using_indicator(matches,
                                                     below_low_threshold,
                                                     -1)
        matches = self._set_values_using_indicator(matches,
                                                     between_thresholds,
                                                     -2)
        

        if self.allow_low_quality_matches:
            force_match_column_ids = tf.argmax(match_quality_matrix, 1,
                                           output_type=tf.int32)
            # print(force_match_column_ids)
            force_match_column_indicators = tf.one_hot(
                force_match_column_ids, depth=tf.shape(match_quality_matrix)[1])
            # print(force_match_column_indicators)
            force_match_row_ids = tf.argmax(force_match_column_indicators, 0,
                                            output_type=tf.int32)
            force_match_column_mask = tf.cast(
                tf.reduce_max(force_match_column_indicators, 0), tf.bool)
            # print(force_match_column_mask, force_match_row_ids, matches)
            final_matches = tf.where(force_match_column_mask,
                                    force_match_row_ids, matches)
            return final_matches

        return matches



def test_matcher():
    import numpy as np
    similarity = np.array([[1, 1, 1, 3, 1],
                           [2, -1, 2, 0, 4],
                           [3, 0, -1, 0, 0]], dtype=np.float32)
    matcher = Matcher(3,2,allow_low_quality_matches=True)
    out=matcher(similarity)
    print(out)
    print("expected out = [2,-1,-2,0,1]")



    
