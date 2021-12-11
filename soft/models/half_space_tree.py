from pysad.models.half_space_trees import HalfSpaceTrees
from .mixin import ModelMixin

import numpy as np


class HalfSpaceTrees(HalfSpaceTrees, ModelMixin):
    def evaluate(self, X, y, metric):
        scores = []
        for instance in X:
            score = self.fit_score_partial(instance)
            scores.append(score)

        scores = np.array(score).squeeze()

        candidate_tresholds = np.unique(scores)

        f1_scores = []
        for threshold in candidate_tresholds:
            labels = np.where(scores < threshold, 0, 1)
            f1_scores.append(metric.evaluate(labels))

        q = list(zip(f1_scores, thresholds))

        thres = sorted(q, reverse=True, key=lambda x: x[0])[0][1]
        threshold = thres
        arg = np.where(thresholds == thres)

        return np.where(scores < threshold, 0, 1)
