import abc

import torch

from eta_core import EntropyCriterion, SimilarityCriterion

from sedna.common.class_factory import ClassFactory, ClassType
from sedna.algorithms.hard_example_mining.hard_example_mining import BaseFilter


@ClassFactory.register(ClassType.HEM, alias="ETA")
class ETAFilter(BaseFilter, abc.ABC):
    def __init__(self, ent_threshold: float, sim_threshold: float, **kwargs):
        self.ent_threshold = float(ent_threshold)
        self.sim_threshold = float(sim_threshold)

        self.entropy_criterion = EntropyCriterion(self.ent_threshold)
        self.similarity_criterion = SimilarityCriterion(d_margin=self.sim_threshold)

    def __call__(self, infer_result=None) -> bool:
        """judge the img is hard sample or not.

        Parameters
        ----------
        infer_result: array_like
            prediction classes list, such as
            [class1-score, class2-score, class2-score,....],
            where class-score is the score corresponding to the class,
            class-score value is in [0,1], who will be ignored if its
            value not in [0,1].

        Returns
        -------
        is hard sample: bool
            `True` means hard sample, `False` means not.
        """

        if not infer_result:
            # if invalid input, return False
            return False

        # infer_result should be a tensor with shape (N, D),
        # N is the batch size, D is the number of classes
        infer_result = torch.tensor(infer_result)

        # according to ETA algorithm, we first check entropy of the image
        ent_remain_ids = self.entropy_criterion.filter_out(infer_result)

        if ent_remain_ids.numel() <= 0:
            return False
        else:
            # then we check the similarity of the image
            filtered_outputs = infer_result[ent_remain_ids]
            sim_remain_ids = self.similarity_criterion.filter_out(filtered_outputs)
            # filtered_outputs = infer_result[sim_remain_ids]

            return sim_remain_ids.numel() > 0
