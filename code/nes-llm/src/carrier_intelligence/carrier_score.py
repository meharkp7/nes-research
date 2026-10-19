import torch


class CarrierScore:
    """
    Multi-dimensional Carrier Utility Scoring.

    Pipeline

        Feature Matrix
              ↓
      Column-wise Normalization
              ↓
      Data-driven Feature Weighting
              ↓
      Weighted Feature Fusion
              ↓
      Non-linear Amplification
              ↓
         Carrier Utility Score
    """

    def __init__(

        self,

        gamma=2.5,

        eps=1e-8,

    ):

        self.gamma = gamma
        self.eps = eps

    ##########################################################

    def normalize(

        self,

        features,

    ):

        features = features.float()

        minimum = features.min(
            dim=0
        ).values

        maximum = features.max(
            dim=0
        ).values

        return (

            features - minimum

        ) / (

            maximum - minimum + self.eps

        )

    ##########################################################

    def compute(

        self,

        features,

    ):

        ##################################################
        # Normalize every feature independently
        ##################################################

        features = self.normalize(
            features
        )

        ##################################################
        # Data-driven feature importance
        #
        # Features with higher variance
        # contribute more because they
        # discriminate carriers better.
        ##################################################

        importance = features.std(
            dim=0
        )

        importance = importance / (

            importance.sum()

            + self.eps

        )

        ##################################################
        # Weighted feature fusion
        ##################################################

        score = (

            features

            * importance

        ).sum(
            dim=1
        )

        ##################################################
        # Non-linear carrier amplification
        ##################################################

        score = score.pow(
            self.gamma
        )

        return score

    ##########################################################

    def rank(

        self,

        score,

    ):

        return torch.argsort(

            score,

            descending=True,

        )

    ##########################################################

    def top_k(

        self,

        score,

        k,

    ):

        return self.rank(
            score
        )[:k]

    ##########################################################

    def statistics(

        self,

        score,

    ):

        return {

            "mean":
            score.mean().item(),

            "std":
            score.std().item(),

            "min":
            score.min().item(),

            "max":
            score.max().item(),

        }