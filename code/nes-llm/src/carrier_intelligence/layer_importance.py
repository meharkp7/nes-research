import torch


class LayerImportance:
    """
    Computes a global importance score for a layer.

    The importance descriptor combines

        • Frobenius norm
        • Weight variance
        • Spectral norm

    into a single normalized score.

    Every element in the weight matrix receives
    the same importance value.
    """

    def __init__(

        self,

        eps=1e-8,

    ):

        self.eps = eps

    ##########################################################

    def _normalize(

        self,

        x,

    ):

        return (

            x - x.min()

        ) / (

            x.max()

            - x.min()

            + self.eps

        )

    ##########################################################

    def compute(

        self,

        weight,

    ):

        weight = weight.float()

        ######################################################
        # Frobenius Norm
        ######################################################

        frob = torch.linalg.matrix_norm(

            weight,

            ord="fro",

        )

        ######################################################
        # Variance
        ######################################################

        variance = weight.var()

        ######################################################
        # Spectral Norm
        ######################################################

        spectral = torch.linalg.matrix_norm(

            weight,

            ord=2,

        )

        ######################################################
        # Feature Vector
        ######################################################

        descriptor = torch.tensor(

            [

                frob,

                variance,

                spectral,

            ],

            dtype=torch.float32,

            device=weight.device,

        )

        ######################################################
        # Normalize descriptor
        ######################################################

        descriptor = self._normalize(

            descriptor

        )

        ######################################################
        # Importance Score
        ######################################################

        importance = descriptor.mean()

        ######################################################
        # Expand to matrix shape
        ######################################################

        return torch.full(

            weight.shape,

            importance,

            dtype=torch.float32,

            device=weight.device,

        )


##############################################################

def main():

    weight = torch.randn(

        2048,

        2048,

    )

    estimator = LayerImportance()

    importance = estimator.compute(

        weight

    )

    print()

    print("Shape :", importance.shape)

    print("Mean  :", importance.mean().item())

    print("Std   :", importance.std().item())

    print("Min   :", importance.min().item())

    print("Max   :", importance.max().item())


if __name__ == "__main__":
    main()