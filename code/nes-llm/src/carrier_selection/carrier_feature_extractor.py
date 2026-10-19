class CarrierFeatureExtractor:

    def extract(
        self,
        residual_tensor,
        fp16_tensor,
        quantized_tensor,
    ):
        """
        Returns

            feature_matrix

        shape

            [num_coefficients, num_features]
        """