import torch

from bitsandbytes.functional import (
    dequantize_4bit,
)


class ModelProfiler:
    """
    Profiles every transformer layer and
    constructs residual tensors.

    This class is model-agnostic and works
    for TinyLlama, Llama2, Llama3, Mistral,
    Qwen, Phi, etc., as long as the model
    exposes `model.layers`.
    """

    def __init__(self):

        pass

    def profile(
        self,
        fp16_model,
        nf4_model,
    ):

        profiles = []

        for idx, (

            fp16_layer,
            nf4_layer,

        ) in enumerate(

            zip(

                fp16_model.model.layers,

                nf4_model.model.layers,

            )

        ):

            modules = [

                "q_proj",

                "k_proj",

                "v_proj",

                "o_proj",

                "gate_proj",

                "up_proj",

                "down_proj",

            ]

            for module_name in modules:

                if hasattr(
                    fp16_layer.self_attn,
                    module_name,
                ):

                    fp16_module = getattr(

                        fp16_layer.self_attn,

                        module_name,

                    )

                    nf4_module = getattr(

                        nf4_layer.self_attn,

                        module_name,

                    )

                elif hasattr(
                    fp16_layer.mlp,
                    module_name,
                ):

                    fp16_module = getattr(

                        fp16_layer.mlp,

                        module_name,

                    )

                    nf4_module = getattr(

                        nf4_layer.mlp,

                        module_name,

                    )

                else:

                    continue

                fp16_weight = (

                    fp16_module.weight

                    .detach()

                    .float()

                    .cpu()

                )

                nf4_weight = (

                    dequantize_4bit(

                        nf4_module.weight.data,

                        quant_state=nf4_module.weight.quant_state,

                    )

                    .detach()

                    .float()

                    .cpu()

                )

                residual = (

                    fp16_weight

                    -

                    nf4_weight

                )

                profiles.append(

                    {

                        "layer": idx,

                        "module": module_name,

                        "fp16": fp16_weight,

                        "nf4": nf4_weight,

                        "residual": residual,

                    }

                )

        return profiles