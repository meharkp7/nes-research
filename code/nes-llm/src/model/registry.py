MODEL_REGISTRY = {
'llama': { 'root': 'model.layers', 'mlp': '.mlp', 'attn': '.self_attn' },
'mistral': { 'root': 'model.layers', 'mlp': '.mlp', 'attn': '.self_attn' },
'gemma': { 'root': 'model.layers', 'mlp': '.mlp', 'attn': '.self_attn' },
'qwen': { 'root': 'model.layers', 'mlp': '.mlp', 'attn': '.self_attn' },
'phi': { 'root': 'model.layers', 'mlp': '.mlp', 'attn': '.self_attn' },
'deepseek': { 'root': 'model.layers', 'mlp': '.mlp', 'attn': '.self_attn' },
'olmo': { 'root': 'model.layers', 'mlp': '.mlp', 'attn': '.self_attn' },
'falcon': { 'root': 'transformer.h', 'mlp': '.mlp', 'attn': '.self_attention' },
'mixtral': { 'root': 'model.layers', 'mlp': '.block_sparse_moe', 'attn': '.self_attn' },
}
def get_num_layers(model):
    if hasattr(model, "model"):
        base = model.model
    else:
        base = model

    if hasattr(base, "layers"):
        return len(base.layers)

    raise ValueError(
        f"Could not find transformer layers in {type(model).__name__}"
    )


def get_layer_module(model, family, layer_idx, component):
    if hasattr(model, "model"):
        base = model.model
    else:
        base = model

    if not hasattr(base, "layers"):
        raise ValueError(
            f"Could not find layers in {type(model).__name__}"
        )

    layer = base.layers[layer_idx]

    if component == "mlp":
        return layer.mlp

    if component == "self_attn":
        return layer.self_attn

    raise ValueError(
        f"Unknown component: {component}"
    )