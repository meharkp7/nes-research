"""
Experiment registry and dependency graph.

Separates three things the handoff §23 asks to keep apart:

    registry         what experiments exist and what their gates are
    dependency graph which experiment needs which earlier experiment
    runner           how a selected experiment is actually executed

Gates live here, not inside individual experiment bodies, so that no
experiment can quietly relax its own threshold (§25 rule 12).
"""

from typing import Dict, List

# ---------------------------------------------------------------------
# Thresholds — the research gates.
#
# These are never changed to make an experiment pass. Changing one is a
# research decision, recorded in the manifest notes, not a code fix.
# ---------------------------------------------------------------------

THRESHOLDS = {
    "exp1": {
        "description": "Residual extraction succeeds and QACI accepts the result.",
        "max_allowed_layer_mismatch": 0,
    },
    "exp2": {
        "description": "Residual fingerprint is informative.",
        "min_mean_magnitude": 0.002,
        "min_fraction_layers_above": 0.80,
    },
    "exp3": {
        "description": "Clean embed -> extract -> decrypt round trip.",
        "max_ber": 0.0,
    },
    "exp4": {
        "description": "Capacity curve: payload sizes recovering at BER=0.",
        "min_bits_required_by_guide": 500_000,
    },
    "exp5": {
        "description": "Perplexity degradation attributable to embedding.",
        "max_ppl_degradation_pct": 2.0,
    },
    "exp6": {
        "description": "BER under additive Gaussian noise.",
        "max_ber_at_sigma_0_001": 0.02,
        "max_ber_at_sigma_0_002": 0.10,
    },
    "exp7_statistical": {
        "description": "Statistical undetectability of the embedding.",
        "max_kl_divergence": 0.05,
        "max_detector_accuracy": 0.55,
    },
    "exp7_neural": {
        "description": "Carrier-centered neural steganalysis stress test.",
        "max_detector_accuracy": 0.55,
    },
    "exp11": {
        "description": "LWE grid width vs both gates (robustness + detectability).",
        "max_ber_at_sigma_0_001": 0.02,
        "max_detector_accuracy": 0.55,
    },
    "exp8": {
        "description": "Cross-model aggregation of G2-G6.",
        "required_gates": [
            "g2_qaci",
            "g3_ber",
            "g4_ppl",
            "g5_robustness",
            "g6_statistical",
            "g6_neural_detector",
        ],
    },
    "exp9": {
        "description": (
            "Clean BER through format-specific dequantization "
            "(GPTQ / AWQ), each verified against its FP16 reference."
        ),
        "max_ber": 0.0,
        # The dequantizer gate, stated here rather than only as a
        # default in `verify_dequantization`, so that ground rule 2
        # ("thresholds live in experiment_registry.THRESHOLDS") is
        # true of *every* threshold the suite uses. Exp9 reads these
        # and passes them in; adapters.py keeps the same numbers as
        # its defaults for callers outside the suite.
        "min_dequant_correlation": 0.95,
        "max_dequant_residual_ratio": 0.5,
    },
    "exp13": {
        "description": (
            "Keyless recovery (W4.2): without any key, an attacker "
            "holding only the released model must be at chance reading "
            "the embedded bitstream, and must not locate the LWE grid "
            "width to within 1% from the weights alone."
        ),
        "min_keyless_ber": 0.5,
        "min_width_search_relative_error": 0.01,
    },
    "exp14": {
        "description": (
            "Blind-patch adversary (W3.2): an attacker who does not "
            "know carrier positions must be at chance. The 55% number "
            "is exp7_neural's existing gate, reused, not a new "
            "threshold; the carrier-centered control arm is a "
            "validity check and carries no gate of its own."
        ),
        "max_blind_detector_accuracy": 0.55,
    },
    "exp15": {
        "description": (
            "LWE fidelity (W2): three-way perplexity in exp5's "
            "protocol — only reconstruction-control to embedded is "
            "attributable to the payload. The 2% number is exp5's own "
            "threshold, reused, not a new one."
        ),
        "max_ppl_degradation_pct": 2.0,
    },
    "exp16": {
        "description": (
            "Cross-scheme detector (W3.1): BOTH cross directions — "
            "sign-trained tested on LWE, and LWE-trained tested on "
            "sign — must stay at chance. The 55% number is "
            "exp7_neural's existing gate, reused, not a new "
            "threshold; within-scheme controls carry no gate but "
            "claim_audit requires them to clear the line."
        ),
        "max_cross_scheme_detector_accuracy": 0.55,
    },
}


# ---------------------------------------------------------------------
# Target models (§5).
#
# expected_layers is a guide-level expectation, never an assumption. The
# real orchestrator records both expected and actual layer counts and
# flags a mismatch rather than silently trusting either.
# ---------------------------------------------------------------------

TARGET_MODELS: List[Dict] = [
    {
        "model_id": "meta-llama/Llama-3.1-8B",
        "family": "llama",
        "expected_layers": 32,
    },
    {
        "model_id": "mistralai/Mistral-7B-v0.3",
        "family": "mistral",
        "expected_layers": 32,
    },
    {
        "model_id": "google/gemma-2-9b",
        "family": "gemma",
        "expected_layers": 42,
    },
    {
        "model_id": "Qwen/Qwen2.5-7B",
        "family": "qwen",
        "expected_layers": 28,
    },
    {
        "model_id": "Qwen/Qwen2.5-3B",
        "family": "qwen",
        "expected_layers": 36,
    },
    {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "family": "llama",
        "expected_layers": 22,
    },
    {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "family": "phi3",
        "expected_layers": 32,
    },
]

# Architectures claimed by the registry versus architectures with real
# experimental evidence (§6). MoE/Falcon are NOT validated merely
# because registry entries could be written for them.
ARCHITECTURE_SUPPORT = {
    "llama": "VALIDATED",
    "mistral": "VALIDATED",
    "gemma": "VALIDATED",
    "qwen": "VALIDATED",
    "phi3": "VALIDATED",
    "falcon": "NOT_VALIDATED",
    "mixtral": "NOT_VALIDATED",
    "moe": "NOT_VALIDATED",
}


def get_model(model_id: str) -> Dict:
    for entry in TARGET_MODELS:
        if entry["model_id"] == model_id:
            return dict(entry)

    raise KeyError(f"Unknown model: {model_id}")


def expected_layers(model_id: str) -> int:
    return int(get_model(model_id)["expected_layers"])


def family_of(model_id: str) -> str:
    return str(get_model(model_id)["family"])


def is_known_architecture(family: str) -> bool:
    return family in ARCHITECTURE_SUPPORT


def architecture_status(family: str) -> str:
    return ARCHITECTURE_SUPPORT.get(family, "UNKNOWN")


# ---------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------

EXPERIMENTS: Dict[str, Dict] = {
    "exp1": {
        "name": "Architecture Probe",
        "gate_key": "exp1",
        "depends_on": [],
        # No model load. Loading NF4 + FP16 together costs ~19GB for a 7B
        # model and OOMs MPS, purely to read a layer count the residual
        # dictionary already states. Everything from exp1 to exp7 then
        # runs cache-only.
        "requires_models": False,
        "requires_residuals": True,
        "module": "src.experiments.experiments.exp1_architecture",
    },
    "exp2": {
        "name": "Residual Fingerprint",
        "gate_key": "exp2",
        "depends_on": ["exp1"],
        "requires_models": False,
        "requires_residuals": True,
        "module": "src.experiments.experiments.exp2_fingerprint",
    },
    "exp3": {
        "name": "Clean BER",
        "gate_key": "exp3",
        "depends_on": ["exp1", "exp2"],
        "requires_models": False,
        "requires_residuals": True,
        "module": "src.experiments.experiments.exp3_clean_ber",
    },
    "exp4": {
        "name": "Capacity Curve",
        "gate_key": "exp4",
        "depends_on": ["exp3"],
        "requires_models": False,
        "requires_residuals": True,
        "module": "src.experiments.experiments.exp4_capacity",
    },
    "exp5": {
        "name": "Fidelity (real PPL)",
        "gate_key": "exp5",
        "depends_on": ["exp3"],
        # Loaded lazily by the experiment itself: it reuses a recorded
        # three-way PPL result when one exists, and preloading two models
        # would cost exactly the resources that reuse avoids.
        "requires_models": False,
        "requires_residuals": True,
        "module": "src.experiments.experiments.exp5_fidelity",
    },
    "exp6": {
        "name": "Robustness",
        "gate_key": "exp6",
        "depends_on": ["exp3"],
        "requires_models": False,
        "requires_residuals": True,
        "module": "src.experiments.experiments.exp6_robustness",
    },
    "exp7": {
        "name": "Security (statistical)",
        "gate_key": "exp7_statistical",
        "depends_on": ["exp3"],
        "requires_models": False,
        "requires_residuals": True,
        "module": "src.experiments.experiments.exp7_security",
    },
    "exp7_neural": {
        "name": "Security (neural steganalysis)",
        "gate_key": "exp7_neural",
        # Depends on the same embedding evidence as exp7, and tracked as
        # its own cell so its FAIL verdict stays visible next to the
        # statistical PASS rather than being averaged into one number.
        "depends_on": ["exp3"],
        "requires_models": False,
        "requires_residuals": False,
        "module": (
            "src.experiments.experiments.exp7_neural_detector"
        ),
    },
    "exp10": {
        "name": "Strategy Head-to-Head",
        "gate_key": "exp10",
        "depends_on": ["exp3"],
        "requires_models": False,
        "requires_residuals": True,
        "module": "src.experiments.exp10_strategy_comparison",
    },
    "exp11": {
        "name": "LWE Grid Width Frontier",
        "gate_key": "exp11",
        "depends_on": ["exp10"],
        "requires_models": False,
        "requires_residuals": True,
        "module": "src.experiments.exp11_lwe_alpha_pareto",
    },
    "exp8": {
        "name": "Real Cross-Model Table",
        "gate_key": "exp8",
        "depends_on": [
            "exp1", "exp2", "exp3", "exp4",
            "exp5", "exp6", "exp7", "exp7_neural",
        ],
        "requires_models": True,
        "requires_residuals": False,
        "module": "src.experiments.experiments.exp8_cross_model",
    },
    "exp9": {
        "name": "GPTQ / AWQ",
        "gate_key": "exp9",
        "depends_on": ["exp3"],
        "requires_models": False,
        "requires_residuals": False,
        "module": "src.experiments.experiments.exp9_alternative_quant",
    },
}


def experiment_names() -> List[str]:
    return list(EXPERIMENTS.keys())


def gate_for(experiment: str) -> Dict:
    """Return the thresholds for an experiment name or a gate key.

    Accepting either keeps call sites honest: experiment modules refer to
    themselves by registry name, while their gates are named
    ``exp7_statistical`` / ``exp7_neural``. Looking up a gate key as if it
    were an experiment name raises a KeyError at run time, which is how
    that mix-up surfaces when it happens.
    """
    if experiment in EXPERIMENTS:
        return THRESHOLDS[EXPERIMENTS[experiment]["gate_key"]]

    if experiment in THRESHOLDS:
        return THRESHOLDS[experiment]

    raise KeyError(f"Unknown experiment or gate: {experiment}")


def dependencies_of(experiment: str) -> List[str]:
    return list(EXPERIMENTS[experiment]["depends_on"])


def resolve_order(requested: List[str]) -> List[str]:
    """Topologically order the requested experiments by dependency.

    The handoff requires dependency order to be preserved even when an
    experiment is reused from an existing artifact: a reused artifact
    still needs its prerequisites to have been decided.
    """
    if not requested:
        return list(EXPERIMENTS.keys())

    for name in requested:
        if name not in EXPERIMENTS:
            raise KeyError(f"Unknown experiment: {name}")

    ordered: List[str] = []
    visiting = set()

    def visit(name: str) -> None:
        if name in ordered:
            return
        if name in visiting:
            raise ValueError(f"Circular dependency at {name}")
        visiting.add(name)
        for dep in dependencies_of(name):
            if dep in requested:
                visit(dep)
        visiting.discard(name)
        ordered.append(name)

    for name in requested:
        visit(name)

    return ordered


def transitive_dependencies(experiment: str) -> List[str]:
    """All prerequisites of an experiment, in dependency order."""
    seen: List[str] = []

    def visit(name: str) -> None:
        for dep in dependencies_of(name):
            if dep not in seen:
                seen.append(dep)
                visit(dep)

    visit(experiment)
    return seen