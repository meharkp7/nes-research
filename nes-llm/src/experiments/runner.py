"""
Experiment runner.

Loads each model at most once, runs the requested experiments in
dependency order, writes a standardized artifact per experiment, and
updates the manifest.

Three rules shape the control flow:

1.  A completed cell (PASS **or** FAIL) is skipped by default. Rerunning
    a FAIL until it passes would be result-shopping, so forcing a rerun
    requires an explicit flag and archives the previous artifact.

2.  A missing prerequisite does not block downstream work by silently
    passing it. The dependent experiment runs, but if its own input is
    genuinely unavailable it reports NOT_RUN with the reason.

3.  A model that raises does not abort the suite. The error is recorded
    against that model's cells and the run continues, because losing a
    7B model's results must not discard another model's completed work.
"""

import importlib
import time
import traceback
from typing import Any, Dict, List, Optional

import torch

from src.experiments import manifest as manifest_mod
from src.experiments.artifact_manager import save_json
from src.experiments.experiment_registry import (
    EXPERIMENTS,
    TARGET_MODELS,
    architecture_status,
    experiment_names,
    resolve_order,
)
from src.experiments.model_context import ModelContext, seed_everything
from src.experiments.paths import model_slug, result_path
from src.experiments.residual_source import get_residuals


class Runner:
    def __init__(
        self,
        models: List[str],
        experiments: Optional[List[str]] = None,
        force: bool = False,
        seed: int = 42,
        allow_neural_training: bool = False,
        exp9_download: bool = False,
        recompute_exp5: bool = False,
    ):
        self.models = list(models)
        self.experiments = resolve_order(experiments or experiment_names())
        self.force = force
        self.seed = seed
        self.allow_neural_training = allow_neural_training
        self.exp9_download = exp9_download
        self.recompute_exp5 = recompute_exp5

        self.manifest = manifest_mod.load()
        self.contexts: Dict[str, ModelContext] = {}
        self.skipped: List[str] = []
        self.executed: List[str] = []

    # -----------------------------------------------------------------
    # Model lifecycle
    # -----------------------------------------------------------------

    def context_for(self, model_id: str) -> ModelContext:
        """Return a context, loading the model and residuals once."""
        if model_id in self.contexts:
            return self.contexts[model_id]

        spec = next(
            (m for m in TARGET_MODELS if m["model_id"] == model_id),
            None,
        )
        if spec is None:
            raise KeyError(f"Model {model_id} is not in the registry")

        context = ModelContext(
            model_id=model_id,
            family=spec["family"],
            expected_layers=int(spec["expected_layers"]),
        )
        self.contexts[model_id] = context

        arch = architecture_status(context.family)
        if arch == "NOT_VALIDATED":
            print(
                f"  [runner] NOTE: architecture {context.family} has "
                "registry support but no experimental validation."
            )

        print(f"\n{'=' * 70}")
        print(f"LOADING {model_id} ({context.family})")
        print("=" * 70)

        # --- models (only if some experiment needs them) -----------
        if self._needs(model_id, "requires_models"):
            from src.model.model_loader import load_model_pair

            try:
                models = load_model_pair(model_id)
                context.models = models
                context.actual_layers = len(models[0].model.layers)

                if not context.layer_count_matches_expected:
                    print(
                        f"  [runner] layer count mismatch: expected "
                        f"{context.expected_layers}, actual "
                        f"{context.actual_layers}"
                    )
            except Exception as exc:
                context.model_error = f"{type(exc).__name__}: {exc}"
                print(f"  [runner] model load failed: {context.model_error}")

        # --- residuals ---------------------------------------------
        if self._needs(model_id, "requires_residuals"):
            try:
                residuals, provenance = get_residuals(
                    model_id=model_id,
                    family=context.family,
                    num_layers=(
                        context.actual_layers or context.expected_layers
                    ),
                    use_cache=context.models is None,
                    models=context.models,
                )
                context.residuals = residuals
                context.residual_provenance = provenance
            except Exception as exc:
                context.residual_provenance = {
                    "source": "unavailable",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"  [runner] residual extraction failed: {exc}")

        return context

    def _needs(self, model_id: str, flag: str) -> bool:
        """Whether any selected experiment requires models/residuals."""
        if model_id not in self.models:
            return False
        return any(
            EXPERIMENTS[name].get(flag)
            for name in self.experiments
        )

    # -----------------------------------------------------------------
    # Experiment execution
    # -----------------------------------------------------------------

    def _already_done(self, experiment: str, model_id: str) -> bool:
        if self.force:
            return False
        return manifest_mod.is_completed(self.manifest, experiment, model_id)

    def run_experiment(
        self,
        experiment: str,
        model_id: str,
    ) -> Optional[Dict[str, Any]]:
        spec = EXPERIMENTS[experiment]

        # Exp8 aggregates; Exp9 targets its own checkpoints.
        if experiment == "exp8":
            return self._run_exp8()

        if experiment == "exp9":
            return self._run_exp9()

        key = f"{experiment}::{model_id}"

        if self._already_done(experiment, model_id):
            self.skipped.append(key)
            print(f"  [{experiment}] {model_id}: already completed, skipping")
            return None

        context = self.context_for(model_id)
        module = importlib.import_module(spec["module"])

        kwargs = {}
        if experiment == "exp7_neural":
            kwargs["allow_training"] = self.allow_neural_training
        elif experiment == "exp5":
            # Separate from --force on purpose. --force means "redo this
            # cell"; recomputing the three-way PPL is a much heavier
            # decision, so it needs its own explicit opt-in. Otherwise a
            # routine artifact refresh would silently trigger an hours-
            # long MPS PPL run.
            kwargs["force"] = self.recompute_exp5

        started = time.time()

        try:
            artifact = module.run(context, **kwargs)
        except Exception as exc:
            artifact = {
                "experiment": experiment,
                "title": experiment,
                "model_id": model_id,
                "family": context.family,
                "configuration": {},
                "metrics": {},
                "thresholds": {},
                "status": manifest_mod.ERROR,
                "gate_status": manifest_mod.ERROR,
                "reproducibility": context.reproducibility(),
                "notes": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "source": "run",
            }

        artifact["duration_seconds"] = round(
            time.time() - started, 2
        )
        artifact.setdefault("model_id", model_id)
        artifact.setdefault("family", context.family)

        path = result_path(experiment, model_id)
        save_json(
            path,
            artifact,
            archive_previous=True,
            reason=f"rerun_{experiment}",
        )

        manifest_mod.record(
            self.manifest,
            experiment=experiment,
            model_id=model_id,
            family=context.family,
            status=artifact["status"],
            gate_status=artifact.get("gate_status"),
            artifact_path=str(path),
            configuration=artifact.get("configuration"),
            source=artifact.get("source", "run"),
            notes=artifact.get("notes", ""),
            metrics=_summary_metrics(artifact),
            extra={"duration_seconds": artifact.get("duration_seconds")},
        )

        self.executed.append(key)
        print(
            f"  [{experiment}] {model_id}: {artifact['status']}"
        )

        # Exp2 additionally mirrors the historical profile filename.
        if experiment == "exp2" and artifact.get("per_layer"):
            try:
                mirror = module.write_legacy_profile(context, artifact)
                print(f"  [exp2] profile mirrored to {mirror}")
            except Exception:
                pass

        return artifact

    def _run_exp8(self) -> Optional[Dict[str, Any]]:
        """Run the real cross-model orchestrator, then normalize its table."""
        key = "exp8::aggregate"

        if self._already_done("exp8", "aggregate"):
            self.skipped.append(key)
            print("  [exp8] already completed, skipping")
            return None

        print("\n" + "=" * 70)
        print("RUNNING EXP8 — REAL CROSS-MODEL ORCHESTRATOR")
        print("=" * 70)

        import src.model.exp8_real_cross_model_table as real_exp8

        # The orchestrator is authoritative; point it at this run's
        # models without editing its module-level list.
        original_models = real_exp8.MODELS
        real_exp8.MODELS = [
            (m["model_id"], m["family"], m["expected_layers"])
            for m in TARGET_MODELS
            if m["model_id"] in self.models
        ]

        try:
            real_exp8.main()
        except SystemExit:
            # Non-zero exit means the table is not all-PASS, which is a
            # legitimate outcome when a gate genuinely failed.
            pass
        except Exception as exc:
            print(f"  [exp8] orchestrator raised: {exc}")
            traceback.print_exc()
        finally:
            real_exp8.MODELS = original_models

        from src.experiments.experiments import exp8_cross_model

        artifact = exp8_cross_model.run(
            contexts=list(self.contexts.values()),
            selected_models=self.models,
        )

        path = result_path("exp8", "cross_model")
        save_json(
            path,
            artifact,
            archive_previous=True,
            reason="rerun_exp8",
        )

        for model_id in self.models:
            row = artifact["metrics"]["rows"].get(model_id)
            if row is None:
                manifest_mod.record(
                    self.manifest,
                    experiment="exp8",
                    model_id=model_id,
                    family=next(
                        m["family"]
                        for m in TARGET_MODELS
                        if m["model_id"] == model_id
                    ),
                    status=manifest_mod.NOT_RUN,
                    gate_status=manifest_mod.NOT_RUN,
                    artifact_path=str(path),
                    source="aggregation",
                    notes=(
                        "No row in the cross-model table for this model."
                    ),
                )
                continue

            manifest_mod.record(
                self.manifest,
                experiment="exp8",
                model_id=model_id,
                family=row["family"],
                status=row["overall_status"],
                gate_status=row["overall_status"],
                artifact_path=str(path),
                configuration=artifact.get("configuration"),
                source="aggregation",
                notes=artifact.get("notes", ""),
                metrics={
                    gate: cell["status"]
                    for gate, cell in row["gates"].items()
                },
            )

        self.executed.append(key)
        print(f"  [exp8] overall {artifact['metrics']['overall_status']}")
        return artifact

    def _run_exp9(self) -> Optional[Dict[str, Any]]:
        key = "exp9::alternative_quant"

        if self._already_done("exp9", "alternative_quant"):
            self.skipped.append(key)
            print("  [exp9] already completed, skipping")
            return None

        from src.experiments.experiments import exp9_alternative_quant

        artifact = exp9_alternative_quant.run(
            download=self.exp9_download
        )

        path = result_path("exp9", "formats")
        save_json(
            path,
            artifact,
            archive_previous=True,
            reason="rerun_exp9",
        )

        for model_id, result in artifact["metrics"]["results"].items():
            manifest_mod.record(
                self.manifest,
                experiment="exp9",
                model_id=model_id,
                family=result.get("family", "unknown"),
                status=result["status"],
                gate_status=result["gate_status"],
                artifact_path=str(path),
                configuration=result.get("configuration"),
                source="run",
                notes=result.get("notes", ""),
                metrics=result.get("metrics"),
                extra={
                    "quantization_format": result.get(
                        "quantization_format"
                    )
                },
            )

        self.executed.append(key)
        print(f"  [exp9] overall {artifact['status']}")
        return artifact

    # -----------------------------------------------------------------
    # Suite
    # -----------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        seed_everything(self.seed)

        started = time.time()

        for experiment in self.experiments:
            print("\n" + "=" * 70)
            print(f"EXPERIMENT {experiment}: {EXPERIMENTS[experiment]['name']}")
            print("=" * 70)

            if experiment in ("exp8", "exp9"):
                self.run_experiment(experiment, "aggregate")
                continue

            for model_id in self.models:
                self.run_experiment(experiment, model_id)

        # Persist the manifest even when a model failed, so partial real
        # results survive.
        manifest_mod.save(self.manifest)

        for context in self.contexts.values():
            context.release()
        self.contexts.clear()

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return {
            "duration_seconds": round(time.time() - started, 2),
            "executed": self.executed,
            "skipped": self.skipped,
            "coverage": manifest_mod.coverage(self.manifest),
            "matrix": manifest_mod.matrix(self.manifest),
        }


def _summary_metrics(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the headline numbers in the manifest, not whole curves."""
    metrics = artifact.get("metrics") or {}

    keep = (
        "ber", "payload_bits", "max_tested_payload_at_ber_zero_bits",
        "kl_divergence", "statistical_detector_accuracy", "accuracy",
        "embedding_specific_delta_pct", "ppl_degradation_pct",
        "nf4_baseline_ppl", "reconstruction_control_ppl",
        "embedded_ppl", "reconstruction_only_delta_pct",
        "ber_at_sigma_0_001", "ber_at_sigma_0_002",
        "fraction_above_threshold", "layer_mismatch",
    )

    return {k: metrics[k] for k in keep if k in metrics}