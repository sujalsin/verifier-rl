"""Durable generation and full-state recovery around pinned, unmodified TRL loss.

Torch/TRL imports are lazy: importing this module does not load a model or run
candidate code. Only trainer-produced files in a protected directory are read.
"""

import hashlib
import json
from pathlib import Path

from .evaluation_journal import ReconciliationRequired, persist
from .progress import ProgressLog
from .suites import canonical_json, digest

VERSION = "grpo-full-state-recovery-0.3"
PACKAGES = {"torch": "2.8.0", "transformers": "4.57.1", "trl": "0.28.0", "accelerate": "1.12.0"}
CUDA_WORKSPACE = ":4096:8"


def deterministic_training():
    """Use deterministic kernels, not relaxed comparison tolerances."""
    import os
    import torch
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUDA_WORKSPACE:
        raise ValueError("deterministic CUDA workspace must be set before process startup")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": CUDA_WORKSPACE, "cudnn_benchmark": False,
            "cudnn_deterministic": True}


def read_json(path):
    return json.loads(Path(path).read_text())


def capture_rng():
    import random
    import numpy as np
    import torch
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "cpu": torch.get_rng_state().tolist(), "cuda": [s.tolist() for s in torch.cuda.get_rng_state_all()]}


def restore_rng(state):
    import random
    import numpy as np
    import torch
    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, (list, tuple)) else value
    random.setstate(tuples(state["python"]))
    name, values, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(values, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(torch.tensor(state["cpu"], dtype=torch.uint8))
    torch.cuda.set_rng_state_all([torch.tensor(s, dtype=torch.uint8) for s in state["cuda"]])


def state_hash(value):
    """Hash tensor contents, not nondeterministic serialization/file metadata."""
    import torch
    h = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            h.update(canonical_json(["tensor", str(tensor.dtype), list(tensor.shape)]).encode())
            h.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            h.update(b"dict")
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(canonical_json(["sequence", len(item)]).encode())
            for element in item:
                visit(element)
        else:
            h.update(canonical_json([type(item).__name__, item]).encode())
    visit(value)
    return h.hexdigest()


def generation_once(directory, binding, produce, *, commit, get_rng=capture_rng, set_rng=restore_rng):
    """An existing pending group is replayed, never replaced by a fresh draw."""
    directory = Path(directory)
    intent, completed = directory / "intent.json", directory / "generation.json"
    if completed.exists():
        if not intent.exists() or read_json(intent) != binding:
            raise ValueError("pending rollout binding changed")
        saved = read_json(completed)
        if saved["binding"] != binding or saved["tokens_hash"] != digest(canonical_json(saved["output"])):
            raise ValueError("pending rollout contents changed")
        set_rng(saved["after_rng"])
        return tuple(saved["output"]), True
    if intent.exists():
        raise ReconciliationRequired("generation intent without tokens; do not resample")
    persist(directory, {"intent": binding})
    commit()
    output = produce()
    if (not isinstance(output, tuple) or len(output) != 4 or output[2] is not None
            or output[3] != {} or len(output[0]) != 4 or len(output[1]) != 4
            or any(not ids or any(type(t) is not int or t < 0 for t in ids) for ids in output[0] + output[1])):
        raise ValueError("unsupported TRL generation contract; no silent adaptation")
    saved = {"binding": binding, "output": output, "tokens_hash": digest(canonical_json(output)), "after_rng": get_rng()}
    persist(directory, {"generation": saved})
    commit()
    return output, False


def checkpoint_inventory(path, step):
    path = Path(path)
    files = {p.name: p.stat().st_size for p in path.iterdir() if p.is_file()}
    required = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "config.json", "tokenizer.json"}
    if not required.issubset(files) or any(files[n] <= 0 for n in required):
        raise ValueError("incomplete full-state checkpoint")
    if not any(n.endswith(".safetensors") for n in files):
        raise ValueError("checkpoint model missing")
    if read_json(path / "trainer_state.json")["global_step"] != step:
        raise ValueError("checkpoint step mismatch")
    return files


def latest_checkpoint(directory, binding_hash):
    """A partial newer save is not silently overwritten or skipped."""
    root = Path(directory)
    found = sorted((p for p in root.glob("checkpoint-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[-1]))
    if not found:
        return None
    path = found[-1]
    if not (path / "recovery.json").exists():
        raise ReconciliationRequired("partial checkpoint requires reconciliation")
    receipt = read_json(path / "recovery.json")
    if receipt["version"] != VERSION or receipt["binding_hash"] != binding_hash:
        raise ValueError("checkpoint belongs to a different run/configuration")
    files = checkpoint_inventory(path, receipt["step"])
    if any(files.get(n) != size for n, size in receipt["files"].items()):
        raise ValueError("checkpoint inventory changed")
    return path


def validate_configuration(args, processes, tools):
    # The frozen effective batch is FOUR accumulated one-program microbatches.
    # Saving only at optimizer boundaries needs no partially consumed buffer.
    expected = {"gradient_accumulation_steps": 4, "steps_per_generation": 4,
                "num_iterations": 1, "per_device_train_batch_size": 1,
                "num_generations": 4, "use_vllm": False, "save_only_model": False, "save_steps": 1}
    differences = {k: {"expected": v, "actual": getattr(args, k)} for k,v in expected.items() if getattr(args,k) != v}
    if differences or processes != 1 or tools:
        raise ValueError("unsupported recovery configuration: " + canonical_json(differences))


def trainer_class():
    """Wrap only generation/checkpoint hooks, not GRPO's loss or advantages."""
    import importlib.metadata
    from trl import GRPOTrainer
    for package, expected in PACKAGES.items():
        if importlib.metadata.version(package).split("+")[0] != expected:
            raise ValueError("recovery requires pinned package: " + package)

    class DurableGRPOTrainer(GRPOTrainer):
        def __init__(self, *args, journal, binding_hash, initial_parameter_hash, parameter_hash,
                     commit, after_checkpoint=None, **kwargs):
            self.journal = Path(journal)
            self.binding_hash = binding_hash
            self.boundary_hash = initial_parameter_hash
            self.parameter_hash = parameter_hash
            self.commit = commit
            self.after_checkpoint = after_checkpoint
            self.replayed_groups = []
            super().__init__(*args, **kwargs)
            validate_configuration(self.args, self.accelerator.num_processes, self.tools)

        def _generate_single_turn(self, prompts):
            step = self.state.global_step
            binding = {"version": VERSION, "binding_hash": self.binding_hash, "step": step,
                       "parameter_hash": self.boundary_hash, "prompts_hash": digest(canonical_json(prompts))}
            parent_generate = super()._generate_single_turn
            with ProgressLog(self.journal.name, label="ROLLOUT") as progress:
                progress.stage("generate_or_reuse", update=step + 1)
                output, reused = generation_once(self.journal / f"group-{step:02d}", binding,
                    lambda: parent_generate(prompts), commit=self.commit)
            if reused:
                self.replayed_groups.append(step)
            print("ROLLOUT READY", self.journal.name, step + 1, "reused", reused, flush=True)
            return output

        def _save_checkpoint(self, model, trial):
            step = self.state.global_step
            if self._step != 4 * step:
                raise ValueError("checkpoint is not a completed four-microbatch boundary")
            with ProgressLog(self.journal.name, label="CHECKPOINT") as progress:
                progress.stage("save_weights_optimizer_scheduler_rng", update=step)
                super()._save_checkpoint(model, trial)
                path = Path(self._get_output_dir(trial)) / f"checkpoint-{step}"
                progress.stage("verify_full_state", update=step)
                self.boundary_hash = self.parameter_hash(self.model)
                receipt = {"version": VERSION, "binding_hash": self.binding_hash, "step": step,
                           "trl_step": self._step, "parameter_hash": self.boundary_hash,
                           "optimizer_hash": state_hash(self.optimizer.state_dict()),
                           "scheduler_hash": state_hash(self.lr_scheduler.state_dict()),
                           "rng_hash": digest(canonical_json(capture_rng())),
                           "files": checkpoint_inventory(path, step)}
                persist(path, {"recovery": receipt})
                persist(self.journal / "boundaries", {str(step): receipt})
                progress.stage("commit_full_state", update=step)
                self.commit()
            if self.after_checkpoint:
                self.after_checkpoint(self, path, receipt)

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            receipt = read_json(Path(resume_from_checkpoint) / "recovery.json")
            if receipt["binding_hash"] != self.binding_hash or receipt["version"] != VERSION:
                raise ValueError("resume binding differs")
            checkpoint_inventory(resume_from_checkpoint, receipt["step"])
            super()._load_from_checkpoint(resume_from_checkpoint, model=model)
            self.boundary_hash = self.parameter_hash(self.model)
            if self.boundary_hash != receipt["parameter_hash"]:
                raise ValueError("restored weights differ")
            self._step = receipt["trl_step"]
            self._buffered_inputs = None

        def _load_optimizer_and_scheduler(self, checkpoint):
            super()._load_optimizer_and_scheduler(checkpoint)
            if checkpoint:
                receipt = read_json(Path(checkpoint) / "recovery.json")
                if (state_hash(self.optimizer.state_dict()) != receipt["optimizer_hash"]
                        or state_hash(self.lr_scheduler.state_dict()) != receipt["scheduler_hash"]):
                    raise ValueError("restored optimizer/scheduler differs")

        def _load_rng_state(self, checkpoint):
            super()._load_rng_state(checkpoint)
            if checkpoint and digest(canonical_json(capture_rng())) != read_json(Path(checkpoint) / "recovery.json")["rng_hash"]:
                raise ValueError("restored RNG differs")

    return DurableGRPOTrainer
