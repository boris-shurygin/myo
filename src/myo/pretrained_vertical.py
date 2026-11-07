"""PPO with frozen or adapting features from a separately trained body predictor."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .body_predictor import contract as body_contract
from .detached_vertical import DetachedVerticalEnv, DetachedPolicy, DetachedPPO, make_model as base_model
from .vertical_muscles import VerticalConfig

TRIALS = ("random_frozen", "pretrained_frozen", "pretrained_online")
DESIGN = {
    "schema": "myo.pretrained_vertical.v1",
    "features": "detached physical8->4x128 SiLU; actor/critic each retain private branches",
    "normalization": "pretraining whole-train delta statistics, frozen in all three trials",
    "ppo": "unchanged detached PPO; exact full-rollout KL <= .03 with optimizer rollback",
    "online": "fresh predictor RMSprop; 10 epochs on latest rollout after PPO; no replay or KL guard",
    "runtime": "predictor weights and normalization bundled inside policy checkpoint",
}


def predictor_identity(metadata):
    return {key: metadata[key] for key in ("contract", "checkpoint_sha256", "dataset_sha256", "update")}


class PretrainedVerticalEnv(DetachedVerticalEnv):
    def __init__(self, identity, trial="pretrained_frozen", config=None, episode_seconds=4.):
        if trial not in TRIALS or identity["contract"] != body_contract():
            raise ValueError("Pretrained predictor identity/contract mismatch")
        for key in ("checkpoint_sha256", "dataset_sha256"):
            if len(identity[key]) != 64 or any(c not in "0123456789abcdef" for c in identity[key]):
                raise ValueError("Pretrained predictor identity hash mismatch")
        if not isinstance(identity["update"], int) or identity["update"] < 0:
            raise ValueError("Pretrained predictor update mismatch")
        super().__init__(config, episode_seconds, "online" if trial == "pretrained_online" else "frozen")
        if self.physical.config != VerticalConfig(**identity["contract"]["config"]):
            raise ValueError("Pretrained predictor physical configuration mismatch")
        self.transfer_trial, self.body_identity = trial, identity

    def contract(self):
        return {**super().contract(), "pretrained_design": DESIGN, "trial": self.transfer_trial,
            "predictor_identity": {"initialization": self.transfer_trial, "pretraining": self.body_identity},
            "transfer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def make_model(env, trial, predictor, identity, **kwargs):
    if trial not in TRIALS:
        raise ValueError("Unknown pretrained-predictor trial")
    model = base_model(env, "online" if trial == "pretrained_online" else "frozen", **kwargs)
    if trial != "random_frozen":
        predictor.copy_into_policy(model.policy)
    else:
        # Match output units without giving the random-feature control learned weights.
        with torch.no_grad():
            for name in ("delta_mean", "delta_std", "delta_calibrated"):
                getattr(model.policy, name).copy_(getattr(predictor, name))
    if not bool(model.policy.delta_calibrated):
        raise ValueError("Pretrained predictor needs frozen training normalization")
    model.transfer_trial, model.body_identity = trial, identity
    return model


def load_pretrained(path, episode_seconds=4.):
    path = Path(path)
    metadata = json.loads(Path(str(path)+".metadata.json").read_text())
    saved = metadata["contract"]
    env = PretrainedVerticalEnv(saved["predictor_identity"]["pretraining"], saved["trial"],
        VerticalConfig(**saved["physical"]["config"]), episode_seconds)
    if saved != env.contract():
        raise ValueError("Pretrained policy contract mismatch")
    if metadata["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Pretrained policy checkpoint hash mismatch")
    model = DetachedPPO.load(path, device="cpu")
    if (not isinstance(model.policy, DetachedPolicy) or model.prediction_coef != 0.
            or not bool(model.policy.delta_calibrated)):
        raise ValueError("Pretrained policy architecture/normalization mismatch")
    if model.num_timesteps != metadata["trained_steps"]:
        raise ValueError("Pretrained policy step counter mismatch")
    if (model.body_identity != env.body_identity or model.transfer_trial != env.transfer_trial
            or model.train_predictor != (env.transfer_trial == "pretrained_online")):
        raise ValueError("Pretrained policy predictor identity/mode mismatch")
    return model, env, metadata
