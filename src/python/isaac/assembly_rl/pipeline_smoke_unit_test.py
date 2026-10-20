import copy

import pytest
import torch

from isaac.assembly_rl import pipeline_smoke, train
from isaac.assembly_rl.trainer_unit_test import make_trainer


def test_resume_verifier_checks_optimizer_sampler_physics_and_rng(tmp_path):
    torch.set_num_threads(1)
    original = make_trainer()
    original.train_iteration()
    checkpoint = original.save_checkpoint(tmp_path)
    saved = torch.load(checkpoint / "rank-00000.pt", weights_only=True)
    resumed = make_trainer(seed=700)
    resumed.load_checkpoint(checkpoint)
    pipeline_smoke.verify_restored(resumed, saved)
    for component in ("model", "optimizer", "sampler", "environment", "rng"):
        broken = copy.deepcopy(saved)
        if component == "model":
            next(iter(broken[component].values())).add_(1)
        elif component == "optimizer":
            broken[component]["param_groups"][0]["lr"] *= 2
        elif component == "sampler":
            broken[component]["cursor"][0] += 1
        elif component == "environment":
            broken[component]["age"][0] += 1
        else:
            broken[component]["torch"][0] ^= 1
        with pytest.raises(AssertionError):
            pipeline_smoke.verify_restored(resumed, broken)


def test_frozen_evaluation_verifier_detects_parameter_or_rng_changes():
    trainer = make_trainer()
    before = pipeline_smoke.frozen_state(trainer)
    pipeline_smoke.assert_equal(before, pipeline_smoke.frozen_state(trainer))
    torch.rand(1)
    with pytest.raises(AssertionError):
        pipeline_smoke.assert_equal(before, pipeline_smoke.frozen_state(trainer))
    train.restore_random_state(before["rng"])
    with torch.no_grad():
        next(trainer.model.parameters()).add_(1)
    with pytest.raises(AssertionError):
        pipeline_smoke.assert_equal(before, pipeline_smoke.frozen_state(trainer))
