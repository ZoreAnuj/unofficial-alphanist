# Training and evaluation

These commands describe the actual implementation interfaces. They require the simulator runtime and external inputs in [setup](setup.md); they are not a download-and-run release of the full experiment. Run from the repository root. Example paths are local placeholders, and the small environment counts below are conservative examples, not measured capacity claims.

## 1. Generate and validate reset banks

```bash
python -m isaac.assembly_rl.reset_bank_builder \
  --assets data/training_assets --out data/bank_candidates \
  --samples-per-task 500 --num-envs 2 --seed 11 --device cuda:0
```

The builder validates the actual settled robot/object state, grasp retention where applicable, and task-family constraints. Failed proposals remain failures; an IK solution alone is not acceptance. Generation may stop below its requested quota, so inspect `bank.json` and complete missing task/family quotas before proceeding.

Generate held-out candidates with an independent seed, then merge with overlap exclusion:

```bash
python -m isaac.assembly_rl.reset_bank_builder \
  --assets data/training_assets --out data/eval_candidates \
  --samples-per-task 100 --num-envs 2 --seed 100011 --device cuda:0
python -m isaac.assembly_rl.merge_reset_banks \
  --input data/bank_candidates --assets data/training_assets \
  --out data/train_bank --samples-per-task 500
python -m isaac.assembly_rl.merge_reset_banks \
  --input data/eval_candidates --exclude data/train_bank \
  --assets data/training_assets --out data/eval_bank --samples-per-task 100
```

For ten tasks the intended balanced split is 5,000 train and 1,000 held-out states. This is a preparation target, not a statement that those banks ship here. Rejected, duplicate, or excess-family samples do not fill required balanced quotas.

## 2. Qualify capacity

```bash
python -m isaac.assembly_rl.training_capacity_probe \
  --assets data/training_assets --bank data/train_bank \
  --eval-bank data/eval_bank --out runs/capacity \
  --num-envs 2 --eval-num-envs 2
```

Qualify the exact dense task mix, contact allocation, control/physics configuration, rollout memory, PPO update, checkpoint, and evaluation load before increasing parallelism. Do not infer usable training capacity from free GPU memory or bank-generation throughput.

## 3. Train

```bash
python -m isaac.assembly_rl.train \
  --assets data/training_assets --bank data/train_bank \
  --eval-bank data/eval_bank --out runs/assembly \
  --num-envs 2 --eval-num-envs 2 --iterations 25000 \
  --eval-every 100 --eval-seeds 5 --eval-trials-per-task 32 \
  --checkpoint-every 100 --render-every 200 --seed 0 --device cuda:0
```

`PPOConfig` in `policy.py` is the configuration source: 32 rollout steps, five epochs, four minibatches, learning rate 1e-4, gamma 0.99, GAE lambda 0.95, clip 0.2, and entropy coefficient 0.006. The provenance map separates paper settings from referenced-code defaults. The UR7e environment supplies 1,024 XYZ points, five proprioception frames, and seven actions. Its policy rate is 25 Hz; the default horizon is 375 steps per active assembly pair. A control cost supplements binary full-assembly reward without dense assembly-progress shaping.

For a qualified multi-GPU node, replace `python -m` with `torchrun --standalone --nproc_per_node=2 -m` and keep the same module and arguments. Worker counts are examples. Each rank uses its local device, reset sampler, and normalizer history; gradients are synchronized. All ranks must agree on source, assets, physical identity, and configuration. Cross-node launch/network provisioning is deliberately outside this source release.

## 4. Resume and evaluate

Resume by adding `--resume runs/assembly/checkpoint-NNNNNNNN-GENERATION` with an actual checkpoint directory. The manifest, per-rank files, bank/asset identities, RNG, optimizer, sampler, and environment state are checked. Keep the same world size and input identities.

Scheduled evaluation uses a frozen held-out bank, uniformly balanced trials per task, stochastic actions, and five seeds by default. It does not update the policy, running statistics, or SGS history. Outputs include per-task and per-family episode, success, and safety-failure counts. Retain those counts and seed-level results when reporting aggregate performance.

For an integration check that verifies resume, performs one further update, and evaluates:

```bash
python -m isaac.assembly_rl.pipeline_smoke \
  --assets data/training_assets --bank data/train_bank \
  --eval-bank data/eval_bank \
  --checkpoint runs/assembly/checkpoint-NNNNNNNN-GENERATION \
  --out runs/resume_check --num-envs 2 --eval-trials-per-task 2
```

This smoke check intentionally changes the resumed policy by one update; it is not an evaluation-only benchmark. `train.evaluate(model, env, trials_per_task=32, seed=...)` is the frozen evaluation API for a loaded model. Checkpoints may also contain a measured-state snapshot for `preview.py`; rendering does not advance physics. A target-scene render or passing integration check must never be reported as successful learned assembly.
