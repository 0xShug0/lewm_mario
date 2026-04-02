# LeWM Mario

Goal-conditioned Super Mario Bros. experiments built around a LeWM-style world model.

This repo keeps the publishable core only:

- export exact FM2 playback from FCEUX into `.npz` episodes
- convert per-frame episodes into blocked LeWM trajectories
- train a reward-free world model from offline traces
- evaluate offline losses
- run local goal-conditioned live demos
- record demo videos with start/goal reference panels

What this repo is **not**:

- not a full-game autonomous Mario agent
- not a reinforcement learning project
- not a full emulator bundle

The main demo replays an existing trace up to a chosen point, then lets the model take over for a short local segment toward a target image.

## External dependencies

This repo does **not** include:

- FCEUX binaries
- BizHawk
- the Super Mario Bros. ROM
- large local datasets, traces, checkpoints, or rendered demos

You need to provide:

- an FCEUX install directory containing `fceux64.exe`
- a compatible Super Mario Bros. ROM
- `.fm2` trace files under a local `traces/` directory

By default, the scripts expect:

- FCEUX under `./fceux`
- the ROM at `./fceux/SMB.nes`
- traces under `./traces`

Those paths can be overridden with command-line flags.

## Repo layout

Core Python package:

- `mario_lewm/__init__.py`
- `mario_lewm/dataset.py`
- `mario_lewm/fm2.py`
- `mario_lewm/model.py`
- `mario_lewm/planning.py`

Supported scripts:

- `export_fceux_dataset.py`
- `build_lewm_mario_dataset.py`
- `split_mario_dataset.py`
- `precompute_mario_dataset.py`
- `train_mario.py`
- `test_mario.py`
- `show_goal_frames.py`
- `demo_mario_goal_live_fixed.py`
- `demo_mario_goal_live_record.py`
- `fceux_export_trace.lua`
- `fceux_live_bridge_fixed.lua`
- `fceux_live_bridge_record.lua`

## Demo

Sample rendered videos are available under `demo/`, for example:

- `demo/demo1.mp4`
- `demo/demo5.mp4`
- `demo/demo9.mp4`

These were produced with the recorded local-goal demo pipeline and are meant as examples, not as required repo assets.

### Inline examples

<video src="demo/demo1.mp4" controls width="720"></video>

<video src="demo/demo5.mp4" controls width="720"></video>

<video src="demo/demo9.mp4" controls width="720"></video>

If your Markdown renderer does not show embedded video, use the direct file links above.

## Install

```bash
python -m pip install -r requirements.txt
```

You also need `ffmpeg` on `PATH` for recorded demos.

## Workflow

### 1. Export raw per-frame episodes from FCEUX

```bash
python export_fceux_dataset.py --trace-root traces --output-dir mario_dataset_raw_lewm --max-frames 2000 --capture-initial-frame
```

This produces `.npz` files with:

- RGB frames from exact FM2 playback
- per-frame controller rows
- metadata copied from the FM2 header

### 2. Build blocked LeWM trajectories

```bash
python build_lewm_mario_dataset.py --dataset-root mario_dataset_raw_lewm --output-dir mario_dataset_lewm --frame-skip 5
```

The blocked dataset uses:

- one blocked action = 5 raw emulator frames
- one blocked frame sequence aligned to those action blocks

### 3. Split into train/test episode sets

This keeps one or more whole traces out of training so demos and evaluation can use held-out data.

```bash
python split_mario_dataset.py --dataset-root mario_dataset_lewm --train-dir mario_dataset_lewm_train --test-dir mario_dataset_lewm_test --test-name 141presses2
```

### 4. Optional preprocessing for faster training

```bash
python precompute_mario_dataset.py --dataset-root mario_dataset_lewm_train --output-dir mario_dataset_lewm_train_224 --image-size 224
```

### 5. Train

```bash
python train_mario.py --dataset-root mario_dataset_lewm_train --precomputed-root mario_dataset_lewm_train_224 --output-dir mario_runs/run_lewm_mario --epochs 100 --batch-size 128 --num-workers 6 --save-every 20 --npz-load-mode lazy --max-cached-episodes 4 --batching episode --log-every-steps 20 --compile
```

Resume:

```bash
python train_mario.py --dataset-root mario_dataset_lewm_train --precomputed-root mario_dataset_lewm_train_224 --output-dir mario_runs/run_lewm_mario --epochs 100 --batch-size 128 --num-workers 6 --save-every 20 --npz-load-mode lazy --max-cached-episodes 4 --batching episode --log-every-steps 20 --compile --resume-latest
```

### 6. Offline evaluation on held-out test episodes

```bash
python test_mario.py --checkpoint mario_runs/run_lewm_mario/best.pt --dataset-root mario_dataset_lewm_test --mode offline --batch-size 32 --num-workers 0
```

### 7. Inspect a held-out start/goal pair

```bash
python show_goal_frames.py --dataset-root mario_dataset_lewm_test 141presses2 60 65
```

### 8. Live local-goal demo on held-out data

```bash
python demo_mario_goal_live_fixed.py --checkpoint mario_runs/run_lewm_mario/best.pt --dataset-root mario_dataset_lewm_test --episode-name 141presses2 --start-index 60 --goal-index 65 --horizon 5 --replan-every 1 --max-steps 10 --visual-debug
```

### 9. Record a held-out demo video

```bash
python demo_mario_goal_live_record.py --checkpoint mario_runs/run_lewm_mario/best.pt --dataset-root mario_dataset_lewm_test --episode-name 141presses2 --start-index 60 --goal-index 65 --horizon 5 --replan-every 1 --max-steps 10 --output-dir demo_141presses2
```

Batch several recordings:

```bash
python demo_mario_goal_live_record.py --checkpoint mario_runs/run_lewm_mario/best.pt --dataset-root mario_dataset_lewm_test --episode-name 141presses2 --start-index 60 --goal-index 65 --horizon 5 --replan-every 1 --max-steps 10 --output-dir demo_141presses2_batch --num-runs 10
```

## Notes on evaluation

- The demos are **local goal-conditioned control**, not full-level gameplay.
- The video starts with trace replay, then switches to model control.
- The planner does **not** use rewards or RL.
- Planning uses a target image and latent-distance cost only.

