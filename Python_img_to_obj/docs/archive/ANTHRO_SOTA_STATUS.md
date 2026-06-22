# Anthropometry SOTA Status

Date checked: 2026-06-13

## Official Targets

HBW is the official SHAPY-style body-shape benchmark for anthropometry. The public SHAPY
leaderboard reports the following SHAPY baseline on the HBW test set, in millimetres:

- height: 51
- chest: 65
- waist: 69
- hips: 57
- P2P20K: 21

Source: `https://shapy.is.tue.mpg.de/hbwleaderboard.html`

SSP-3D is a useful sanity benchmark for body shape. Its official dataset notes that GT
SMPL shape labels are pseudo-ground truth from multi-frame shape-consistent optimisation.
PVE-T-SC is scale-corrected, so it is not enough to prove absolute anthropometric accuracy.

Source: `https://github.com/akashsengupta1997/SSP-3D`

CameraHMR is the current single-image HPS reference this repo tracks for SSP-3D shape
accuracy. That is a single-image setting; our same-subject multi-view/SKF experiments are
labelled separately and must not be presented as single-image leaderboard results.

Source: `https://arxiv.org/abs/2411.08128`

## Local Status

`python tools/benchmark/bench_discover.py` currently finds local SSP-3D at
`datasets/SSP-3D/ssp_3d`, but no local HBW, MMTS, or SHAPY checkpoints on this
Windows checkout. Fresh HBW official anthropometry benchmarking is therefore blocked.

Latest local benchmark runner result:

- HBW: `insufficient_ground_truth`
- SSP-3D: `beats_shapy_only_on_ssp3d`

The SSP-3D result is based on existing `runs/` artifacts plus the now-present SSP-3D
dataset. It is labelled sanity evidence only; HBW remains the official anthropometry gate.

## Claim Gate

The repo may claim "SOTA anthropometry" only when all of the following are true:

1. HBW test data is present or an equivalent official-evaluation artifact is supplied.
2. The same measurement extractor is used for every method.
3. `p2p20k_mm < 21`.
4. At least three of height/chest/waist/hips errors are lower than SHAPY's HBW values.
5. No measurement regresses by more than 10 percent unless P2P20K improves by more than 20 percent and the regression is explicitly reported.

Until then, the honest claim is narrower: the repo has a tested anthropometric measurement
stack and labelled SSP-3D/multi-view evidence, but not a verified HBW SOTA result.
