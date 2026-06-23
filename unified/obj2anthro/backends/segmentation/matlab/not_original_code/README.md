# Avatar Testing Scripts

This directory contains test scripts for the Avatar class located in `../original_code/`.

## How It Works

Each script automatically adds `../original_code/` to the MATLAB path, allowing them to access `Avatar.m` and its dependencies.

## Available Scripts

### 1. `inspect_avatar_step1.m` (Recommended for testing)
Runs only mesh cleaning (step 1) - most reliable test.
```matlab
run('inspect_avatar_step1.m')
```

### 2. `run_avatar_demo_skip_repair.m`
Skips mesh repair (steps [1,3]) - useful for bypassing repair errors.
```matlab
run('run_avatar_demo_skip_repair.m')
```

### 3. `run_avatar_demo.m`
Full pipeline with all steps [1,2,3] - may fail on problematic meshes.
```matlab
run('run_avatar_demo.m')
```

### 4. `run_avatar_demo_raw.m`
Attempts to read OBJ file directly and construct minimal Avatar.
```matlab
run('run_avatar_demo_raw.m')
```

## Path Setup

All scripts use this pattern to find Avatar.m:

```matlab
% Add original_code directory to path to access Avatar.m
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, '..', 'original_code'));
```

This means you can run these scripts from anywhere and they'll correctly locate the Avatar class.

## Model File Paths

The scripts reference `../../model_files/man.obj` which assumes:
- Scripts are in `matlab/not_original_code/`
- Model files are in `model_files/` at the repo root

If you move files, update the `modelPath` variable in each script.
