# Unified Pipeline

Run both backends on the standard OBJ test set:

```bash
python -m unified --input Python_Fall2025/model_files/OBJ --backend all --units auto
```

CSV output: `unified/results/<TIMESTAMP>_full_anthro.csv`

Raw backend artifacts: `unified/results/raw/<TIMESTAMP>_full_anthro/` (git ignored)
