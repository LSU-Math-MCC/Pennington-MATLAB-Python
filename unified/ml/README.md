# ML

`experiment/` contains the relocated historical `Python_ML_2021/` ML and DOE code.

Training and cross-validation experiments are separate from inference runs under
`runs/`. Future prediction integration belongs in the staged
`python -m unified --input ...` pipeline, but this relocation does not define a
standardized ML prediction API or experiment tracker.

Use:

```bash
python -m unified ml
```

for the current stage guidance.
