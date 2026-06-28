# Slice Backend

This directory owns the slice-based OBJ anthropometry implementation.

Run it through the staged wrapper from the repository root:

```bash
python -m unified obj2anthro --input data/obj --method slice --units auto
```

The backend code lives in `slice.py`. Core OBJ examples live in `data/obj/`.
