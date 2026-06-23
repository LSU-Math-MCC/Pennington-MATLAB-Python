# Relocated

The project formerly stored at `Python_img_to_obj/` was relocated verbatim to
`unified/img2obj/`.

Its original README is now:

`unified/img2obj/README.md`

Old native usage:

`cd Python_img_to_obj && python -m pipeline.run ...`

New native usage:

`cd unified/img2obj && $env:PYTHONPATH="src"; python -m pipeline.run ...`

New unified usage:

`python -m unified img2obj --input ...`

Do not add new implementation files here. Update the relocated project instead.
See `unified/RELOCATION_MAP.md` for the complete mapping.
