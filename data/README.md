# Data folder

Drop the Sensei-provided dataset here (it is git-ignored):

```
data/
├── planogram.json     # planogram (camera -> areas with coordinates)
└── images/
    ├── 1751052241.jpg # one file per capture timestamp (Unix timestamp name)
    ├── 1751052301.jpg
    └── ...
```

Notes:
- Image filenames are Unix timestamps shared across cameras (same capture
  moment), matching the annotation keying `camera -> image -> area`.
- If the dataset arrives as per-camera folders instead, flatten it into
  `data/images/` or point `SENSEI_IMAGES_DIR` at the right place.
- Train, test and validation sets: keep them in separate folders
  (e.g. `data/train/images`, `data/val/images`) and run the pipeline with
  `SENSEI_IMAGES_DIR` / `SENSEI_PLANOGRAM` overrides.
