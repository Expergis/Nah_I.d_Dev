# Audit set (manual labels)

Our calibration loop: hand-label ~50-150 areas from the UNANNOTATED train/test
images (via contact sheets from `scripts/contact_sheet.py`), then compare the
pipeline against these labels with `scripts/audit_report.py`.

This is our only safe tuning surface — we do NOT tune against the hidden
validation set (use official runs sparingly, for confirmation).

## Label file: `audit/labels.json`

```json
{
  "6215": {
    "1751052241.jpg": {
      "6215_2": {"compliant": true,  "stock": "full", "occluded": false},
      "6215_3": {"compliant": false, "stock": "empty", "occluded": true}
    }
  }
}
```

Rules of thumb (from the brief):
- `stock` is the team's best read of empty / half / full coverage.
- `compliant` = the product shown is the one the planogram expects.
  An empty area is a stock issue, not automatically a mismatch —
  (confirm with the Sensei mentors how they annotated empty areas).
- `occluded` = a hand/person blocks a significant part of the area.
