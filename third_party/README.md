# third_party: LightGlue

The current project implements A (SIFT + ratio test), B (SIFT + LightGlue), and C (SuperPoint + LightGlue). The optional local LightGlue checkout provides upstream source and visualization examples; its revision is recorded in `../third_party_versions.json`.

| Directory | Source | Role |
| --- | --- | --- |
| LightGlue/ | https://github.com/cvg/LightGlue | Pretrained matching, SuperPoint extraction, and demo visualization utilities |

From the project root, `pip install -e .` installs the pinned LightGlue package and all ABC runtime dependencies automatically. A local third_party checkout is not required for execution. The root .gitignore excludes downloaded external code while retaining this README.

Keep project changes in scripts/ and preserve upstream licenses and citations. scripts/train.py selects matching parameters on the 96 train sequences; scripts/evaluate.py uses the selected parameters on the 20 test sequences.
