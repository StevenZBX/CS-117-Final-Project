# third_party: Official external code

Store the official repositories used by this project here, separately from the team's scripts/.

| Directory | Source | Project role |
| --- | --- | --- |
| LightGlue/ | https://github.com/cvg/LightGlue | Shared SIFT feature support for A/B and pretrained matching/feature models for B/C |
| glue-factory/ | https://github.com/cvg/glue-factory | Training and validation framework for optional method D |

Both repositories include their official README and license files. Team installation instructions are in the root README.md. Sources and cloned commits are recorded in ../third_party_versions.json.

## Usage conventions

- First run a minimal A/B experiment with LightGlue, then extend it to C.
- Use separate evaluation and training environments; refer to requirements.txt and requirements-train.txt respectively.
- Prefer writing project code in scripts/. If upstream source changes are necessary, document the files and reasons and preserve reproducible patches or a fork.
- Verify upstream evaluation metrics before use; they may differ from this project's MMA.
- Preserve upstream licenses and citation information.

## Git and reproducibility

The root .gitignore excludes the two external repositories but retains this README in the team repository. After cloning the team repository, teammates must restore both external repositories using the URL, path, and commit in third_party_versions.json.

## Check-in relevance

Documents external sources and distinguishes borrowed code from team-written code. Item 5 requires the team's GitHub URL and evidence of each member's actual contributions. Cloning an official repository alone does not constitute a completed experiment.
