# results: Measured outputs and figures

Store outputs produced by running code as evidence for the check-in and final report.

## What belongs here

- Machine-readable baseline output, such as baseline_test.json (an example name; not yet generated).
- Per-pair CSV/JSON results: method, sequence ID, category, k, match count, and MMA@1-10px.
- Summary tables reporting MMA@1, 3, and 5px and mean match count for illumination/viewpoint subsets.
- Figures showing MMA@3px versus k=2-6 and MMA versus thresholds of 1-10px.
- Relevant logs and run metadata: command, configuration, split, code version, Python/dependency versions, and device.

## Saving conventions

- Clearly separate pilot and test outputs; avoid overwriting different methods or runs.
- Record whether pixel thresholds use original or resized coordinates, plus zero-match scoring and aggregation rules.
- Distinguish unique image counts from image-pair counts.
- Do not present expected numbers, placeholders, or proposal numbers as newly measured results.
- Commit small result tables, figures, and necessary logs to the team repository. Store large datasets and model weights separately.

## Check-in relevance

Item 2 requires a measured baseline, its producing command, and a saved output file here. For Item 3, fill in the baseline; unfinished methods may remain blank with a named owner and completion date.
