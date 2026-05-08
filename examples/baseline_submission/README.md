# Reference Submission

The `submission.json` file in this folder is produced by running the
reference baseline under the `baseline` policy scenario:

```bash
python scripts/make_example_submission.py
```

It is a complete, schema-compliant example you can use as a starting
template for your own submissions.

To validate it:

```bash
python scripts/score_submission.py examples/baseline_submission/submission.json
```

To score it against the synthetic public ground-truth file:

```bash
python scripts/build_ground_truth_public.py
python scripts/score_submission.py examples/baseline_submission/submission.json \
    --ground-truth data/ground_truth_public.json
```
