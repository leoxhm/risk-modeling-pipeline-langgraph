# AI assistance and staged confirmation gates

The assistant orchestrates eight user-facing stages. In v1 data-read, EDA and
sample diagnosis are executable; the remaining stages are reserved for the
modeling phase.
It must not expose the
engine's internal profiler, preflight, cleaning, EDA, tuning, or report nodes as
separate decisions.

## Gate sequence

1. After **data-read**, validate `configs/node_configs/data-read.yaml` and
   confirm the read options, contract roles and target/date parsing behavior.
2. After the confirmed data-read, execute **EDA analysis** through
   `workflow.nodes.eda_analysis.run`. Validate `configs/node_configs/eda-analysis.yaml`,
   summarize field quality, IV/KS/PSI, monthly discrimination, correlations and
   report options, then wait for EDA config confirmation. EDA must not split
   Train/Test/OOT, apply class weights or train a model.
3. If selected, execute **sample diagnosis** through
   `workflow.nodes.sample_diagnosis.run` after EDA. Validate
   `configs/node_configs/sample-diagnosis.yaml`,
   summarize class imbalance, missing months, monthly bad-rate shifts, duplicate
   IDs, missing/invalid labels and incomplete latest months, then wait for user
   confirmation. The confirmed call records the policy artifact; it does not
   modify or save treated data, split data, train a model or apply class weights
   (class_weight is recorded as a later training hint).
4. After **feature preprocessing and selection**, summarize every retained,
   excluded and review-required feature; wait for confirmation. The preview is
   marked `sample_treatment_applied=false`; if the user changes a sample action,
   rerun the relevant node confirmation before continuing.
5. After **model configuration**, summarize OOT/Test, class-imbalance policy,
   LightGBM settings and Optuna budget; wait for confirmation.
6. After **training and tuning**, summarize baseline/tuned/Test/OOT metrics and
   ask whether to accept the candidate or revise configuration.

The confirmation manifest remains the final hash-bound execution gate. If the
host wants to run the stages as separate jobs, each job must use a new request
and approval hash after any contract, sample policy or model configuration
change.

## What AI may do

- Summarize deterministic profiling and data-readiness evidence.
- Propose ID, date, target, feature, and exclusion roles with reasons and uncertainty.
- Organize a baseline-versus-tuning experiment plan within the configured budget.
- Red-team completed model metrics for overfitting, weak OOT support, unstable training, and concentrated feature importance.

AI output is advisory. Python remains the source of truth for counts, hashes, splits, metrics, tuning, and report tables.

## What AI must not do

- Decide target meaning, label polarity, leakage fields, or observation windows without user confirmation.
- Treat a field-name guess as a confirmed schema decision.
- Acknowledge a blocker or write an approval manifest on the user's behalf.
- Modify data or configurations after confirmation and reuse the old approval.
- Tune, select, or approve a candidate using OOT performance.
- Claim production readiness from an automated report.

## Gate artifacts

The legacy batch `prepare` command creates:

- `column_profile.csv`: deterministic column statistics.
- `sample_diagnostics.json`: class balance, target completeness, duplicate IDs, row missingness, month continuity, monthly target rates, findings, and the proposed treatment actions.
- `ai_preflight.json`: data-readiness findings and evidence.
- `ai_schema_proposal.json`: field-role and leakage-review suggestions.
- `ai_experiment_plan.json`: proposed baseline/tuning sequence and guardrails.
- `feature_proposal.json`, `feature_preprocessing.csv`, and
  `feature_selection.csv`: Train-only type, encoding, quality, IV,
  correlation and stability evidence for the feature confirmation gate.
- `model_config_proposal.yaml`: deterministic snapshot of the model parameters
  waiting for the model-configuration confirmation gate.
- `confirmation_request.json`: exact roles, blocker codes, file paths, and SHA-256 hashes awaiting confirmation.
- `review_artifacts.node_configs`: the selected node YAML files. The approval
  command requires one `--confirm-node <node-id>` for every selected node.

`approve` creates `approval_manifest.json` only after explicit user confirmation. Every blocker code must be individually acknowledged. Before execution, the CLI rejects a manifest when its mode, path, or file hash does not match the requested run.

## Required confirmation summary

Before asking the user to approve, state:

1. ID fields and their uniqueness meaning.
2. Observation date and available time range. Do not ask for OOT months yet.
3. Target field plus good/bad label mapping.
4. Included features and suspected leakage fields only when present in the
   confirmed read/EDA evidence.
5. EDA-specific quality, binning, PSI baseline and report options. Do not ask
   for class imbalance, Test ratio or training parameters during EDA.
6. When sample diagnosis is selected, ask for the diagnostic thresholds and
   treatment actions in `sample-diagnosis.yaml` (duplicate IDs, missing/invalid
   labels, all-null features, high-missing rows, incomplete latest month and
   the class-imbalance training hint). Do not apply those actions before the
   user confirms them.
7. For later modeling nodes only, ask for OOT/Test, class-imbalance policy,
   Optuna budget and training guardrails.
8. Every blocker with its evidence, consequence, and safer alternative.

If the user changes any item, update the appropriate YAML, rerun `prepare`, and present the refreshed request.
