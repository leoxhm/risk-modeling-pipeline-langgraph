# Node configuration contract

Node configuration files live in the user's workspace under
`configs/node_configs/`. Each file has the same envelope:

The initial `data-read.yaml` can be copied from
`assets/data_read.template.yaml`; other node files are created from the
engine's conservative defaults when first requested.

```yaml
node_id: eda-analysis
version: 1
parameters:
  month_col: event_month
  psi_base_month: null
  bin_count: 10
  ks_method: quantile
  ks_bucket: 10
  binning_method: quantile
  max_categories: 20
  correlation_method: pearson
  metrics_backend: toad
  missing_rate_threshold: 0.8
  constant_rate_threshold: 0.8
  duplicate_strategy: keep_first
  generate_plots: true
  plot_top_n: 50
  generate_report: true
```

The executor creates a conservative default when a file is missing, then
validates the node ID, version and node-specific parameter constraints. The
confirmation request records every selected file path and SHA-256. `approve`
requires one `--confirm-node` for each selected node, and execution rejects a
changed file or a missing node entry.

In v1:

- `data-read.yaml` controls the optional role names (`id_col_nm`, `dt_col_nm`,
  `label_col_nm`), CSV encoding, Excel sheet, schema inference and read-time
  contract tolerance. A null role value falls back to `data_contract.yaml`;
  a non-null value is an explicit override and is validated before EDA/modeling.

Preparation uses a bootstrap read first. Missing role values are filled only
after the raw table is loaded, using contract values when present and then the
deterministic profiler candidates. OpenCode may revise the same file with an
additional suggestion. If the user edits the YAML, run `prepare` again to
refresh the evidence and confirmation request; `approve` then binds its
SHA-256, and execution reloads the approved YAML before reading again.
- `eda-analysis.yaml` controls EDA binning, PSI baseline, correlation, metric
  backend, quality thresholds, plots and workbook generation.
- `sample-diagnosis.yaml` contains both diagnostic thresholds and the selected
  treatment actions. The first standalone invocation only writes diagnostics
  and waits for confirmation. A confirmed invocation writes an auditable
  `sample_treatment_policy.json`; it does not modify or save treated data.
  Later feature preprocessing/modeling nodes apply the confirmed policy.
- The remaining files are modeling-phase templates and are validated only when
  their nodes are selected.
