# Node configuration contract

Node configuration files live in the user's workspace under
`configs/node_configs/`. Each file has the same envelope. The shared
`configs/data_contract.yaml` is not a node config: it defines the data source
and field roles consumed by every node. User-confirmation manifests are kept
in the sibling `configs/approvals/` directory.

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
- `sample-diagnosis.yaml` is retained as a visible default-parameter template
  for compatibility. The standalone node loads the Skill template defaults,
  writes diagnostics and a short summary immediately, and does not wait for
  confirmation or write a treatment approval. Treatment actions are confirmed
  later in `model-config.yaml`.
- `feature-processing.yaml` may include an optional `parameters.llm` block
  (`enabled`, `base_url`, `model`, `api_key`/`api_key_env`, TLS and timeout
  settings). The advisor receives EDA aggregates only and creates a reviewable
  YAML draft; it never mutates the original config implicitly.
- `model-config.yaml` contains the concise modeling decisions and the optional
  `tuning.llm` settings. The LLM service may be configured directly with
  `tuning.llm.api_key`, `tuning.llm.base_url` and `tuning.llm.model`; an empty
  `api_key` falls back to `api_key_env` (default `LLM_TUNING_API_KEY`). A key
  written in YAML is copied into confirmation snapshots, so keep such a
  workspace private and out of version control. TLS verification is enabled by
  default; use `ca_bundle` for an internal CA, and only use
  `verify_ssl: false` for isolated internal tests. If Python's OpenSSL bundle
  rejects a certificate that the host system trusts, the tuner makes one
  secure fallback request through the system `curl` binary before recording a
  provider error.
- The remaining files are modeling-phase templates and are validated only when
  their nodes are selected.
