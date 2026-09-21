# EDA analysis node

This node is the second user-facing stage (the UI label is “EDA分析”).
It runs independently from `data-read` and never reuses an in-memory dataframe.

## Execution

After `data-read` has been confirmed, each invocation:

1. reads the current `data-read.yaml` and source file;
2. checks `configs/approvals/data-read.approval.json` and its SHA-256;
3. creates `configs/node_configs/eda-analysis.yaml` when missing;
4. writes `eda_confirmation.json` as an execution marker with `approved` status;
5. reloads the source, applies the configured cleaning policy, calculates EDA
   tables and writes both `data_eda_report.html` and `data_eda_report.xlsx`
   below `reports/`.

EDA uses the current template parameters directly and does not require a
second parameter form. After the report is generated, the node is automatically
approved and the next selected node starts; no EDA summary confirmation is
shown in chat. The HTML report is the recommended
user-facing view: it contains monthly sample charts, variable overview,
chi-square/WOE binning details, PSI heatmaps, the correlation matrix, monthly
discrimination tables and one distribution chart per variable. Excel remains
available for filtering and offline detail export. Both reports end with an
automatic result overview (`结果概述` in the HTML/workbook and
`eda_conclusion.md`).

The public command `sample-diagnosis` is a separate node implemented under
`workflow/nodes/sample_diagnosis/`; it runs only after this EDA gate has
completed. Historical batch runs may still retain their combined compatibility
path, but standalone OpenCode execution must use `eda-analysis` for EDA.
