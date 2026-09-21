# 样本诊断节点回复模板

样本诊断使用 Skill 默认参数，直接输出一张风险提示表，不等待 `sample-diagnosis.yaml` 配置确认，但结果反馈后必须等待用户确认；不修改样本。

```markdown
## 样本诊断与建模风险提示

已检查 {row_count} 条样本；本节点使用 Skill 默认阈值，未修改原始数据。

| 检查项 | 当前结果 | 判定 | 阈值解释 | 建模建议 |
|---|---|---|---|---|
| 标签与主键质量 | {quality_result} | {quality_status} | 重复主键和缺失标签应为 0 | {quality_advice} |
| 整体坏样本率 | {bad_rate} | {bad_rate_status} | 通用参考区间 {bad_rate_min}–{bad_rate_max} | {bad_rate_advice} |
| IV 达标特征占比 | {iv_pass_ratio} | {iv_status} | IV ≥ {iv_threshold}；达标占比 ≥ {quality_ratio_threshold} | {iv_advice} |
| PSI 有效特征占比 | {psi_effective_ratio} | {psi_status} | 最大 PSI < {psi_threshold}；达标占比 ≥ {quality_ratio_threshold} | {psi_advice} |
| PSI 稳定特征占比 | {psi_stable_ratio} | {psi_stable_status} | 最大 PSI < {psi_stable_threshold} 为稳定 | {psi_stable_advice} |
| 缺失率达标特征占比 | {missing_ratio} | {missing_status} | 字段缺失率 ≤ 80%；达标占比 ≥ {quality_ratio_threshold} | {missing_advice} |
| 类别不平衡 | {minority_rate} | {imbalance_status} | 少数类占比 < {imbalance_threshold} 提示不平衡 | {imbalance_advice} |
| 时间完整性与月度波动 | {time_result} | {time_status} | 月份断层、坏账率变化 ≥ {monthly_change_threshold} 或最新月不足 {latest_ratio} 会提示 | {time_advice} |

判定仅为建模前提示，不是自动阻断；详细字段明细请查看 EDA 报告。

风险提示表：{modeling_risk_summary_path}
诊断明细：{diagnostics_path}
```

要求：告警使用“需关注”，不能写成“阻断”；不展示完整诊断 JSON，不提出新的参数确认问题。
