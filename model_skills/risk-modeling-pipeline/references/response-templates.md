# 节点回复模板索引

节点回复模板按用户节点拆分，避免单个文件过长：

- `data-read-response-template.md`：数据文件与字段角色确认表
- `eda-response-template.md`：EDA 报告生成后的简短状态提示（不复制明细）
- `sample-diagnosis-response-template.md`：样本诊断与建模风险提示表
- `feature-processing-response-template.md`：特征保留/剔除、原因、阈值与时间切分说明表
- `model-config-response-template.md`：样本时间划分、调参方式和模型选择表

后续节点可继续按同样规则增加独立模板，例如 `model-review-response-template.md`。

通用要求：中文输出、先结论后证据；EDA 只提示报告路径，样本诊断使用 1 张带阈值解释的风险表，
特征处理使用 1 张结果表（剔除字段和原因、阈值及指标范围），模型配置使用 1 张决策表
（时间划分及原因、调参方式和模型选择），其余节点通常使用 1 张小表；
建议必须引用实际产物，告警不阻断流程。
