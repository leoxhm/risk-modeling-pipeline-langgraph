# data-read 节点回复模板

`data-read` 的聊天回复只展示一张固定的中文表格。表格由工作流代码根据
`data-read.yaml`、`data_contract.yaml` 和本次实际读取结果生成，OpenCode 不应
自行推断或改写数值。

```markdown
## data-read 结果

| 项目 | 值 |
|---|---|
| 数据文件 | `{文件名}` |
| 文件格式 | `{CSV/Parquet/Excel}` |
| 样本量 | `{行数} 行 × {列数} 列` |
| ID 字段 | `{字段}`（唯一率 `{唯一率}`） |
| 日期字段 | `{字段}`（解析率 `{解析率}`） |
| 目标字段（Y） | `{字段}` |
| 好坏标签映射 | 好 = `{good_label}`；坏 = `{bad_label}` |
| 特征字段 | `{特征预览}`（共 `{数量}` 个） |
| 排除字段 | `{排除字段}` |
| 读取参数 | `encoding`、`sheet_name`、`infer_schema_length` |
| 校验容忍开关 | `allow_id_duplicates`、`allow_target_issues` |

以上字段角色和读取参数来自已解析的 YAML/数据契约，统计值来自本次实际读取。
请确认表格内容后再继续；如需调整，请直接用文字说明要修改的字段或读取策略，助手会更新内部配置并重新执行本节点。
```

机器可读结果位于 `outputs/data-read/data_read_summary.json`，同内容的 Markdown
副本位于 `outputs/data-read/data_read_summary.md`。配置文件仅用于内部复现、哈希校验和审计，
不在聊天窗口展示。
