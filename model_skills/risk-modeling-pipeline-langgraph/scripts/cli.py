"""Terminal entrypoint for the self-contained LangGraph Skill trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

from graph.builder import NODE_ORDER, build_graph
from graph.checkpointer import PersistentMemorySaver


def _parser() -> argparse.ArgumentParser:
    """构造 start、resume 和本地 interactive 三种命令行入口。"""
    parser = argparse.ArgumentParser(description="Self-contained LangGraph risk-modeling Skill")
    command = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "interactive"):
        item = command.add_parser(name)
        item.add_argument("--project-root", required=True)
        item.add_argument("--skill-root")
        item.add_argument("--data")
        item.add_argument("--run-id")
    resume = command.add_parser("resume")
    resume.add_argument("--project-root", required=True)
    resume.add_argument("--skill-root")
    resume.add_argument("--data")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--decision", required=True, help='JSON decision, e.g. {"approved":true}')
    return parser


def _print_result(result: dict) -> None:
    """把 LangGraph 普通结果或 interrupt 结果序列化为 OpenCode 可读 JSON。"""
    items = result.get("__interrupt__")
    if items:
        value = items[0].value if hasattr(items[0], "value") else items[0]
        print(json.dumps({"status": "waiting_confirmation", "interrupt": value}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    """解析命令并执行一次工作流或从持久化检查点恢复工作流。"""
    args = _parser().parse_args(argv)
    skill_root = Path(args.skill_root).expanduser().resolve() if args.skill_root else Path(__file__).resolve().parents[1]
    project_root = Path(args.project_root).expanduser().resolve()
    from langgraph.types import Command

    run_id = args.run_id or f"lg-{uuid.uuid4().hex[:12]}"
    checkpoint_path = project_root / "outputs" / ".langgraph" / f"{run_id}.pkl"
    saver = PersistentMemorySaver(checkpoint_path)
    graph = build_graph(project_root=str(project_root), skill_root=str(skill_root), data=args.data, selected_nodes=list(NODE_ORDER), checkpointer=saver)
    config = {"configurable": {"thread_id": run_id}}
    if args.command == "resume":
        try:
            decision = json.loads(args.decision)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--decision 必须是合法 JSON: {exc}") from exc
        result = graph.invoke(Command(resume=decision), config)
        _print_result(result)
        print(f"run_id: {run_id}")
        return 0

    result = graph.invoke({"run_id": run_id, "project_root": str(project_root), "skill_root": str(skill_root), "selected_nodes": list(NODE_ORDER), "status": "running"}, config)
    _print_result(result)
    print(f"run_id: {run_id}")
    if args.command == "interactive":
        while result.get("__interrupt__"):
            raw = input("输入 y 确认，或输入 JSON：").strip()
            decision = {"approved": True} if raw.lower() in {"y", "yes"} else json.loads(raw)
            result = graph.invoke(Command(resume=decision), config)
            _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
