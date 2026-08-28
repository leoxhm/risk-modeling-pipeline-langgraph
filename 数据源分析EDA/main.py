#!/usr/bin/env python3
"""风控 EDA 流水线入口。"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from eda.analytics import run_all
from eda.config import load_config
from eda.data import load_and_clean
from eda.excel_writer import write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="风控变量 EDA 分析与 Excel 报告导出")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="配置文件路径（默认: config.yaml）",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="输出 DEBUG 日志",
    )
    return parser.parse_args()

def is_jupyter():
    """检测是否在 Jupyter 环境中运行"""
    try:
        # 检查是否在 IPython/Jupyter 环境中
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    
    # 检查命令行参数中是否包含 Jupyter 相关标志
    if 'ipykernel' in sys.modules:
        return True
    
    return False

def main() -> int:
    if not is_jupyter():
        args = parse_args()
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        config_path = Path(args.config).resolve()
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        config_path = Path('./config.yaml').resolve()
        
    logger = logging.getLogger(__name__)
    if not config_path.exists():
        logger.error("配置文件不存在: %s", config_path)
        return 1

    cfg = load_config(config_path)
    base_dir = cfg.config_dir

    logger.info("加载数据: %s", cfg.data.input_path)
    df, yearmonth_col, yearmonth_exclude = load_and_clean(cfg.data, base_dir=base_dir)
    logger.info("数据 shape: %s", df.shape)

    # 在临时目录生成图片
    with tempfile.TemporaryDirectory() as temp_dir:
        plots_dir = Path(temp_dir) / "plots"
        tables, _ = run_all(
            df, cfg,
            yearmonth_col=yearmonth_col,
            yearmonth_exclude=yearmonth_exclude,
            plots_dir=plots_dir,
        )
        # for name, table in tables.items():
        #     logger.info("  %s: %s 行 × %s 列", name, table.shape[0], table.shape[1])

        excel_path = cfg.resolve_path(cfg.output.excel_path)
        out = write_report(tables, excel_path, cfg, df, yearmonth_exclude, plots_dir)
        logger.info("Excel 报告已写入: %s", out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
