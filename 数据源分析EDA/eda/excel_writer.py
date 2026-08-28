"""多 Sheet Excel 导出与样式/条件格式。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable
import logging
import pandas as pd
from openpyxl import load_workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.drawing.image import Image
from openpyxl.formatting.rule import ColorScaleRule, FormulaRule, DataBarRule, IconSetRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from eda.analytics import (
    SHEET_BINNING,
    SHEET_CORR,
    SHEET_KS,
    SHEET_MONTHLY,
    SHEET_OVERVIEW,
    SHEET_PSI,
    SHEET_PLOTS,
    SHEET_SAMPLE,
)
from eda.config import ExcelConfig

logger = logging.getLogger(__name__)


HEADER_FILL = PatternFill("solid", fgColor="0070C0")
HEADER_FONT = Font(name="微软雅黑", size=12, bold=True, color="FFFFFF")
BODY_FONT = Font(name="微软雅黑", size=10)
THIN_BORDER = Border(
    left=Side(style="thin", color="D9D9D9"),
    right=Side(style="thin", color="D9D9D9"),
    top=Side(style="thin", color="D9D9D9"),
    bottom=Side(style="thin", color="D9D9D9"),
)
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center")

PERCENT_COLS = {
    "缺失率",
    "完整率",
    "坏账率",
    "好账率",
    "好样本率",
    "样本占比",
    "坏样本占比",
    "好样本占比",
    "累计坏样本占比",
    "反向累计坏样本占比",
    "累计好样本占比",
    "反向累计好样本占比",
    "累计样本占比",
    "反向累计样本占比",
}
PERCENT_SUFFIX = ("率", "占比", "比")
DECIMAL4_COLS = {"IV值", "KS值", "平均PSI", "最大PSI", "PSI值", "AUC值", "WOE值", "分箱IV值", "累计IV值", "累计KS值"}
INTEGER_COLS = {
    "唯一值",
    "好样本",
    "坏样本",
    "总样本",
    "好样本数",
    "坏样本数",
    "总样本数",
    "样本总量",
}
DECIMAL2_COLS = {
    "平均值",
    "最小值",
    "最大值",
    "标准差",
    "1%分位",
    "10%分位",
    "50%分位",
    "75%分位",
    "90%分位",
    "99%分位",
    "Lift值",
    "累计Lift值",
    "Lift前10%",
    "好坏比",
}
YEARMONTH_RE = re.compile(r"^\d{6}$")


def _excel_str_width(s: str) -> float:
    """计算字符串在Excel中的显示宽度（中文按2倍字符宽度计算）。"""
    if not s:
        return 0
    width = 0
    for char in str(s):
        if ord(char) > 255:  # 中文字符
            width += 2
        else:
            width += 1
    return width


def _number_format(col_name: str) -> str | None:
    if col_name in PERCENT_COLS or any(col_name.endswith(s) for s in PERCENT_SUFFIX):
        if col_name in DECIMAL4_COLS:
            return "0.0000"
        return "0.00%"
    if col_name in DECIMAL4_COLS:
        return "0.0000"
    if col_name in INTEGER_COLS:
        return "#,##0"
    if col_name in DECIMAL2_COLS:
        return "#,##0.00"
    if col_name == "类型":
        return "@"
    if col_name in ("变量名", "分箱区间", "稳定性评估", "年月"):
        return "@"
    return "#,##0.00"


def _col_range(ws: Worksheet, col_idx: int, start_row: int = 2) -> str:
    letter = get_column_letter(col_idx)
    return f"${letter}${start_row}:${letter}${ws.max_row}"


def _find_col(ws: Worksheet, name: str) -> int | None:
    for idx, cell in enumerate(ws[1], start=1):
        if cell.value == name:
            return idx
    return None


def _apply_color_scale(ws: Worksheet, col_name: str, colors) -> None:
    col_idx = _find_col(ws, col_name)
    if col_idx is None or ws.max_row < 2:
        return
    ref = _col_range(ws, col_idx)
    if len(colors) < 3:
        rule = ColorScaleRule(
            start_type="min",
            start_color=colors[0],
            end_type="max",
            end_color=colors[1],
        )
    else:
        rule = ColorScaleRule(
            start_type="min",
            start_color=colors[0],
            mid_type="percentile",
            mid_value=50,
            mid_color=colors[1],
            end_type="max",
            end_color=colors[2],
        )
    ws.conditional_formatting.add(ref, rule)

def _apply_data_bar(ws: Worksheet, col_name: str, color='638EC6') -> None:
    col_idx = _find_col(ws, col_name)
    if col_idx is None or ws.max_row < 2:
        return
    ref = _col_range(ws, col_idx)
    rule = DataBarRule(
        start_type="min",
        end_type="max",
        color=color,
        showValue=True
    )
    ws.conditional_formatting.add(ref, rule)

def _apply_icon_set(ws: Worksheet, col_name: str, icon_style='3Arrows') -> None:
    col_idx = _find_col(ws, col_name)
    if col_idx is None or ws.max_row < 2:
        return
    ref = _col_range(ws, col_idx)
    rule = IconSetRule(
        icon_style=icon_style,
        type='percent',
        values=[0, 20, 60],
        showValue=True
    )
    ws.conditional_formatting.add(ref, rule)


def _apply_formula_fill(
    ws: Worksheet, col_name: str, formula: str, fill_color: str, font_color: str = "9C0006"
) -> None:
    col_idx = _find_col(ws, col_name)
    if col_idx is None or ws.max_row < 2:
        return
    ref = _col_range(ws, col_idx)
    fill = PatternFill("solid", fgColor=fill_color)
    font = Font(color=font_color)
    rule = FormulaRule(formula=[formula], fill=fill, font=font)
    ws.conditional_formatting.add(ref, rule)


def _apply_global_style(ws: Worksheet, df: pd.DataFrame, excel_cfg: ExcelConfig) -> None:
    # 找到年月列索引
    yearmonth_col_idx = None
    for col_idx in range(1, ws.max_column + 1):
        if ws.cell(1, col_idx).value == "年月":
            yearmonth_col_idx = col_idx
            break

    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column):
        for cell in row:
            cell.border = THIN_BORDER
            if cell.row == 1:
                cell.fill = HEADER_FILL
                cell.font = HEADER_FONT
                cell.alignment = CENTER
            else:
                # 年月列强制转换为文本
                if cell.column == yearmonth_col_idx:
                    cell.value = str(cell.value) if cell.value is not None else ""
                    cell.number_format = "@"
                    cell.alignment = CENTER
                    cell.font = BODY_FONT
                    continue

                cell.font = BODY_FONT
                col_name = ws.cell(1, cell.column).value
                if col_name in ("变量名", "分箱区间"):
                    cell.alignment = LEFT
                else:
                    cell.alignment = CENTER
                fmt = _number_format(str(col_name) if col_name else "")
                if fmt and isinstance(cell.value, (int, float)):
                    cell.number_format = fmt

    # 设置行高：第一行30，其他行20
    ws.row_dimensions[1].height = 30
    for row_idx in range(2, ws.max_row + 1):
        ws.row_dimensions[row_idx].height = 20

    # 设置列宽：第一列22，其他列根据header字符数计算
    ws.column_dimensions["A"].width = 22
    for col_idx in range(2, ws.max_column + 1):
        col_letter = get_column_letter(col_idx)
        header = ws.cell(1, col_idx).value
        header_str = str(header or "")
        width = _excel_str_width(header_str)
        # 基础宽度11，小于等于4字符保持11
        if width <= 8:
            ws.column_dimensions[col_letter].width = 11
        else:
            ws.column_dimensions[col_letter].width = width + 3

    if excel_cfg.freeze_panes and ws.max_row > 1:
        ws.freeze_panes = "A2"
    if excel_cfg.autofilter and ws.max_row > 1:
        ws.auto_filter.ref = ws.dimensions


def _style_overview(ws: Worksheet, excel_cfg: ExcelConfig) -> None:
    _apply_icon_set(ws, "唯一值", icon_style='3Arrows')
    _apply_data_bar(ws, "IV值", color="638EC6")
    _apply_data_bar(ws, "KS值", color="63BE7B")
    col = _find_col(ws, "缺失率")
    if col:
        ref = _col_range(ws, col)
        ws.conditional_formatting.add(
            ref,
            FormulaRule(
                formula=[f"{get_column_letter(col)}2>0.1"],
                fill=PatternFill("solid", fgColor="FFC7CE"),
            ),
        )
    col = _find_col(ws, "完整率")
    if col:
        letter = get_column_letter(col)
        ws.conditional_formatting.add(
            _col_range(ws, col),
            FormulaRule(
                formula=[f"{letter}2<0.9"],
                fill=PatternFill("solid", fgColor="FFEB9C"),
            ),
        )


def _add_var_separator_border(ws: Worksheet, var_col_name: str = "变量名") -> None:
    """为同一变量的最后一行添加蓝色下框线。"""
    var_col = _find_col(ws, var_col_name)
    if var_col is None or ws.max_row < 2:
        return

    thick_border = Border(bottom=Side(style="medium", color="0070C0"))
    prev_var = None
    for row_idx in range(2, ws.max_row + 1):
        curr_var = ws.cell(row_idx, var_col).value
        if prev_var is not None and curr_var != prev_var:
            # 变量发生变化，前一行是上一个变量的最后一行
            for col_idx in range(1, ws.max_column + 1):
                cell = ws.cell(row_idx - 1, col_idx)
                cell.border = Border(
                    left=cell.border.left,
                    right=cell.border.right,
                    top=cell.border.top,
                    bottom=Side(style="medium", color="0070C0"),
                )
        prev_var = curr_var


def _style_binning(ws: Worksheet, excel_cfg: ExcelConfig) -> None:
    _add_var_separator_border(ws, "变量名")
    # 设置分箱区间列宽为17
    col = _find_col(ws, "分箱区间")
    if col:
        ws.column_dimensions[get_column_letter(col)].width = 17
    _apply_color_scale(ws, "坏账率", ("FFFFFF", "F8696B"))
    _apply_data_bar(ws, "累计IV值", color="638EC6")
    _apply_data_bar(ws, "累计KS值", color="63BE7B")
    # _apply_color_scale(ws, "累计IV值", ("63BE7B", "FFEB84", "F8696B"))
    # _apply_color_scale(ws, "累计KS值", ("63BE7B", "FFEB84", "F8696B"))


def _style_psi(ws: Worksheet, excel_cfg: ExcelConfig) -> None:
    # 对所有数值列（除变量名、稳定性评估列外）应用ColorScaleRule
    for col_idx in range(1, ws.max_column + 1):
        name = str(ws.cell(1, col_idx).value or "")
        if name in ("变量名", "稳定性评估"):
            continue
        ref = _col_range(ws, col_idx)
        rule = ColorScaleRule(
            start_type="num",
            start_value=-1,
            start_color="63BE7B",
            mid_type="num",
            mid_value=0,
            mid_color="FFFFFF",
            end_type="num",
            end_value=1,
            end_color="F8696B",
        )
        ws.conditional_formatting.add(ref, rule)

    col = _find_col(ws, "稳定性评估")
    if col:
        letter = get_column_letter(col)
        # 包含"!!!"鲜红色
        ws.conditional_formatting.add(
            _col_range(ws, col),
            FormulaRule(
                formula=[f'ISNUMBER(SEARCH("! ! !",{letter}2))'],
                font=Font(color="FF0000", bold=True),
            ),
        )
        # 包含"!!"暗红色（排除已匹配"!!!"的）
        ws.conditional_formatting.add(
            _col_range(ws, col),
            FormulaRule(
                formula=[f'AND(ISNUMBER(SEARCH("! !",{letter}2)),NOT(ISNUMBER(SEARCH("! ! !",{letter}2))))'],
                font=Font(color="9C0006", bold=True),
            ),
        )
        # 包含"-"绿色
        ws.conditional_formatting.add(
            _col_range(ws, col),
            FormulaRule(
                formula=[f'ISNUMBER(SEARCH("-",{letter}2))'],
                font=Font(color="006100", bold=True),
            ),
        )


def _apply_psi_thresholds(ws: Worksheet, col_idx: int) -> None:
    if ws.max_row < 2:
        return
    ref = _col_range(ws, col_idx)
    rule = ColorScaleRule(
            start_type="num",
            start_value=-1,
            start_color="63BE7B",
            mid_type="num",
            mid_value=0,
            mid_color="FFFFFF",
            end_type="num",
            end_value=1,
            end_color="F8696B",
        )
    ws.conditional_formatting.add(ref, rule)


def _style_corr(ws: Worksheet, excel_cfg: ExcelConfig) -> None:
    ws.sheet_view.zoomScale = 80
    if ws.max_column < 3 or ws.max_row < 2:
        return

    # 设置列宽
    for col_idx in range(1, ws.max_column + 1):
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = excel_cfg.corr_col_width

    # 设置行高
    for row_idx in range(1, ws.max_row + 1):
        ws.row_dimensions[row_idx].height = excel_cfg.corr_row_height

    # 设置单元格样式
    CORR_BODY_FONT = Font(name="微软雅黑", size=12)
    LEFT_WRAP = Alignment(horizontal="left", vertical="center", wrap_text=True)
    CENTER_WRAP = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row_idx in range(1, ws.max_row + 1):
        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row_idx, col_idx)
            cell.border = THIN_BORDER

            if row_idx == 1 and col_idx == 1:
                # 首行第一个单元格居中
                cell.fill = HEADER_FILL
                cell.font = Font(name="微软雅黑", size=12, bold=True, color="FFFFFF")
                cell.alignment = CENTER_WRAP
            elif row_idx == 1:
                # 首行左对齐，自动换行
                cell.fill = HEADER_FILL
                cell.font = Font(name="微软雅黑", size=12, bold=True, color="FFFFFF")
                cell.alignment = LEFT_WRAP
            elif col_idx == 1:
                # 首列粗体左对齐，自动换行
                cell.fill = HEADER_FILL
                cell.font = Font(name="微软雅黑", size=12, bold=True, color="FFFFFF")
                cell.alignment = LEFT_WRAP
            else:
                # 其余单元格14号字体
                cell.font = CORR_BODY_FONT
                cell.alignment = CENTER

    start = get_column_letter(2)
    end = get_column_letter(ws.max_column)
    ref = f"${start}$2:${end}${ws.max_row}"
    rule = ColorScaleRule(
        start_type="num",
        start_value=-1,
        start_color="63BE7B",
        mid_type="num",
        mid_value=0,
        mid_color="FFFFFF",
        end_type="num",
        end_value=1,
        end_color="F8696B",
    )
    ws.conditional_formatting.add(ref, rule)


def _style_monthly(ws: Worksheet, excel_cfg: ExcelConfig) -> None:
    _add_var_separator_border(ws, "变量名")
    _apply_data_bar(ws, "AUC值", color="638EC6")
    _apply_data_bar(ws, "KS值", color="63BE7B")
    # _apply_color_scale(ws, "KS值", ("63BE7B", "FFEB84", "F8696B"))
    # _apply_color_scale(ws, "AUC值", ("63BE7B", "FFEB84", "F8696B"))
    col = _find_col(ws, "PSI值")
    if col:
        _apply_psi_thresholds(ws, col)
    # col = _find_col(ws, "Lift前10%")
    # if col:
    #     letter = get_column_letter(col)
    #     ref = _col_range(ws, col)
    #     ws.conditional_formatting.add(
    #         ref,
    #         FormulaRule(
    #             formula=[f"AND(ISNUMBER({letter}2),{letter}2>1)"],
    #             fill=PatternFill("solid", fgColor="C6EFCE"),
    #         ),
    #     )
    #     ws.conditional_formatting.add(
    #         ref,
    #         FormulaRule(
    #             formula=[f"AND(ISNUMBER({letter}2),{letter}2<1)"],
    #             fill=PatternFill("solid", fgColor="FFC7CE"),
    #         ),
    #     )


def _style_ks(ws: Worksheet, excel_cfg: ExcelConfig) -> None:
    col = _find_col(ws, "分箱区间")
    if col:
        ws.column_dimensions[get_column_letter(col)].width = 17
    _add_var_separator_border(ws, "变量名")
    _apply_color_scale(ws, "坏账率", ("FFFFFF", "F8696B"))
    _apply_data_bar(ws, "Lift值", color="638EC6")
    _apply_data_bar(ws, "KS值", color="63BE7B")
    # _apply_color_scale(ws, "KS值", ("63BE7B", "FFEB84", "F8696B"))
    # _apply_color_scale(ws, "Lift值", ("63BE7B", "FFEB84", "F8696B"))


def _style_plots(ws: Worksheet, excel_cfg: ExcelConfig, plots_dir: str | Path) -> None:
    """将plots_dir中的图片嵌入到Excel"""
    plots_dir = Path(plots_dir)
    if not plots_dir.exists():
        return

    image_files = sorted(plots_dir.glob("*.png"))
    if not image_files:
        return
    
    # 11.85厘米转换为像素
    target_width_cm = 11.85
    target_width_px = int(target_width_cm / 2.54 * 96)
    # 每行3张图，每张图占7列，每行3张图
    cols_per_image = 7
    images_per_row = 3
    rows_per_image = 14
    start_row = 1
    start_col = 1

    for idx, img_path in enumerate(image_files):
        row_offset = idx // images_per_row
        col_offset = idx % images_per_row

        img = Image(str(img_path))
        # 等比例缩放到7列宽度
        original_width = img.width
        img.width = target_width_px
        img.height = int(img.height * (target_width_px / original_width))

        row = start_row + row_offset * rows_per_image
        col = start_col + col_offset * cols_per_image
        cell = ws.cell(row=row, column=col)
        ws.add_image(img, cell.coordinate)


def _style_sample(ws: Worksheet, excel_cfg: ExcelConfig) -> None:
    """样本基础统计表样式"""
    _apply_data_bar(ws, "样本总量", color="638EC6")
    _apply_data_bar(ws, "坏账率", color="F8696B")
    _apply_psi_thresholds(ws, _find_col(ws, "PSI值"))

    # 为样本基础统计表添加簇状柱状图
    _add_monthly_y_chart(ws)


def _add_monthly_y_chart(ws: Worksheet) -> None:
    from openpyxl.chart import BarChart, LineChart, Reference
    from openpyxl.chart.label import DataLabelList
    from openpyxl.drawing.line import LineProperties
    from openpyxl.chart.shapes import GraphicalProperties
    col_yearmonth = _find_col(ws, "年月")
    col_total = _find_col(ws, "样本总量")
    col_bad_rate = _find_col(ws, "坏账率")
    if not all([col_yearmonth, col_total, col_bad_rate]) or ws.max_row < 2:
        logger.warning("图表数据不完整，跳过创建")
        return
    # ================= 柱状图 =================
    bar = BarChart()
    bar.type = "col"
    bar.style = 1
    bar.width, bar.height = 16.8, 10
    bar.y_axis.majorGridlines = None
    data_total = Reference(ws, min_col=col_total, min_row=1, max_row=ws.max_row)
    bar.add_data(data_total, titles_from_data=True)
    cats = Reference(ws, min_col=col_yearmonth, min_row=2, max_row=ws.max_row)
    bar.set_categories(cats)
    bar.dLbls = DataLabelList()
    bar.dLbls.showVal = True
    # 柱状图颜色（蓝色）
    bar.series[0].graphicalProperties.solidFill = "0070C0"
    # ================= 折线图 =================
    line = LineChart()
    data_bad = Reference(ws, min_col=col_bad_rate, min_row=1, max_row=ws.max_row)
    line.add_data(data_bad, titles_from_data=True)
    s = line.series[0]
    s.graphicalProperties.line.solidFill = "FF0000"  # 红色线
    s.graphicalProperties.line.width = 25000
    # 红色点
    s.marker.symbol = "circle"
    s.marker.size = 7
    s.marker.graphicalProperties.solidFill = "FF0000"
    s.marker.graphicalProperties.line = LineProperties(solidFill="FF0000")
    s.dLbls = DataLabelList()
    s.dLbls.showVal = True
    s.dLbls.numFmt = '0.0%'
    line.y_axis.axId = 200
    line.y_axis.crosses = "max"
    line.y_axis.numFmt = '0.0%'
    line.y_axis.majorGridlines = None
    bar.y_axis.crosses = "min"
    bar += line
    bar.legend.position = "t"
    bar.graphical_properties = GraphicalProperties(ln=LineProperties(noFill=True))
    bar.plot_area.graphicalProperties = GraphicalProperties(ln=LineProperties(noFill=True))
    ws.add_chart(bar, f"A{ws.max_row + 2}")



SHEET_STYLERS: dict[str, Callable[[Worksheet, ExcelConfig], None]] = {
    SHEET_SAMPLE: _style_sample,
    SHEET_OVERVIEW: _style_overview,
    SHEET_BINNING: _style_binning,
    SHEET_PSI: _style_psi,
    SHEET_CORR: _style_corr,
    SHEET_MONTHLY: _style_monthly,
    SHEET_KS: _style_ks,
    SHEET_PLOTS: _style_plots,
}


def write_report(
    tables: dict[str, pd.DataFrame],
    path: str | Path,
    cfg,
    df,
    yearmonth_exclude,
    plots_dir: Path | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 检查是否有图片
    has_plots = plots_dir is not None and plots_dir.exists() and list(plots_dir.glob("*.png"))

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for sheet_name, df_table in tables.items():
            safe_name = sheet_name[:31]
            df_table.to_excel(writer, sheet_name=safe_name, index=False)
        # 添加图片sheet
        if has_plots:
            safe_name = SHEET_PLOTS[:31]
            pd.DataFrame().to_excel(writer, sheet_name=safe_name, index=False)

    wb = load_workbook(out)
    for sheet_name, df_table in tables.items():
        safe_name = sheet_name[:31]
        if safe_name not in wb.sheetnames:
            continue
        ws = wb[safe_name]
        _apply_global_style(ws, df_table, cfg.excel)
        styler = SHEET_STYLERS.get(sheet_name)
        if styler:
            styler(ws, cfg.excel)

    # 处理图片sheet
    if has_plots and plots_dir is not None:
        safe_plots_name = SHEET_PLOTS[:31]
        if safe_plots_name in wb.sheetnames:
            ws = wb[safe_plots_name]
            styler = SHEET_STYLERS.get(SHEET_PLOTS)
            if styler:
                styler(ws, cfg.excel, plots_dir)

    wb.save(out)

    return out.resolve()
