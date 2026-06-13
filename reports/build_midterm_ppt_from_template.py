from __future__ import annotations

import html
import io
import json
import math
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
UPLOADED_TEMPLATE = Path(
    "/Users/rtx/Library/Containers/com.tencent.xinWeChat/Data/Documents/"
    "xwechat_files/wxid_mtyo0mzac7bt22_a8c4/msg/file/2026-05/"
    "事件驱动投资策略：以世界杯为例.pptx"
)
FALLBACK_TEMPLATE = ROOT / "reference" / "中期报告_事件驱动投资策略：以世界杯为例.pptx"
TEMPLATE = UPLOADED_TEMPLATE if UPLOADED_TEMPLATE.exists() else FALLBACK_TEMPLATE

OUT_DIR = ROOT / "outputs" / "ppt"
ASSET_DIR = OUT_DIR / "assets"
OUT = OUT_DIR / "中期报告_体育大事件事件驱动量化投资策略与系统_模板扩展版.pptx"
RESULTS = ROOT / "outputs" / "results"

EMU_PER_PX = 9525

REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

ET.register_namespace("", REL_NS)
ET.register_namespace("p", P_NS)
ET.register_namespace("a", A_NS)
ET.register_namespace("r", R_NS)
ET.register_namespace("ct", CT_NS)


def pct(x: float, digits: int = 2) -> str:
    return f"{float(x) * 100:.{digits}f}%"


def pct_num(x: float, digits: int = 2) -> str:
    return f"{float(x):.{digits}f}%"


def money(x: float) -> str:
    return f"{float(x):,.0f}"


def load_metrics():
    summary = json.loads((RESULTS / "run_summary_2026-05-26.json").read_text(encoding="utf-8"))
    orders = pd.read_csv(RESULTS / "paper_orders_2026-05-27.csv")
    trades = pd.read_csv(RESULTS / "backtest_trades_2026-05-26.csv")
    wf = pd.read_csv(RESULTS / "walk_forward_trades_2026-05-26.csv")
    car = pd.read_csv(RESULTS / "event_study_summary_2026-05-26.csv")
    diag = pd.read_csv(RESULTS / "stop_loss_diagnostic_fixed_calendar.csv")
    hyper = pd.read_csv(RESULTS / "hyperparam_search_2026-05-26.csv")
    scores = pd.read_csv(RESULTS / "paper_scores_2026-05-26.csv")
    return summary, orders, trades, wf, car, diag, hyper, scores


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc" if bold else "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy, fnt, fill, width: int, line_gap: int = 5):
    x, y = xy
    line = ""
    for ch in text:
        trial = line + ch
        if draw.textbbox((0, 0), trial, font=fnt)[2] <= width or not line:
            line = trial
        else:
            draw.text((x, y), line, font=fnt, fill=fill)
            y += fnt.size + line_gap
            line = ch
    if line:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def save_chart_event(car: pd.DataFrame, path: Path) -> None:
    img = Image.new("RGBA", (900, 560), "white")
    d = ImageDraw.Draw(img)
    red, dark, blue, gray, light = "#b00000", "#333333", "#2d63c8", "#777777", "#e9edf2"
    title_f, label_f, small_f = font(34, True), font(22, False), font(18, False)
    d.text((30, 22), "事件研究：赛前窗口CAAR", font=title_f, fill=dark)
    d.text((30, 70), "累计异常收益率，市场模型剔除大盘影响", font=small_f, fill=gray)

    events = ["FIFA_WC_2022_OPEN", "UEFA_EURO_2024_OPEN", "PARIS_OLYMPICS_2024_OPEN"]
    labels = ["2022世界杯", "2024欧洲杯", "巴黎奥运会"]
    windows = [("T-60_T-10", "T-60至T-10", red), ("T-30_T-10", "T-30至T-10", blue), ("T-10_T+10", "T-10至T+10", "#777777")]
    vals = []
    for event in events:
        row = []
        for w, _, _ in windows:
            row.append(float(car[(car.event_id == event) & (car.window == w)].iloc[0]["caar"]) * 100)
        vals.append(row)
    ymin, ymax = -12, 30
    x0, y0, w, h = 80, 135, 760, 330
    d.line((x0, y0 + h, x0 + w, y0 + h), fill="#aeb7c2", width=2)
    for tick in [-10, 0, 10, 20, 30]:
        y = y0 + h - (tick - ymin) / (ymax - ymin) * h
        d.line((x0, y, x0 + w, y), fill=light, width=1)
        d.text((28, y - 10), f"{tick}%", font=small_f, fill=gray)
    group_w = w / len(events)
    bar_w = 44
    for gi, row in enumerate(vals):
        center = x0 + group_w * gi + group_w / 2
        for si, val in enumerate(row):
            x = center - 78 + si * 58
            y = y0 + h - (val - ymin) / (ymax - ymin) * h
            zero_y = y0 + h - (0 - ymin) / (ymax - ymin) * h
            top, bottom = min(y, zero_y), max(y, zero_y)
            d.rounded_rectangle((x, top, x + bar_w, bottom), radius=5, fill=windows[si][2])
            d.text((x - 10, top - 24 if val >= 0 else bottom + 4), f"{val:.1f}%", font=small_f, fill=dark)
        d.text((center - 62, y0 + h + 18), labels[gi], font=label_f, fill=dark)
    lx = 80
    for _, name, color in windows:
        d.rounded_rectangle((lx, 500, lx + 24, 514), radius=3, fill=color)
        d.text((lx + 32, 494), name, font=small_f, fill=dark)
        lx += 210
    img.save(path)


def save_chart_backtest(summary: dict, trades: pd.DataFrame, path: Path) -> None:
    img = Image.new("RGBA", (900, 560), "white")
    d = ImageDraw.Draw(img)
    red, dark, gray, light = "#b00000", "#333333", "#777777", "#e9edf2"
    title_f, label_f, small_f, big_f = font(34, True), font(22, False), font(18, False), font(42, True)
    d.text((30, 22), "回测收益贡献", font=title_f, fill=dark)
    b = summary["backtest_summary"]
    kpis = [("净收益", pct(b["net_return"])), ("交易笔数", f"{int(b['num_trades'])}笔"), ("交易成本", money(b["total_cost"]) + "元")]
    for i, (lab, val) in enumerate(kpis):
        x = 30 + i * 285
        d.rounded_rectangle((x, 82, x + 250, 168), radius=12, outline="#d6dce4", width=2, fill="#fafafa")
        d.text((x + 20, 98), lab, font=small_f, fill=gray)
        d.text((x + 20, 122), val, font=big_f if i == 0 else label_f, fill=red if i == 0 else dark)

    by_event = trades.groupby("event_id").agg(net=("net_return_on_initial_cash", "sum"), gross=("gross_return", "mean"))
    items = [("2022世界杯", by_event.loc["FIFA_WC_2022_OPEN", "net"] * 100), ("2024欧洲杯", by_event.loc["UEFA_EURO_2024_OPEN", "net"] * 100)]
    x0, y0, w, h = 120, 230, 650, 190
    maxv = max(v for _, v in items) * 1.2
    for i, (label, val) in enumerate(items):
        y = y0 + i * 82
        d.text((30, y + 12), label, font=label_f, fill=dark)
        d.rounded_rectangle((x0, y + 12, x0 + w, y + 46), radius=8, fill=light)
        bw = max(6, int(w * val / maxv))
        d.rounded_rectangle((x0, y + 12, x0 + bw, y + 46), radius=8, fill=red if i == 0 else "#2d63c8")
        d.text((x0 + bw + 14, y + 9), f"+{val:.2f}%", font=label_f, fill=dark)
    d.text((30, 470), "结论：收益主要来自世界杯强预热期，欧洲杯提供同类事件扩展验证。", font=label_f, fill=dark)
    img.save(path)


def save_chart_stop(diag: pd.DataFrame, summary: dict, path: Path) -> None:
    img = Image.new("RGBA", (900, 560), "white")
    d = ImageDraw.Draw(img)
    red, dark, blue, gray, light = "#b00000", "#333333", "#2d63c8", "#777777", "#e9edf2"
    title_f, label_f, small_f = font(34, True), font(22, False), font(18, False)
    d.text((30, 22), "止损触发原因诊断", font=title_f, fill=dark)
    fixed_net = diag["net_return_on_initial_cash"].sum() * 100
    fixed_stop = (diag["exit_reason"].eq("stop_loss").sum() / len(diag)) * 100
    current_net = summary["backtest_summary"]["net_return"] * 100
    current_stop = 0.0
    panels = [("固定日历压力测试", fixed_net, fixed_stop), ("当前中期系统", current_net, current_stop)]
    for i, (name, net, stop) in enumerate(panels):
        x = 55 + i * 430
        d.rounded_rectangle((x, 100, x + 360, 315), radius=14, outline="#d6dce4", width=2, fill="#fafafa")
        d.text((x + 24, 124), name, font=label_f, fill=dark)
        d.text((x + 24, 178), "净收益率", font=small_f, fill=gray)
        d.text((x + 150, 168), f"{net:.2f}%", font=font(34, True), fill=red if net >= 0 else blue)
        d.text((x + 24, 242), "止损率", font=small_f, fill=gray)
        d.text((x + 150, 232), f"{stop:.2f}%", font=font(34, True), fill=blue if stop == 0 else red)
    reasons = [
        "T-90固定入场过早：事件热度尚未稳定启动，容易被普通波动洗出。",
        "巴黎奥运会对当前A股体育候选池适配度不足，属于应过滤事件。",
        "固定8%止损忽略个股波动率差异，当前系统改用事件门槛+动态止损。",
    ]
    y = 355
    for r in reasons:
        d.rounded_rectangle((55, y, 845, y + 48), radius=8, fill=light)
        d.text((76, y + 12), "• " + r, font=small_f, fill=dark)
        y += 58
    img.save(path)


def save_chart_hyper(summary: dict, wf: pd.DataFrame, path: Path) -> None:
    img = Image.new("RGBA", (900, 560), "white")
    d = ImageDraw.Draw(img)
    red, dark, gray, light = "#b00000", "#333333", "#777777", "#e9edf2"
    title_f, label_f, small_f = font(34, True), font(22, False), font(18, False)
    d.text((30, 22), "参数验证与走步结果", font=title_f, fill=dark)
    params = summary["best_params"]
    rows = [
        ("入场/退出", f"T-{params['entry_days_before_event']} / T-{params['exit_days_before_event']}", "由搜索选择事件预热窗口"),
        ("TopK", str(params["top_k"]), "控制单事件持仓数量"),
        ("事件适配门槛", str(params["min_event_fit_for_trade"]), "过滤低适配赛事"),
        ("暴露门槛", str(params["min_exposure_for_trade"]), "过滤弱产业链关系"),
        ("动态止损", f"{pct(params['stop_loss'], 0)} + {params['stop_vol_multiplier']}倍波动", "避免正常波动触发"),
    ]
    x0, y0 = 35, 90
    colw = [170, 210, 425]
    headers = ["参数", "取值", "金融含义"]
    x = x0
    for cw, h in zip(colw, headers):
        d.rounded_rectangle((x, y0, x + cw, y0 + 42), radius=5, fill=red)
        d.text((x + 12, y0 + 10), h, font=small_f, fill="white")
        x += cw
    y = y0 + 48
    for row in rows:
        x = x0
        for cw, txt in zip(colw, row):
            d.rectangle((x, y, x + cw, y + 48), outline="#d6dce4", fill="#ffffff")
            d.text((x + 12, y + 12), txt, font=small_f, fill=dark)
            x += cw
        y += 48
    wsum = summary["walk_forward_summary"]
    d.rounded_rectangle((80, 405, 820, 515), radius=12, outline="#d6dce4", width=2, fill="#fafafa")
    d.text((105, 425), "走步验证：用前一事件训练，后一事件验证", font=label_f, fill=dark)
    d.text((105, 470), f"净收益 {pct(wsum['net_return'])}   |   交易 {int(wsum['num_trades'])}笔   |   平均毛收益 {pct(wsum['gross_return_mean'])}", font=label_f, fill=red)
    img.save(path)


def generate_charts(summary, trades, wf, car, diag) -> dict[str, Path]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "event": ASSET_DIR / "chart_event_study.png",
        "backtest": ASSET_DIR / "chart_backtest_contribution.png",
        "stop": ASSET_DIR / "chart_stop_loss.png",
        "hyper": ASSET_DIR / "chart_hyperparams.png",
    }
    save_chart_event(car, paths["event"])
    save_chart_backtest(summary, trades, paths["backtest"])
    save_chart_stop(diag, summary, paths["stop"])
    save_chart_hyper(summary, wf, paths["hyper"])
    return paths


def pad(items: list[str], n: int) -> list[str]:
    items = [str(x) for x in items]
    if len(items) < n:
        items += [""] * (n - len(items))
    return items[:n]


def replace_slide_text(xml: str, texts: list[str]) -> str:
    i = 0

    def repl(_: re.Match) -> str:
        nonlocal i
        value = texts[i] if i < len(texts) else ""
        i += 1
        return f"<a:t>{html.escape(value, quote=False)}</a:t>"

    return re.sub(r"<a:t>.*?</a:t>", repl, xml, flags=re.DOTALL)


def max_shape_id(xml: str) -> int:
    ids = [int(x) for x in re.findall(r"<p:cNvPr[^>]+ id=\"(\d+)\"", xml)]
    return max(ids) if ids else 1000


def add_picture(xml: str, rel_id: str, name: str, bbox_px: tuple[int, int, int, int]) -> str:
    x, y, w, h = [int(v * EMU_PER_PX) for v in bbox_px]
    sid = max_shape_id(xml) + 1
    pic = f"""
<p:pic>
  <p:nvPicPr>
    <p:cNvPr id="{sid}" name="{html.escape(name)}"/>
    <p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>
    <p:nvPr/>
  </p:nvPicPr>
  <p:blipFill>
    <a:blip r:embed="{rel_id}"/>
    <a:stretch><a:fillRect/></a:stretch>
  </p:blipFill>
  <p:spPr>
    <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
  </p:spPr>
</p:pic>"""
    return xml.replace("</p:spTree>", pic + "\n</p:spTree>", 1)


def next_rid(root: ET.Element) -> str:
    max_id = 0
    for rel in root.findall(f"{{{REL_NS}}}Relationship"):
        rid = rel.attrib.get("Id", "")
        m = re.match(r"rId(\d+)$", rid)
        if m:
            max_id = max(max_id, int(m.group(1)))
    return f"rId{max_id + 1}"


def rels_with_notes_and_images(src_rels_xml: str, slide_no: int, image_targets: list[str]) -> tuple[bytes, list[str]]:
    root = ET.fromstring(src_rels_xml)
    for rel in list(root):
        if rel.attrib.get("Type", "").endswith("/notesSlide"):
            root.remove(rel)
    note_rid = next_rid(root)
    ET.SubElement(
        root,
        f"{{{REL_NS}}}Relationship",
        {
            "Id": note_rid,
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide",
            "Target": f"../notesSlides/notesSlide{slide_no}.xml",
        },
    )
    image_rids: list[str] = []
    for target in image_targets:
        rid = next_rid(root)
        image_rids.append(rid)
        ET.SubElement(
            root,
            f"{{{REL_NS}}}Relationship",
            {
                "Id": rid,
                "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                "Target": f"../media/{target}",
            },
        )
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True), image_rids


def build_notes_slide_xml(note: str) -> bytes:
    with zipfile.ZipFile(TEMPLATE) as z:
        xml = z.read("ppt/notesSlides/notesSlide1.xml").decode("utf-8")
    texts = ["2026.05.27", note, "‹#›"]
    xml = replace_slide_text(xml, texts)
    return xml.encode("utf-8")


def build_notes_rels_xml(slide_no: int) -> bytes:
    root = ET.Element(f"{{{REL_NS}}}Relationships")
    ET.SubElement(
        root,
        f"{{{REL_NS}}}Relationship",
        {
            "Id": "rId2",
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster",
            "Target": "../notesMasters/notesMaster1.xml",
        },
    )
    ET.SubElement(
        root,
        f"{{{REL_NS}}}Relationship",
        {
            "Id": "rId1",
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide",
            "Target": f"../slides/slide{slide_no}.xml",
        },
    )
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True)


def update_presentation_xml(xml: bytes, slide_count: int, rids: list[str]) -> bytes:
    root = ET.fromstring(xml)
    sld_lst = root.find(f"{{{P_NS}}}sldIdLst")
    if sld_lst is None:
        raise ValueError("presentation.xml missing sldIdLst")
    for child in list(sld_lst):
        sld_lst.remove(child)
    for i in range(slide_count):
        ET.SubElement(sld_lst, f"{{{P_NS}}}sldId", {"id": str(256 + i), f"{{{R_NS}}}id": rids[i]})
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True)


def update_presentation_rels(xml: bytes, slide_count: int) -> tuple[bytes, list[str]]:
    root = ET.fromstring(xml)
    for rel in list(root):
        if rel.attrib.get("Type", "").endswith("/slide"):
            root.remove(rel)
    rids: list[str] = []
    for i in range(1, slide_count + 1):
        rid = next_rid(root)
        rids.append(rid)
        ET.SubElement(
            root,
            f"{{{REL_NS}}}Relationship",
            {
                "Id": rid,
                "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide",
                "Target": f"slides/slide{i}.xml",
            },
        )
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True), rids


def update_content_types(xml: bytes, slide_count: int) -> bytes:
    text = xml.decode("utf-8")
    additions: list[str] = []

    def add(part: str, ctype: str) -> None:
        if f'PartName="{part}"' not in text and part not in "".join(additions):
            additions.append(f'<Override PartName="{part}" ContentType="{ctype}"/>')

    for i in range(1, slide_count + 1):
        add(f"/ppt/slides/slide{i}.xml", "application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
        add(f"/ppt/notesSlides/notesSlide{i}.xml", "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml")
    if additions:
        text = text.replace("</Types>", "".join(additions) + "</Types>")
    return text.encode("utf-8")


def update_app_xml(xml: bytes, slide_count: int) -> bytes:
    text = xml.decode("utf-8")
    text = re.sub(r"<Slides>\d+</Slides>", f"<Slides>{slide_count}</Slides>", text)
    text = re.sub(r"<Notes>\d+</Notes>", f"<Notes>{slide_count}</Notes>", text)
    return text.encode("utf-8")


def slide_plan(summary, orders, trades, wf, car, diag, hyper, scores):
    b = summary["backtest_summary"]
    w = summary["walk_forward_summary"]
    bp = summary["best_params"]
    fifa_pre = car[(car.event_id == "FIFA_WC_2022_OPEN") & (car.window == "T-60_T-10")].iloc[0]
    fifa_near = car[(car.event_id == "FIFA_WC_2022_OPEN") & (car.window == "T-10_T+10")].iloc[0]
    order_rows = {row["ticker"]: row for _, row in orders.iterrows()}

    def order_card(ticker: str, fallback: str):
        row = order_rows.get(ticker)
        if row is None:
            return [fallback, "未入选", "风险收益比不足", "当前时点未进入模拟盘订单。"]
        return [
            row["company"],
            f"{row['ticker']} | {pct(row['target_weight'])}",
            f"{row['target_shares']}股 · {money(row['estimated_trade_value'])}元",
            "按2026-05-26收盘价、100股整数手和成本约束生成。",
        ]

    notes = {}
    slides = [
        {
            "src": 1,
            "texts": ["体育大事件事件驱动投资策略与系统", "答辩人：", "指导教师：", "软件与微电子学院"],
            "note": "各位老师好，本次中期汇报的主题是体育大事件事件驱动量化投资策略与系统。我们仍以2026 FIFA世界杯作为核心案例，但已经把策略扩展为可支持欧洲杯、奥运会等体育大事件的自动化系统。",
        },
        {
            "src": 2,
            "texts": ["目录", "CONTENTS", "策略基本思想", "01", "核心流程", "02", "回测及结果分析", "03", "模拟交易配置 / 总结展望", "04"],
            "note": "汇报结构仍沿用原PPT框架：先讲策略基本思想，再讲系统流程，然后展示回测和模拟盘配置，最后单独总结展望。重点回应中期要求中的回测、模拟交易和持仓配置理由。",
        },
        {"src": 3, "texts": ["PART 01", "策略基本思想"], "note": "第一部分说明为什么体育大事件可以作为事件驱动策略，以及我们如何从世界杯主题扩展到体育大事件框架。"},
        {
            "src": 4,
            "texts": [
                "投资策略 | 价值发现",
                "策略核心：事件预期差",
                "01 / 05",
                "盈利的核心公式",
                "事件收益 = 预期修正",
                "- 已被价格反映的部分",
                "投资获利不是赌比赛结果",
                "而是捕捉赛前注意力扩散",
                "盈利的本质是“预期差”",
                "大型体育赛事的日期、主题和传播强度高度确定。市场常在开幕前逐步交易产业链受益预期，开幕后则容易出现利好兑现。",
                "● 正向预期差：",
                "赛事关注度与产业链受益未被充分定价 → 相关资产提前重估",
                "● 负向预期差：",
                "热度拥挤或事件兑现不足 → 主题退潮与回撤风险上升",
                "🎯 策略目标：",
                "在事件确定、热度启动、产业链相关、风险可控的阶段参与，并在市场充分兑现前退出。",
            ],
            "note": "这一页强调我们不是预测比赛胜负，也不是简单炒概念，而是研究事件临近时市场预期如何修正。策略真正想赚的是赛前预期差，而不是比赛当天的结果。",
        },
        {
            "src": 5,
            "texts": [
                "思想自由兼容并包",
                "宏观背景：产业增长与事件频率扩展",
                "< >",
                "📈 体育产业与赛事周期",
                "长期背景：",
                "体育产业规模持续增长，顶级赛事具备稳定传播周期。",
                "事件日历：",
                "世界杯、欧洲杯、奥运会、亚运会、赛程公布、赞助官宣等。",
                "💹 金融含义",
                "老师指出只做世界杯频率太低，因此系统保留世界杯主线，同时建立体育事件库，用同一套框架验证不同事件的可交易性。",
                "💡 策略启示",
                "体育大事件不是单一概念炒作，而是“确定性事件 + 注意力迁移 + 产业链重估”的组合。系统要判断哪些事件值得交易、何时交易、买多少。",
            ],
            "note": "老师之前的反馈非常关键：只做世界杯，频率太低，不符合量化投资的可重复验证要求。所以我们没有改主题，而是把世界杯作为核心案例，把系统扩展成体育大事件事件库。",
        },
        {
            "src": 4,
            "texts": [
                "投资策略 | 理论检验",
                "三条可检验假设",
                "02 / 05",
                "从“故事”到“检验”",
                "假设必须能被回测否证",
                "而不是事后解释",
                "H1：赛前存在异常收益",
                "用事件研究法计算CAAR检验",
                "H2：事件适配度决定是否交易",
                "不是所有体育赛事都能映射到A股标的。巴黎奥运会样本显示，低适配事件不应强行参与。",
                "● H3：注意力与价格动量共同影响收益：",
                "成交额冲击代表关注度升温，价格动量代表市场反应不足后的延续。",
                "● 风控假设：",
                "先过滤低质量事件，再使用动态止损，优于机械固定止损。",
                "🎯 研究方法：",
                "使用事件研究、RankIC因子权重、超参数搜索和走步验证共同检验。",
            ],
            "note": "为了避免做成主观故事，我们把策略拆成三条可检验假设。后面的CAAR、止损诊断和走步验证，都是围绕这些假设来展开。",
        },
        {"src": 6, "texts": ["PART 02", "核心流程", "System Design"], "note": "第二部分介绍系统流程。我们可以人工调整策略方向，但从数据到订单的过程必须自动化，减少主观挑选结果。"},
        {
            "src": 7,
            "texts": [
                "策略核心流程",
                "PAGE 01",
                "核心流程：事件识别与历史回溯",
                "步骤一：事件识别与筛选",
                "● 确定性与可预测性：",
                "赛事日期、主题和信息来源必须可验证",
                "● 高关注度：",
                "能带来搜索、新闻、社媒和成交额升温",
                "● 经济影响力：",
                "与显示设备、体育设施、传媒、IP衍生品等链条相关",
                "● 时间窗口：",
                "必须存在赛前预热期，避免开幕后追高",
                "步骤二：历史数据回溯",
                "▶ 板块表现分析：",
                "计算事件窗口收益、异常收益AR与累计异常收益CAR。",
                "▶ 龙头股识别：",
                "观察海信视像、中体产业、共创草坪等标的在事件前的反应。",
                "▶ 预期差验证：",
                "验证“赛前预热、临近兑现”的规律是否成立。",
                "历史复盘结论",
                "2022世界杯T-60至T-10窗口CAAR为",
                pct(fifa_pre["caar"]),
                "，正CAR比例为",
                pct(fifa_pre["positive_rate"]),
                "；T-10至T+10转为",
                pct(fifa_near["caar"]),
                "，支持赛前退出。",
            ],
            "note": "这页保留原来事件识别和历史回溯的逻辑，但把证据换成真实测试数据。2022世界杯赛前60到10日CAAR为26.54%，开幕前后窗口反而转负，说明赛前交易比开幕后追高更合理。",
        },
        {
            "src": 7,
            "texts": [
                "策略核心流程",
                "PAGE 02",
                "自动化系统：事件到订单",
                "步骤一：事件输入",
                "● 事件库：",
                "世界杯、欧洲杯、奥运会、赛程公布等进入事件雷达。",
                "● 股票池：",
                "显示设备、体育设施、体育运营、传媒彩票等候选标的。",
                "● 数据快照：",
                "冻结决策日行情，防止超前信息。",
                "● 输出：",
                "形成候选事件与股票关系。",
                "步骤二：证据闭环",
                "▶ 来源与时间：",
                "证据不得晚于决策日。",
                "▶ 实体匹配：",
                "股票代码、公司名与股票池一致。",
                "▶ Schema校验：",
                "字段完整、类型合法。",
                "闭环检验结论",
                "候选关系10条",
                "通过10条",
                "幻觉代理率0%",
                "规则Agent保证可复现",
                "交易输出",
                "因子评分生成目标权重，再按100股整数手取整。",
                "单股12%、行业25%、成本和涨跌停约束订单前检查。",
                "✅ 可审计、可复盘",
            ],
            "note": "这里重点回应LLM可信性问题。LLM或规则Agent只负责信息结构化，不直接决定买卖。每条关系必须经过来源、时间和实体校验，避免把幻觉信息放进交易模型。",
        },
        {
            "src": 7,
            "texts": [
                "策略评分流程",
                "PAGE 03",
                "因子评分：用金融含义解释模型",
                "步骤三：标的筛选",
                "● 低波动：",
                "40.69%，控制事件交易中的下行波动。",
                "● 注意力：",
                "25.75%，用成交额冲击代理关注度。",
                "● 事件暴露：",
                "16.50%，衡量公司与赛事链条关系。",
                "● 动量/流动性：",
                "刻画反应不足和成交可行性。",
                "步骤四：权重生成",
                "▶ RankIC估计：",
                "由训练样本计算因子方向与强弱。",
                "▶ 理论约束：",
                "保留金融上合理的方向。",
                "▶ 成本约束：",
                "成交额、滑点和佣金进入订单。",
                "核心评分结论",
                "低波动权重最高",
                "说明事件主题股需先控风险",
                "注意力与暴露决定能否入选",
                "动量和流动性决定交易可行",
                "组合输出",
                "Alpha评分 → 目标权重 → 模拟盘订单",
                "海信12.00%、共创3.50%、中体2.59%",
                "✅ 金融含义 + 数据计算",
            ],
            "note": "因子设计不是拍脑袋。低波动、注意力、事件暴露、动量和流动性都有金融含义，权重由历史训练样本的RankIC估计，而不是我们主观给分。",
        },
        {"src": 9, "texts": ["PART 03", "回测及结果分析", "Backtesting Analysis"], "note": "第三部分进入回测和结果分析，对应中期要求中的至少一轮回测和结果解释。"},
        {
            "src": 10,
            "texts": [
                "PART 03",
                "回测设定与实验规则",
                "▍ 回测设定",
                "• 回测事件：2022世界杯、2024欧洲杯、2024巴黎奥运会",
                "• 股票池：10只A股体育事件候选标的",
                f"• 交易窗口：赛前{bp['entry_days_before_event']}天进入，赛前{bp['exit_days_before_event']}天退出",
                "• 成本约束：佣金、印花税、滑点、100股整数手",
                "▍ 系统规则",
                "事件库 → 证据核验 → 因子评分 → 参数搜索 → 组合约束 → 回测日志",
                "参数不是凭空设定，而是通过历史事件和走步验证选择。",
                "",
                "",
                "▍ 成本设置",
                "佣金0.03%，最低5元；卖出印花税0.05%；滑点按流动性分层估计。",
                "▍ 市场限制",
                "仅做多、A股T+1、涨跌停、停牌过滤、单股与行业仓位上限。",
            ],
            "note": "这页说明实验规则。回测不再是简单等权买卖，而是把交易成本、A股整数手、涨跌停、仓位上限和事件适配门槛都纳入系统。",
        },
        {
            "src": 11,
            "texts": [
                "事件研究",
                "|",
                "CAAR检验",
                "3.1",
                "事件研究结果",
                "PAGE 09 / 24",
                "关键指标表现 (Key Metrics)",
                "2022世界杯赛前",
                "+26.54%",
                "2022世界杯开幕前后",
                "-8.45%",
                "欧洲杯赛前",
                "+3.18%",
                "巴黎奥运赛前",
                "-5.00%",
                "💡 分析结论：",
                "世界杯赛前窗口异常收益最明显，欧洲杯提供弱正向扩展验证；巴黎奥运会在当前A股候选池中适配度不足。事件研究支持“赛前布局、赛前退出”，而不是开幕后追高。",
            ],
            "note": "这页是最重要的实证证据。CAAR表示累计异常收益，已经剔除了大盘影响。世界杯赛前表现显著，但开幕前后转负，说明策略需要在赛前退出。",
            "images": [{"key": "event", "bbox": (70, 145, 540, 430)}],
        },
        {
            "src": 11,
            "texts": [
                "策略复盘",
                "|",
                "数据回测",
                "3.2",
                "回测收益与成本",
                "PAGE 10 / 24",
                "关键指标表现 (Key Metrics)",
                "回测净收益率",
                pct(b["net_return"]),
                "交易笔数",
                f"{int(b['num_trades'])}笔",
                "单笔平均毛收益",
                pct(b["gross_return_mean"]),
                "交易成本",
                money(b["total_cost"]) + "元",
                "💡 分析结论：",
                "在扣除佣金、印花税和滑点后，系统回测净收益为11.04%。收益主要来自2022世界杯强预热期，2024欧洲杯贡献较小但仍为正；样本仍有限，不夸大年化收益。",
            ],
            "note": "回测结果显示，扣除真实交易摩擦后，策略净收益为11.04%。这里不做夸张年化，因为事件频率低，中期更重要的是证明系统能形成可执行交易并获得初步正收益。",
            "images": [{"key": "backtest", "bbox": (70, 145, 540, 430)}],
        },
        {
            "src": 11,
            "texts": [
                "风险复盘",
                "|",
                "止损诊断",
                "3.3",
                "止损触发原因分析",
                "PAGE 11 / 24",
                "关键指标表现 (Key Metrics)",
                "固定日历止损率",
                "55.56%",
                "当前系统止损率",
                "0.00%",
                "压力测试净收益",
                "+2.70%",
                "当前系统净收益",
                "+11.04%",
                "💡 分析结论：",
                "止损触发并不只代表风险控制有效，也可能说明入场过早或事件不适配。当前系统先过滤低适配事件，再使用动态止损，避免被正常波动过早洗出。",
            ],
            "note": "我们专门分析了前一版容易触发止损的原因。固定T-90入场和固定8%止损过于机械，尤其巴黎奥运会与当前股票池适配不足。因此当前系统先过滤事件，再使用动态止损。",
            "images": [{"key": "stop", "bbox": (70, 145, 540, 430)}],
        },
        {
            "src": 10,
            "texts": [
                "PART 03",
                "超参数搜索与走步验证",
                "▍ 参数不是主观设定",
                "• 搜索对象：注意力窗口、动量窗口、入场日、TopK、止损线、事件适配门槛",
                f"• 当前窗口：T-{bp['entry_days_before_event']}进入，T-{bp['exit_days_before_event']}退出",
                f"• 当前TopK：{bp['top_k']}，事件适配门槛：{bp['min_event_fit_for_trade']}",
                "• 选择原则：验证收益、全样本收益和止损率共同约束",
                "▍ 走步验证",
                f"用前一事件训练，后一事件验证：净收益{pct(w['net_return'])}，交易{int(w['num_trades'])}笔。",
                "样本仍小，但说明信号没有在扩展验证中立即失效。",
                "",
                "",
                "▍ 结果解释",
                "走步验证全部交易为正，但不能单独证明长期稳定。",
                "▍ 下一步",
                "结项前继续加入模拟盘真实表现，并扩充更多体育事件样本。",
            ],
            "note": "这页说明参数来源。我们没有凭感觉设定入场日、TopK和止损线，而是通过参数搜索选择，同时使用走步验证防止只在样本内好看。",
            "images": [{"key": "hyper", "bbox": (708, 426, 520, 240)}],
        },
        {"src": 12, "texts": ["PART 04", "模拟交易配置", "Simulation & Configuration"], "note": "第四部分是中期要求中特别强调的模拟交易配置。这里的订单基于2026年5月26日收盘真实数据生成。"},
        {
            "src": 13,
            "texts": [
                "▶ 来源与时间：\n证据不得晚于决策日。",
                "▶ 实体匹配：\n股票代码、公司名与股票池一致。",
                "闭环检验结论",
                "候选关系10条\n通过10条\n幻觉代理率0%",
                "中期版本先用规则Agent保证可复现；后续可接入OpenAI API。",
                "交易输出",
                "因子评分后生成目标权重，再按100股整数手取整。",
                "单股12%、行业25%、成本和涨跌停约束，全部在订单前检查。",
                "✅ 可审计、可复盘",
            ],
            "note": "这里重点回应LLM可信性问题。LLM或规则Agent只负责信息结构化，不直接决定买卖。每条关系必须经过来源、时间和实体校验，避免把幻觉信息放进交易模型。",
        },
        {
            "src": 7,
            "texts": [
                "策略评分流程",
                "PAGE 03",
                "因子评分：用金融含义解释模型",
                "步骤三：标的筛选",
                "● 低波动：",
                "40.69%，控制事件交易中的下行波动。",
                "● 注意力：",
                "25.75%，用成交额冲击代理关注度。",
                "● 事件暴露：",
                "16.50%，衡量公司与赛事链条关系。",
                "步骤四：权重生成",
                "▶ RankIC估计：\n由训练样本计算因子方向与强弱。",
                "▶ 理论约束：\n保留金融上合理的正向/负向方向。",
                "▶ 成本约束：\n成交额、滑点和佣金进入订单。",
                "核心评分结论",
                "低波动权重最高\n说明事件主题股需先控风险",
                "注意力和事件暴露决定能否入选，动量和流动性决定交易可行性。",
                "组合输出",
                "Alpha评分 → 目标权重 → 模拟盘订单",
                "海信视像12.00%\n共创草坪3.50%\n中体产业2.59%",
                "✅ 金融含义 + 数据计算",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            "note": "因子设计不是拍脑袋。低波动、注意力、事件暴露、动量和流动性都有金融含义，权重由历史训练样本的RankIC估计，而不是我们主观给分。",
        },
        {"src": 9, "texts": ["PART 03", "回测及结果分析", "Backtesting Analysis"], "note": "第三部分进入回测和结果分析，对应中期要求中的至少一轮回测和结果解释。"},
        {
            "src": 10,
            "texts": [
                "PART 03",
                "回测设定与实验规则",
                "▍ 回测设定",
                "• 回测事件：2022世界杯、2024欧洲杯、2024巴黎奥运会",
                "• 股票池：10只A股体育事件候选标的",
                f"• 交易窗口：赛前{bp['entry_days_before_event']}天进入，赛前{bp['exit_days_before_event']}天退出",
                "• 成本约束：佣金、印花税、滑点、100股整数手",
                "▍ 系统规则",
                "事件库 → 证据核验 → 因子评分 → 参数搜索 → 组合约束 → 回测日志",
                "参数不是凭空设定，而是通过历史事件和走步验证选择。",
                "",
                "",
                "▍ 成本设置",
                "佣金0.03%，最低5元；卖出印花税0.05%；滑点按流动性分层估计。",
                "▍ 市场限制",
                "仅做多、A股T+1、涨跌停、停牌过滤、单股与行业仓位上限。",
            ],
            "note": "这页说明实验规则。回测不再是简单等权买卖，而是把交易成本、A股整数手、涨跌停、仓位上限和事件适配门槛都纳入系统。",
        },
        {
            "src": 11,
            "texts": [
                "事件研究",
                "|",
                "CAAR检验",
                "3.1",
                "事件研究结果",
                "PAGE 09 / 24",
                "关键指标表现 (Key Metrics)",
                "2022世界杯赛前",
                "+26.54%",
                "2022世界杯开幕前后",
                "-8.45%",
                "欧洲杯赛前",
                "+3.18%",
                "巴黎奥运赛前",
                "-5.00%",
                "💡 分析结论：",
                "世界杯赛前窗口异常收益最明显，欧洲杯提供弱正向扩展验证；巴黎奥运会在当前A股候选池中适配度不足。事件研究支持“赛前布局、赛前退出”，而不是开幕后追高。",
            ],
            "note": "这页是最重要的实证证据。CAAR表示累计异常收益，已经剔除了大盘影响。世界杯赛前表现显著，但开幕前后转负，说明策略需要在赛前退出。",
            "images": [{"key": "event", "bbox": (70, 145, 540, 430)}],
        },
        {
            "src": 11,
            "texts": [
                "策略复盘",
                "|",
                "数据回测",
                "3.2",
                "回测收益与成本",
                "PAGE 10 / 24",
                "关键指标表现 (Key Metrics)",
                "回测净收益率",
                pct(b["net_return"]),
                "交易笔数",
                f"{int(b['num_trades'])}笔",
                "单笔平均毛收益",
                pct(b["gross_return_mean"]),
                "交易成本",
                money(b["total_cost"]) + "元",
                "💡 分析结论：",
                "在扣除佣金、印花税和滑点后，系统回测净收益为11.04%。收益主要来自2022世界杯强预热期，2024欧洲杯贡献较小但仍为正；样本仍有限，不夸大年化收益。",
            ],
            "note": "回测结果显示，扣除真实交易摩擦后，策略净收益为11.04%。这里不做夸张年化，因为事件频率低，中期更重要的是证明系统能形成可执行交易并获得初步正收益。",
            "images": [{"key": "backtest", "bbox": (70, 145, 540, 430)}],
        },
        {
            "src": 11,
            "texts": [
                "风险复盘",
                "|",
                "止损诊断",
                "3.3",
                "止损触发原因分析",
                "PAGE 11 / 24",
                "关键指标表现 (Key Metrics)",
                "固定日历止损率",
                "55.56%",
                "当前系统止损率",
                "0.00%",
                "压力测试净收益",
                "+2.70%",
                "当前系统净收益",
                "+11.04%",
                "💡 分析结论：",
                "止损触发并不只代表风险控制有效，也可能说明入场过早或事件不适配。当前系统先过滤低适配事件，再使用动态止损，避免被正常波动过早洗出。",
            ],
            "note": "我们专门分析了前一版容易触发止损的原因。固定T-90入场和固定8%止损过于机械，尤其巴黎奥运会与当前股票池适配不足。因此当前系统先过滤事件，再使用动态止损。",
            "images": [{"key": "stop", "bbox": (70, 145, 540, 430)}],
        },
        {
            "src": 10,
            "texts": [
                "PART 03",
                "超参数搜索与走步验证",
                "▍ 参数不是主观设定",
                "• 搜索对象：注意力窗口、动量窗口、入场日、TopK、止损线、事件适配门槛",
                f"• 当前窗口：T-{bp['entry_days_before_event']}进入，T-{bp['exit_days_before_event']}退出",
                f"• 当前TopK：{bp['top_k']}，事件适配门槛：{bp['min_event_fit_for_trade']}",
                "• 选择原则：验证收益、全样本收益和止损率共同约束",
                "▍ 走步验证",
                f"用前一事件训练，后一事件验证：净收益{pct(w['net_return'])}，交易{int(w['num_trades'])}笔。",
                "样本仍小，但说明信号没有在扩展验证中立即失效。",
                "",
                "",
                "▍ 结果解释",
                "走步验证全部交易为正，但不能单独证明长期稳定。",
                "▍ 下一步",
                "结项前继续加入模拟盘真实表现，并扩充更多体育事件样本。",
            ],
            "note": "这页说明参数来源。我们没有凭感觉设定入场日、TopK和止损线，而是通过参数搜索选择，同时使用走步验证防止只在样本内好看。",
            "images": [{"key": "hyper", "bbox": (708, 426, 520, 240)}],
        },
        {"src": 12, "texts": ["PART 04", "模拟交易配置", "Simulation & Configuration"], "note": "第四部分是中期要求中特别强调的模拟交易配置。这里的订单基于2026年5月26日收盘真实数据生成。"},
        {
            "src": 13,
            "texts": [
                "PART 04",
                "标的选择与配置理由",
                "Target Selection & Allocation Reasons",
                *order_card("600060.SH", "海信视像"),
                *order_card("605099.SH", "共创草坪"),
                *order_card("600158.SH", "中体产业"),
                "未入选标的",
                "元隆雅图/粤传媒/舒华体育等",
                "热度或波动约束不足",
                "系统保留其主题属性，但当前风险收益比不如入选标的。",
            ],
            "note": "模拟盘最终选入海信视像、共创草坪和中体产业。海信评分最高，达到单股12%上限；共创草坪和中体产业提供体育设施与体育运营链条补充。",
        },
        {
            "src": 14,
            "texts": [
                "思想自由兼容并包",
                "模拟交易配置：资金管理与建仓计划",
                "< >",
                "总仓位控制",
                "本次策略总资金投入不超过投资组合的",
                "30%",
                "，当前系统目标仓位为",
                pct(summary["paper_total_target_weight"]),
                "，保留现金应对临近开幕期波动。",
                "建仓周期规划",
                "订单生成方式",
                "组合配置方式",
                "上限30% | 严控追高",
                "2026-05-26收盘生成",
                "2026-05-27模拟执行",
                "按模型权重与100股整数手取整",
                "当前已接近2026-06-11世界杯开幕日，系统不再采用六个月慢速建仓，而是定位为赛前最后预热窗口的审慎模拟交易。",
                "系统只使用2026-05-26收盘前可获得数据，避免模拟盘中的超前信息。",
                f"预计买入金额{money(summary['paper_total_estimated_trade_value'])}元，预计成本{money(summary['paper_total_expected_cost'])}元，剩余资金保留现金。",
            ],
            "note": "因为现在已经是2026年5月26日，距离开幕较近，所以模拟盘不是满仓或长期定投，而是18.09%的审慎仓位。所有股数都按100股整数手取整。",
        },
        {
            "src": 15,
            "texts": [
                "04",
                "模拟交易配置：止盈止损与复盘规则",
                "第一止盈点",
                "事件兑现前 · 锁定收益",
                "若组合收益达到",
                "15%",
                "，先卖出",
                "50%",
                "仓位，降低开幕前拥挤交易风险。",
                "第二止盈点",
                "赛前退出 · 不赌兑现",
                "世界杯开幕前",
                "1-3个交易日",
                "卖出剩余仓位的",
                "50%",
                "，避免利好兑现后的主题退潮。",
                "最终复盘点",
                "开幕后 · 归因分析",
                "开幕日不新增仓位，只记录模拟盘表现。",
                "双重止损防线",
                "风险控制 · 底线思维",
                "个股止损：",
                "固定止损与20日波动率倍数取较宽者，上限12%。",
                "组合止损：",
                "组合回撤达到10%或事件热度明显退潮时整体降仓。",
                "💡 配置逻辑：事件适配度 + 动态止损 + 仓位约束 = 避免伪事件和过早入场",
                "PAGE 04",
            ],
            "note": "模拟盘不是只记录买入，还要在结项时复盘配置效果。我们会跟踪每只股票贡献、成本、回撤和事件热度变化，分析配置原因是否成立。",
        },
        {
            "src": 16,
            "texts": [
                "中期结论",
                "总结与展望",
                "▍核心洞察总结",
                "1.",
                "盈利本质是“预期差”：体育事件价值在于赛前预期扩散。",
                "2.",
                "系统化是关键：事件、标的、仓位和风控由规则自动生成。",
                "3.",
                "事件适配度决定是否交易：世界杯和欧洲杯适配度更高。",
                "4.",
                "风控是收益前提：动态止损与仓位约束优于机械固定止损。",
                "▍下一步计划与展望",
                "持续跟踪2026世界杯模拟盘，记录实际成交、持仓收益、成本、回撤和事件热度变化。",
                "",
                "结项汇报将重点复盘：为何这些标的赚钱或亏钱，因子评分是否解释了收益，持仓配置是否需要调整。",
                "",
                "同时扩充欧洲杯、奥运会、亚运会、赛程公布、赞助官宣等事件，提高策略频率和稳健性。",
            ],
            "note": "这是单独的总结与展望内容页，不放在过渡页下方。我们总结中期已经完成的系统和实验，也说明结项时如何复盘模拟盘。",
        },
        {
            "src": 17,
            "texts": ["体育大事件事件驱动投资策略与系统", "答辩人：", "指导教师：", "软件与微电子学院", "感谢各位老师批评与指导"],
            "note": "谢谢各位老师的批评和指导。最后一页附上本项目使用的主要参考文献和规则依据。",
        },
        {
            "src": 16,
            "texts": [
                "参考文献",
                "主要参考依据",
                "▍金融与策略文献",
                "1.",
                "MacKinlay (1997)：事件研究方法与AR/CAR框架。",
                "2.",
                "Edmans, García & Norli (2007)：体育情绪与股票收益。",
                "3.",
                "Kaminski & Lo (2014)：止损规则有效性分析。",
                "4.",
                "Moskowitz, Ooi & Pedersen (2012)：时间序列动量。",
                "▍系统与规则依据",
                "1.",
                "Moreira & Muir (2017)：波动率管理组合。",
                "2.",
                "ReAct (ICLR 2023)：Agent推理与行动闭环。",
                "3.",
                "FIFA 2026官方赛程；上海证券交易所交易机制；2026-05-26行情快照。",
            ],
            "note": "本页为参考文献页，放在全篇最后。前面的实证分析主要基于事件研究法、体育情绪研究、止损文献、动量文献和真实交易规则。",
        },
    ]
    cleaned_slides = []
    skip_until_simulation_detail = False
    for spec in slides:
        first_text = spec.get("texts", [""])[0] if spec.get("texts") else ""
        if skip_until_simulation_detail:
            if spec.get("src") == 13 and first_text == "PART 04":
                skip_until_simulation_detail = False
                cleaned_slides.append(spec)
            continue
        cleaned_slides.append(spec)
        if spec.get("src") == 12 and first_text == "PART 04":
            skip_until_simulation_detail = True
    slides = cleaned_slides

    for i, s in enumerate(slides, start=1):
        notes[i] = s["note"]
    return slides, notes


def build() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, orders, trades, wf, car, diag, hyper, scores = load_metrics()
    charts = generate_charts(summary, trades, wf, car, diag)
    slides, notes = slide_plan(summary, orders, trades, wf, car, diag, hyper, scores)

    chart_media = {f"chart_{key}.png": path.read_bytes() for key, path in charts.items()}
    chart_targets = {key: f"chart_{key}.png" for key in charts}

    with zipfile.ZipFile(TEMPLATE, "r") as zin:
        original_names = set(zin.namelist())
        pres_rels_xml, pres_slide_rids = update_presentation_rels(zin.read("ppt/_rels/presentation.xml.rels"), len(slides))
        pres_xml = update_presentation_xml(zin.read("ppt/presentation.xml"), len(slides), pres_slide_rids)
        content_types_xml = update_content_types(zin.read("[Content_Types].xml"), len(slides))
        app_xml = update_app_xml(zin.read("docProps/app.xml"), len(slides)) if "docProps/app.xml" in original_names else None

        skip_patterns = [
            r"ppt/slides/slide\d+\.xml$",
            r"ppt/slides/_rels/slide\d+\.xml\.rels$",
            r"ppt/notesSlides/notesSlide\d+\.xml$",
            r"ppt/notesSlides/_rels/notesSlide\d+\.xml\.rels$",
        ]

        def should_skip(name: str) -> bool:
            if name in {"ppt/presentation.xml", "ppt/_rels/presentation.xml.rels", "[Content_Types].xml", "docProps/app.xml"}:
                return True
            return any(re.match(p, name) for p in skip_patterns)

        tmp = OUT.with_suffix(".tmp.pptx")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if should_skip(item.filename):
                    continue
                zout.writestr(item, zin.read(item.filename))

            zout.writestr("ppt/presentation.xml", pres_xml)
            zout.writestr("ppt/_rels/presentation.xml.rels", pres_rels_xml)
            zout.writestr("[Content_Types].xml", content_types_xml)
            if app_xml is not None:
                zout.writestr("docProps/app.xml", app_xml)

            for name, data in chart_media.items():
                zout.writestr(f"ppt/media/{name}", data)

            for out_no, spec in enumerate(slides, start=1):
                src = spec["src"]
                slide_xml = zin.read(f"ppt/slides/slide{src}.xml").decode("utf-8")
                text_count = len(re.findall(r"<a:t>.*?</a:t>", slide_xml, flags=re.DOTALL))
                slide_xml = replace_slide_text(slide_xml, pad(spec["texts"], text_count))

                images = spec.get("images", [])
                image_targets = [chart_targets[img["key"]] for img in images]
                src_rels_name = f"ppt/slides/_rels/slide{src}.xml.rels"
                src_rels_xml = zin.read(src_rels_name).decode("utf-8") if src_rels_name in original_names else f'<Relationships xmlns="{REL_NS}"/>'
                slide_rels, image_rids = rels_with_notes_and_images(src_rels_xml, out_no, image_targets)
                for img, rid in zip(images, image_rids):
                    slide_xml = add_picture(slide_xml, rid, chart_targets[img["key"]], img["bbox"])

                zout.writestr(f"ppt/slides/slide{out_no}.xml", slide_xml.encode("utf-8"))
                zout.writestr(f"ppt/slides/_rels/slide{out_no}.xml.rels", slide_rels)
                zout.writestr(f"ppt/notesSlides/notesSlide{out_no}.xml", build_notes_slide_xml(notes[out_no]))
                zout.writestr(f"ppt/notesSlides/_rels/notesSlide{out_no}.xml.rels", build_notes_rels_xml(out_no))

        tmp.replace(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
