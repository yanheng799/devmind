#!/usr/bin/env python3
"""
方案一：按内容类型 + 逻辑单元切分
甘肃～浙江±800kV特高压直流输电工程线路工程 第6施工标段杆塔明细表

输出：
  - 6个说明文档片段 (DOC-S01 ~ DOC-S06)
  - 9个塔位明细表片段 (TBL-T01 ~ TBL-T09)
  - 每个片段含：Markdown文本 + JSON元数据
"""

import json
import re
import os
import pdfplumber
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
PDF_PATH = "E:\\03-汇能博友\\2.351-SA06911S-D0102 第6施工标段塔位明细表.pdf"
OUT_DIR = Path("/mnt/user-data/outputs/chunks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 图片提取设置
TABLE_IMAGE_DPI = 200                # 渲染分辨率
TABLE_IMAGE_DIR = OUT_DIR / "images" # 图片子目录
DRAW_TABLE_BBOXES = True             # 在图片上用彩色矩形标记检测到的表格区域

# 第1-7页左右分栏设置
SPLIT_PAGE_RANGE = (1, 7)

PROJECT_META = {
    "project":    "甘肃～浙江±800千伏特高压直流输电工程线路工程",
    "volume":     "第1卷第2册",
    "section":    "第6施工标段",
    "doc_id":     "351-SA06911S-D0102",
    "designer":   "福建永福电力设计股份有限公司",
    "split_date": datetime.now().strftime("%Y-%m-%d"),
}

# ─────────────────────────────────────────────
# 说明文档分片规则
# PDF页码(1-based) → 分片定义
# ─────────────────────────────────────────────
DOC_CHUNKS = [
    {
        "id":    "DOC-S01",
        "title": "工程概况：标段范围、气象条件、导地线型号",
        "pages": [3],          # PDF页3，说明第1页
        "sections": ["1.1", "1.2", "1.3"],
        "keywords": ["标段范围", "气象条件", "导线型号", "覆冰", "基本风速"],
        "retrieval_hint": "查询工程基础参数、设计气象条件、导地线型号规格时使用",
    },
    {
        "id":    "DOC-S02",
        "title": "绝缘配置：污秽区划分、绝缘子选型、串型代号表",
        "pages": [3, 4],       # 说明第1-2页（跨页）
        "sections": ["1.4", "1.5", "1.6"],
        "keywords": ["污秽区", "复合绝缘子", "盘型绝缘子", "串型编号", "V型悬垂串", "耐张串"],
        "retrieval_hint": "解读塔位明细表中绝缘子串代号时必须关联此片段",
        "is_global_reference": True,   # 标记为全局参考片段
    },
    {
        "id":    "DOC-S03",
        "title": "施工技术要求：带电作业间隙、不允许接头档、跨越原则",
        "pages": [4],
        "sections": ["1.7", "1.8", "1.9", "1.10"],
        "keywords": ["带电作业间隙", "不允许接头", "耐张线夹上扬", "倒置", "房屋拆迁"],
        "retrieval_hint": "查询施工约束条件、特殊处理要求时使用",
    },
    {
        "id":    "DOC-S04",
        "title": "环保与合规：生态敏感点、压覆矿产资源说明",
        "pages": [5, 6],
        "sections": ["1.11", "2.9", "2.10", "2.11", "2.12"],
        "keywords": ["秦岭", "湿地公园", "水产种质资源", "压覆矿", "生态红线", "林地调规"],
        "retrieval_hint": "查询环保合规风险、施工前置审批条件时使用",
    },
    {
        "id":    "DOC-S05",
        "title": "明细表填写说明：坐标系、高差定义、接地要求、防振说明",
        "pages": [5],
        "sections": ["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8"],
        "keywords": ["CGCS2000", "定位高差", "防振锤", "接地电阻", "间隔棒", "档距单位"],
        "retrieval_hint": "解读明细表字段含义、接地电阻标准时使用",
    },
    {
        "id":    "DOC-S06",
        "title": "塔型统计、强制条文执行情况、标准工艺执行情况",
        "pages": [6, 7, 8, 9],
        "sections": ["3", "4", "5", "6", "7"],
        "keywords": ["塔型统计", "强制条文", "十八项反措", "标准工艺", "OPGW"],
        "retrieval_hint": "查询塔型用量统计、合规执行情况时使用",
    },
]

# ─────────────────────────────────────────────
# 塔位明细表分片规则（按耐张段+冰区划分）
# ─────────────────────────────────────────────
TABLE_CHUNKS = [
    {
        "id":         "TBL-T01",
        "tower_from": "N1501",
        "tower_to":   "N1508",
        "pdf_pages":  [10],
        "ice_zone":   "20mm重冰区→15mm冰区",
        "conductor":  "JL1/G2A-1000/80 → JL1/G2A-1250/100",
        "groundwire": "JLB20A-150 + OPGW-150",
        "wind_speed": "27m/s",
        "tower_count": 8,
        "special_crossings": [],
        "no_joint_spans": [],
        "keywords": ["N1501", "N1508", "20mm冰区", "重冰区", "花椒地"],
    },
    {
        "id":         "TBL-T02",
        "tower_from": "N1508",
        "tower_to":   "N1520",
        "pdf_pages":  [10, 11],
        "ice_zone":   "15mm冰区",
        "conductor":  "JL1/G2A-1250/100",
        "groundwire": "JLB20A-150 + OPGW-150",
        "wind_speed": "27m/s",
        "tower_count": 12,
        "special_crossings": ["10kV曙阳线", "10kV石村线", "小峪河", "省道S219"],
        "no_joint_spans": [],
        "keywords": ["N1509", "N1519", "15mm冰区", "复合绝缘子"],
    },
    {
        "id":         "TBL-T03",
        "tower_from": "N1520",
        "tower_to":   "N1529",
        "pdf_pages":  [11],
        "ice_zone":   "20mm重冰区",
        "conductor":  "JL1/G2A-1000/80",
        "groundwire": "JLB20A-150 + OPGW-150",
        "wind_speed": "27m/s",
        "tower_count": 9,
        "special_crossings": [],
        "no_joint_spans": [],
        "keywords": ["N1520", "N1529", "20mm冰区", "盘型绝缘子"],
    },
    {
        "id":         "TBL-T04",
        "tower_from": "N1529",
        "tower_to":   "N1548",
        "pdf_pages":  [11, 12],
        "ice_zone":   "15mm冰区",
        "conductor":  "JL1/G2A-1250/100",
        "groundwire": "JLB20A-150 + OPGW-150（三跨段N1531-N1533用2根OPGW-150）",
        "wind_speed": "27m/s",
        "tower_count": 19,
        "special_crossings": [
            "±500kV德宝直流（N1531-N1532，三跨）",
            "110kV马向Ⅰ线、马向Ⅱ线（N1532-N1533，三跨）",
            "330kV硖石-栖凤线、110kV栖黄Ⅰ/Ⅱ线（N1533-N1534，三跨）",
            "嘉陵江国家湿地公园（N1531-N1532）",
        ],
        "no_joint_spans": [
            "N1531-N1532（±500kV德宝直流）",
            "N1532-N1533（110kV线路）",
            "N1533-N1534（330kV/110kV线路）",
        ],
        "keywords": ["N1531", "N1533", "三跨", "德宝直流", "嘉陵江", "湿地公园"],
        "risk_level": "高",
    },
    {
        "id":         "TBL-T05",
        "tower_from": "N1548",
        "tower_to":   "N1553",
        "pdf_pages":  [12],
        "ice_zone":   "10mm冰区",
        "conductor":  "JL1/G2A-1250/100",
        "groundwire": "2根OPGW-150（N1548-N1549三跨段）",
        "wind_speed": "27m/s",
        "tower_count": 5,
        "special_crossings": [
            "S28太凤高速（N1548-N1549，三跨）",
            "G342国道（N1548-N1549）",
            "35kV向青线、35kV向河Ⅰ/Ⅱ线",
            "安河",
        ],
        "no_joint_spans": ["N1548-N1549（太凤高速、国道）"],
        "keywords": ["N1548", "N1549", "10mm冰区", "太凤高速", "三跨"],
        "risk_level": "高",
    },
    {
        "id":         "TBL-T06",
        "tower_from": "N1553",
        "tower_to":   "N1555",
        "pdf_pages":  [12],
        "ice_zone":   "15mm冰区",
        "conductor":  "JL1/G2A-1250/100",
        "groundwire": "JLB20A-150 + OPGW-150",
        "wind_speed": "27m/s",
        "tower_count": 2,
        "special_crossings": [],
        "no_joint_spans": [],
        "keywords": ["N1553", "N1554", "15mm冰区"],
    },
    {
        "id":         "TBL-T07",
        "tower_from": "N1555",
        "tower_to":   "N1574",
        "pdf_pages":  [12, 13],
        "ice_zone":   "20mm重冰区",
        "conductor":  "JL1/G2A-1000/80",
        "groundwire": "JLB20A-150 + OPGW-150",
        "wind_speed": "27m/s",
        "tower_count": 19,
        "special_crossings": [
            "拟建750kV宝鸡-汉中Ⅰ线（N1558-N1559，禁止接头）",
            "拟建750kV宝鸡-汉中Ⅱ线（N1568-N1569，禁止接头）",
            "35kV凤州线（N1555-N1556）",
        ],
        "no_joint_spans": [
            "N1558-N1559（拟建750kV宝汉Ⅰ线）",
            "N1568-N1569（拟建750kV宝汉Ⅱ线）",
        ],
        "keywords": ["N1555", "N1574", "20mm冰区", "750kV宝汉线", "拟建线路"],
        "risk_level": "中",
    },
    {
        "id":         "TBL-T08",
        "tower_from": "N1574",
        "tower_to":   "N1618",
        "pdf_pages":  [13, 14],
        "ice_zone":   "15mm冰区",
        "conductor":  "JL1/G2A-1250/100",
        "groundwire": "JLB20A-150 + OPGW-150",
        "wind_speed": "27m/s",
        "tower_count": 44,
        "special_crossings": [
            "旺峪河（N1574）",
            "嵩坝河（N1604）",
            "X301县道（N1604）",
            "输气管道（N1604）",
            "通信线多处",
        ],
        "no_joint_spans": [],
        "keywords": ["N1574", "N1618", "15mm冰区", "秦岭核心区", "嵩坝河", "通信线"],
        "note": "N1593、N1594、N1595位于秦岭核心保护区，N1595需Ⅰ级林地调规",
    },
    {
        "id":         "TBL-T09",
        "tower_from": "N1618",
        "tower_to":   "N1625",
        "pdf_pages":  [14],
        "ice_zone":   "20mm重冰区",
        "conductor":  "JL1/G2A-1000/80",
        "groundwire": "JLB20A-150 + OPGW-150",
        "wind_speed": "27m/s",
        "tower_count": 7,
        "special_crossings": [
            "S221省道（N1623-N1624，禁止接头）",
            "35kV留林线（N1623-N1624）",
            "架空国防光缆（N1623-N1624）",
        ],
        "no_joint_spans": ["N1623-N1624（S221省道）"],
        "keywords": ["N1618", "N1625", "20mm冰区", "S221省道", "国防光缆"],
        "risk_level": "中",
    },
]


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def extract_pages_text(pdf, page_indices):
    """提取指定页码（1-based）的文本，拼接返回"""
    texts = []
    for pg_num in page_indices:
        page = pdf.pages[pg_num - 1]
        text = page.extract_text() or ""
        texts.append(f"[第{pg_num}页]\n{text}")
    return "\n\n".join(texts)


def extract_table_rows(pdf, page_indices):
    """从指定页提取表格行（用于塔位明细表）"""
    all_rows = []
    for pg_num in page_indices:
        page = pdf.pages[pg_num - 1]
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                cleaned = [cell.strip().replace("\n", " ") if cell else "" for cell in row]
                if any(cleaned):
                    all_rows.append(cleaned)
    return all_rows


def rows_to_markdown(rows):
    """将表格行转为Markdown格式"""
    if not rows:
        return ""
    lines = []
    # 找最大列数
    max_cols = max(len(r) for r in rows)
    for i, row in enumerate(rows):
        # 补齐列数
        padded = row + [""] * (max_cols - len(row))
        lines.append("| " + " | ".join(padded) + " |")
        if i == 0:
            lines.append("|" + "|".join(["---"] * max_cols) + "|")
    return "\n".join(lines)


def save_table_page_images(pdf, page_indices, out_dir,
                           resolution=TABLE_IMAGE_DPI, draw_bboxes=DRAW_TABLE_BBOXES):
    """渲染表格页面为 PNG 图片，每页一张。返回相对路径列表（相对于 out_dir）。"""
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    bbox_colors = ["red", "blue", "green", "orange", "purple", "cyan"]
    saved_paths = []
    for pg_num in page_indices:
        img_filename = f"page_{pg_num:03d}.png"
        img_path = image_dir / img_filename

        # 多个 chunk 可能共享同一页，已渲染则跳过
        if img_path.exists():
            saved_paths.append(str(img_path.relative_to(out_dir)))
            continue

        page = pdf.pages[pg_num - 1]
        img = page.to_image(resolution=resolution)

        if draw_bboxes:
            tables = page.find_tables()
            for i, table in enumerate(tables):
                img.draw_rect(table.bbox, stroke=bbox_colors[i % len(bbox_colors)], stroke_width=1)

        img.save(img_path, format="PNG", quantize=False)
        saved_paths.append(str(img_path.relative_to(out_dir)))

    return saved_paths


def extract_split_page(pdf, page_num, out_dir, resolution=TABLE_IMAGE_DPI):
    """将左右分栏页面切成两半，分别提取文本并保存图片。

    返回 {"left_text": str, "right_text": str, "images": [相对路径列表]}
    """
    page = pdf.pages[page_num - 1]
    mid_x = page.width / 2

    halves = [
        ("left",  (0, 0, mid_x, page.height)),
        ("right", (mid_x, 0, page.width, page.height)),
    ]

    result = {"left_text": "", "right_text": "", "images": []}
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for side, bbox in halves:
        cropped = page.crop(bbox)
        text = cropped.extract_text() or ""
        result[f"{side}_text"] = text

        img_filename = f"page_{page_num:03d}_{side}.png"
        img_path = image_dir / img_filename
        if not img_path.exists():
            img = cropped.to_image(resolution=resolution)
            img.save(img_path, format="PNG", quantize=False)
        result["images"].append(str(img_path.relative_to(out_dir)))

    return result


def write_chunk(chunk_id, metadata, content, out_dir):
    """写出 .md 文本 + .json 元数据"""
    base = out_dir / chunk_id

    # 写 Markdown
    md_path = base.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        # 在文件头嵌入元数据摘要
        f.write(f"# {chunk_id} — {metadata.get('title', '')}\n\n")
        f.write("## 元数据\n\n```json\n")
        f.write(json.dumps(metadata, ensure_ascii=False, indent=2))
        f.write("\n```\n\n---\n\n")
        f.write("## 内容\n\n")
        f.write(content)

    # 写 JSON（元数据 + 内容，方便向量化）
    json_path = base.with_suffix(".json")
    payload = {
        **metadata,
        "content": content,
        "char_count": len(content),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return md_path, json_path


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    print(f"📂 打开PDF: {PDF_PATH}")
    results = []

    with pdfplumber.open(PDF_PATH) as pdf:
        total_pages = len(pdf.pages)
        print(f"   总页数: {total_pages}")

        # ── 第一步：切分说明文档 ──────────────────
        print("\n── 处理说明文档分片 ──────────────────────")
        for chunk_def in DOC_CHUNKS:
            cid = chunk_def["id"]
            pages = chunk_def["pages"]
            all_images = []
            text_parts = []

            for pg_num in pages:
                if SPLIT_PAGE_RANGE[0] <= pg_num <= SPLIT_PAGE_RANGE[1]:
                    sp = extract_split_page(pdf, pg_num, OUT_DIR)
                    text_parts.append(f"[第{pg_num}页-左]\n{sp['left_text']}")
                    text_parts.append(f"[第{pg_num}页-右]\n{sp['right_text']}")
                    all_images.extend(sp["images"])
                else:
                    page = pdf.pages[pg_num - 1]
                    text = page.extract_text() or ""
                    text_parts.append(f"[第{pg_num}页]\n{text}")

            text = "\n\n".join(text_parts)

            metadata = {
                **PROJECT_META,
                "chunk_id":          cid,
                "chunk_type":        "说明文档",
                "title":             chunk_def["title"],
                "source_pages":      pages,
                "sections":          chunk_def["sections"],
                "keywords":          chunk_def["keywords"],
                "retrieval_hint":    chunk_def["retrieval_hint"],
            }
            if all_images:
                metadata["split_page_images"] = all_images
            if chunk_def.get("is_global_reference"):
                metadata["is_global_reference"] = True

            md_path, json_path = write_chunk(cid, metadata, text, OUT_DIR)
            char_count = len(text)
            print(f"  ✅ {cid} | 页:{pages} | {char_count}字符 → {md_path.name}")
            results.append({"id": cid, "type": "说明文档", "chars": char_count, "pages": pages})

        # ── 第二步：切分塔位明细表 ────────────────
        print("\n── 处理塔位明细表分片 ────────────────────")
        for chunk_def in TABLE_CHUNKS:
            cid = chunk_def["id"]
            pages = chunk_def["pdf_pages"]

            # 先提取原始文本（用于全文搜索）
            raw_text = extract_pages_text(pdf, pages)

            # 再提取结构化表格行（转Markdown表格）
            rows = extract_table_rows(pdf, pages)
            table_md = rows_to_markdown(rows)

            # 保存表格页面图片作为视觉参考
            image_paths = save_table_page_images(pdf, pages, OUT_DIR)

            # 合并内容：先文本段，再Markdown表格
            # （过滤掉表格行中可能重复的表头）
            content = f"### 原始文本\n\n{raw_text}\n\n### 结构化表格\n\n{table_md}"

            metadata = {
                **PROJECT_META,
                "chunk_id":           cid,
                "chunk_type":         "塔位明细表",
                "tower_range":        f"{chunk_def['tower_from']}–{chunk_def['tower_to']}",
                "tower_from":         chunk_def["tower_from"],
                "tower_to":           chunk_def["tower_to"],
                "tower_count":        chunk_def["tower_count"],
                "source_pages":       pages,
                "ice_zone":           chunk_def["ice_zone"],
                "conductor_type":     chunk_def["conductor"],
                "groundwire_type":    chunk_def["groundwire"],
                "wind_speed":         chunk_def["wind_speed"],
                "special_crossings":  chunk_def["special_crossings"],
                "no_joint_spans":     chunk_def["no_joint_spans"],
                "keywords":           chunk_def["keywords"],
                "related_chunks":     ["DOC-S02", "DOC-S05"],  # 必须关联的说明文档
                "table_images":       image_paths,
            }
            if chunk_def.get("risk_level"):
                metadata["risk_level"] = chunk_def["risk_level"]
            if chunk_def.get("note"):
                metadata["note"] = chunk_def["note"]

            md_path, json_path = write_chunk(cid, metadata, content, OUT_DIR)
            char_count = len(content)
            tower_range = f"{chunk_def['tower_from']}–{chunk_def['tower_to']}"
            print(f"  ✅ {cid} | {tower_range} | {chunk_def['tower_count']}基 | 页:{pages} | {char_count}字符")
            results.append({"id": cid, "type": "塔位明细表", "chars": char_count,
                            "towers": tower_range, "pages": pages})

    # ── 第三步：写索引文件 ──────────────────────
    index = {
        "document":    PROJECT_META,
        "total_chunks": len(results),
        "doc_chunks":  [r for r in results if r["type"] == "说明文档"],
        "table_chunks": [r for r in results if r["type"] == "塔位明细表"],
        "global_references": ["DOC-S02"],
        "table_images_dir": "images/",
        "table_image_pages": sorted(set(
            pg for chunk in TABLE_CHUNKS for pg in chunk["pdf_pages"]
        )),
        "retrieval_notes": [
            "解读TBL-T*表格中的绝缘子串代号时，需同时加载DOC-S02",
            "查询接地电阻标准时，加载DOC-S05",
            "三跨特殊段（TBL-T04、TBL-T05）risk_level=高，需优先关注",
            "秦岭核心区塔位（N1593/N1594/N1595）在TBL-T08，有环保前置条件",
        ],
    }
    index_path = OUT_DIR / "00_INDEX.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\n📋 索引文件: {index_path}")
    print(f"✅ 切分完成，共 {len(results)} 个片段，输出目录: {OUT_DIR}")
    return results


if __name__ == "__main__":
    main()
