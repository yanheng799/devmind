import pdfplumber
import re
from typing import List, Dict, Optional, Tuple


class DescriptionPageProcessor:
    # 章节标题的正则（匹配 "1 概况"、"1.1 概述"、"13.2 条文" 等格式）
    SECTION_PATTERN = re.compile(
        r'^(\d+(?:\.\d+)*)\s+(.+)$',
        re.MULTILINE
    )

    def process(
            self,
            page,  # pdfplumber page对象
            page_num: int,
            project: str,
            section: str,
            embedding_fn,
    ) -> List[Dict]:
        """
        处理单张说明页，返回chunk列表
        """
        chunks = []

        # 1. 提取页面所有内容块（文字块 + 表格块，按y坐标排序）
        content_blocks = self._extract_content_blocks(page)

        # 2. 按章节标题分组
        sections = self._group_by_section(content_blocks)

        # 3. 每个章节生成chunk
        for sec in sections:
            sec_chunks = self._build_chunks(
                sec, page_num, project, section, embedding_fn
            )
            chunks.extend(sec_chunks)

        return chunks

    # ──────────────────────────────────────────────────
    # Step 1: 提取内容块
    # ──────────────────────────────────────────────────

    def _extract_content_blocks(self, page) -> List[Dict]:
        """
        将页面内容解析为有序的块列表
        每个块包含：类型(text/table)、内容、y坐标位置

        难点：pdfplumber的words和tables坐标系相同，
              可以按top值排序，还原阅读顺序
        """
        blocks = []

        # ── 获取表格区域（先占位，避免文字重复提取）──
        tables = page.find_tables({
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
        })

        table_bboxes = []
        for t_idx, table in enumerate(tables):
            x0, top, x1, bottom = table.bbox
            table_bboxes.append((x0, top, x1, bottom))

            # 提取表格文字（结构化）
            rows = table.extract()
            table_text = self._table_to_text(rows)

            blocks.append({
                "type": "table",
                "content": table_text,
                "raw_rows": rows,
                "top": top,
                "bottom": bottom,
                "bbox": (x0, top, x1, bottom),
                "table_idx": t_idx,
            })

        # ── 获取文字块（排除表格区域内的文字）──
        # 按行聚合：将同一行的words合并成一行文字
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
        )

        # 过滤掉落在表格bbox内的word
        def in_any_table(word) -> bool:
            for (x0, top, x1, bottom) in table_bboxes:
                if (word["x0"] >= x0 - 2 and word["x1"] <= x1 + 2 and
                        word["top"] >= top - 2 and word["bottom"] <= bottom + 2):
                    return True
            return False

        text_words = [w for w in words if not in_any_table(w)]

        # 按y坐标聚合成行
        lines = self._words_to_lines(text_words, y_tolerance=3)

        # 把连续的文字行合并成段落块
        para_blocks = self._lines_to_paragraphs(lines)
        blocks.extend(para_blocks)

        # 按top坐标排序，还原页面阅读顺序
        blocks.sort(key=lambda b: b["top"])

        return blocks

    def _words_to_lines(self, words: List[Dict], y_tolerance: float = 3) -> List[Dict]:
        """将word列表按y坐标聚合成行"""
        if not words:
            return []

        lines = []
        current_line_words = [words[0]]
        current_y = words[0]["top"]

        for word in words[1:]:
            if abs(word["top"] - current_y) <= y_tolerance:
                current_line_words.append(word)
            else:
                # 同行word按x坐标排序后拼接
                current_line_words.sort(key=lambda w: w["x0"])
                line_text = " ".join(w["text"] for w in current_line_words)
                lines.append({
                    "text": line_text,
                    "top": current_line_words[0]["top"],
                    "bottom": current_line_words[0]["bottom"],
                    "x0": current_line_words[0]["x0"],
                })
                current_line_words = [word]
                current_y = word["top"]

        # 最后一行
        if current_line_words:
            current_line_words.sort(key=lambda w: w["x0"])
            lines.append({
                "text": " ".join(w["text"] for w in current_line_words),
                "top": current_line_words[0]["top"],
                "bottom": current_line_words[0]["bottom"],
                "x0": current_line_words[0]["x0"],
            })

        return lines

    def _lines_to_paragraphs(self, lines: List[Dict]) -> List[Dict]:
        """
        将行列表合并成段落块
        判断段落边界的依据：
        - 行间距明显大于正常行距（新段落）
        - 遇到章节标题行（新段落）
        """
        if not lines:
            return []

        paragraphs = []
        current_lines = [lines[0]]

        # 估算正常行距（取前几行的平均间距）
        sample_gaps = []
        for i in range(1, min(5, len(lines))):
            gap = lines[i]["top"] - lines[i - 1]["bottom"]
            if 0 < gap < 20:
                sample_gaps.append(gap)
        normal_gap = sum(sample_gaps) / len(sample_gaps) if sample_gaps else 5

        for i in range(1, len(lines)):
            line = lines[i]
            prev = lines[i - 1]
            gap = line["top"] - prev["bottom"]
            is_new_section = bool(self.SECTION_PATTERN.match(line["text"].strip()))

            # 段落切分条件
            should_break = (
                    gap > normal_gap * 2.5  # 行距明显变大
                    or is_new_section  # 新章节标题
            )

            if should_break and current_lines:
                paragraphs.append(self._lines_to_block(current_lines))
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            paragraphs.append(self._lines_to_block(current_lines))

        return paragraphs

    def _lines_to_block(self, lines: List[Dict]) -> Dict:
        text = "\n".join(l["text"] for l in lines)
        return {
            "type": "text",
            "content": text,
            "top": lines[0]["top"],
            "bottom": lines[-1]["bottom"],
        }

    # ──────────────────────────────────────────────────
    # Step 2: 按章节标题分组
    # ──────────────────────────────────────────────────

    def _group_by_section(self, blocks: List[Dict]) -> List[Dict]:
        """
        将内容块按章节标题分组
        输出结构：
        [
          {
            "section_no":    "1.2",
            "section_title": "主要设计气象条件",
            "depth":         2,       # 标题层级
            "blocks":        [...],   # 该章节下的内容块
          },
          ...
        ]
        """
        sections = []
        current_section = {
            "section_no": "",
            "section_title": "（页眉/前言）",
            "depth": 0,
            "blocks": [],
        }

        for block in blocks:
            if block["type"] != "text":
                current_section["blocks"].append(block)
                continue

            # 检查每一行是否是章节标题
            lines = block["content"].split("\n")
            non_title_lines = []

            for line in lines:
                line = line.strip()
                m = self.SECTION_PATTERN.match(line)

                if m:
                    # 先保存之前积累的非标题内容
                    if non_title_lines:
                        current_section["blocks"].append({
                            "type": "text",
                            "content": "\n".join(non_title_lines),
                            "top": block["top"],
                            "bottom": block["bottom"],
                        })
                        non_title_lines = []

                    # 保存上一个章节
                    if current_section["blocks"] or current_section["section_no"]:
                        sections.append(current_section)

                    sec_no = m.group(1)
                    sec_title = m.group(2).strip()
                    depth = sec_no.count(".") + 1

                    current_section = {
                        "section_no": sec_no,
                        "section_title": sec_title,
                        "depth": depth,
                        "blocks": [],
                    }
                else:
                    if line:
                        non_title_lines.append(line)

            # 剩余非标题行
            if non_title_lines:
                current_section["blocks"].append({
                    "type": "text",
                    "content": "\n".join(non_title_lines),
                    "top": block["top"],
                    "bottom": block["bottom"],
                })

        if current_section["blocks"] or current_section["section_no"]:
            sections.append(current_section)

        return sections

    # ──────────────────────────────────────────────────
    # Step 3: 构建chunk
    # ──────────────────────────────────────────────────

    def _build_chunks(
            self,
            sec: Dict,
            page_num: int,
            project: str,
            section: str,
            embedding_fn,
    ) -> List[Dict]:
        """
        一个章节生成一个或多个chunk

        分块策略：
        - 纯文字章节：直接作为一个chunk
        - 章节内含表格：文字+表格各自独立成chunk，但metadata里互相关联
        - 内容过长（>800字）：按段落切分成多个chunk，用overlap连接
        """
        chunks = []
        sec_header = f"{sec['section_no']} {sec['section_title']}"

        text_parts = []
        table_parts = []

        for block in sec["blocks"]:
            if block["type"] == "text":
                text_parts.append(block["content"])
            else:
                table_parts.append(block)

        full_text = "\n".join(text_parts).strip()

        # ── 情况A：只有文字，直接一个chunk ──
        if not table_parts:
            if full_text:
                chunks.append(self._make_text_chunk(
                    text=f"{sec_header}\n{full_text}",
                    section_no=sec["section_no"],
                    section_title=sec["section_title"],
                    page_num=page_num,
                    project=project,
                    section=section,
                    embedding_fn=embedding_fn,
                ))
            return chunks

        # ── 情况B：有表格，文字chunk + 表格chunk 分开 ──

        # 文字部分
        if full_text:
            # 长文字切分
            text_chunks = self._split_long_text(
                text=f"{sec_header}\n{full_text}",
                max_chars=600,
                overlap_chars=80,
            )
            for t in text_chunks:
                chunks.append(self._make_text_chunk(
                    text=t,
                    section_no=sec["section_no"],
                    section_title=sec["section_title"],
                    page_num=page_num,
                    project=project,
                    section=section,
                    embedding_fn=embedding_fn,
                    has_related_table=True,
                ))

        # 表格部分：每张表格一个chunk
        for t_block in table_parts:
            table_text = f"{sec_header}（表格）\n{t_block['content']}"
            chunks.append(self._make_table_chunk(
                text=table_text,
                raw_rows=t_block.get("raw_rows", []),
                section_no=sec["section_no"],
                section_title=sec["section_title"],
                page_num=page_num,
                project=project,
                section=section,
                embedding_fn=embedding_fn,
            ))

        return chunks

    def _split_long_text(
            self, text: str, max_chars: int, overlap_chars: int
    ) -> List[str]:
        """
        超长文字按段落切分，相邻chunk有overlap避免语义断裂
        """
        if len(text) <= max_chars:
            return [text]

        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks = []
        current = []
        current_len = 0

        for para in paragraphs:
            if current_len + len(para) > max_chars and current:
                chunk_text = "\n".join(current)
                chunks.append(chunk_text)

                # overlap：保留最后几段作为下一chunk的开头
                overlap_text = ""
                overlap_buf = []
                for p in reversed(current):
                    if len(overlap_text) + len(p) <= overlap_chars:
                        overlap_buf.insert(0, p)
                        overlap_text += p
                    else:
                        break
                current = overlap_buf
                current_len = len(overlap_text)

            current.append(para)
            current_len += len(para)

        if current:
            chunks.append("\n".join(current))

        return chunks

    # ──────────────────────────────────────────────────
    # chunk构造工具
    # ──────────────────────────────────────────────────

    def _make_text_chunk(
            self, text, section_no, section_title,
            page_num, project, section, embedding_fn,
            has_related_table=False
    ) -> Dict:
        return {
            "chunk_id": f"{project}_{section}_desc_{section_no}_p{page_num}",
            "embedding": embedding_fn(text),
            "text": text,
            "chunk_type": "description",
            "project": project,
            "section": section,
            "tower_no": "",
            "tower_type": "",
            "is_tension": False,
            "cumul_dist": 0.0,
            "elevation": 0.0,
            "ice_zone": 0,
            "ground_resist": 0,
            "no_joint": False,
            "crossing_types": [],
            "image_ids": [],
            # 说明页特有字段（用dynamic field存）
            "section_no": section_no,
            "section_title": section_title,
            "page_num": page_num,
            "has_related_table": has_related_table,
        }

    def _make_table_chunk(
            self, text, raw_rows, section_no, section_title,
            page_num, project, section, embedding_fn
    ) -> Dict:
        return {
            "chunk_id": f"{project}_{section}_desc_{section_no}_table_p{page_num}",
            "embedding": embedding_fn(text),
            "text": text,
            "chunk_type": "description_table",
            "project": project,
            "section": section,
            "tower_no": "",
            "tower_type": "",
            "is_tension": False,
            "cumul_dist": 0.0,
            "elevation": 0.0,
            "ice_zone": 0,
            "ground_resist": 0,
            "no_joint": False,
            "crossing_types": [],
            "image_ids": [],
            "section_no": section_no,
            "section_title": section_title,
            "page_num": page_num,
            "raw_table_rows": str(raw_rows),  # 保留原始行便于调试
        }

    def _table_to_text(self, rows: List) -> str:
        """表格转文字，便于向量化"""
        lines = []
        for row in rows:
            if not row:
                continue
            cells = [str(c).strip() if c else "" for c in row]
            non_empty = [c for c in cells if c]
            if non_empty:
                lines.append(" | ".join(non_empty))
        return "\n".join(lines)