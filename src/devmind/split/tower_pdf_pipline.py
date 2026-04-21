import pdfplumber
import fitz
from PIL import Image
import hashlib
import re
from typing import List, Dict, Tuple
import numpy as np
from pymilvus import Collection

from devmind.split.description_page_processor import DescriptionPageProcessor
from devmind.split.image_storage import ImageStorage


class TowerPDFPipeline:

    def __init__(self, minio: ImageStorage, embedding_fn):
        self.minio = minio
        self.embedding_fn = embedding_fn  # callable: str -> List[float]

        self.img_collection = Collection("table_images")
        self.chunk_collection = Collection("tower_chunks")

    def process_pdf(self, pdf_path: str, project: str, section: str):
        """处理一份PDF的完整流程"""
        print(f"\n处理: {pdf_path}")

        all_image_records = []
        all_chunk_records = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_type = self._classify_page(page)
                print(f"  第{page_num + 1}页: {page_type}")

                if page_type == "tower_table":
                    images, chunks = self._process_tower_table_page(
                        pdf_path, page, page_num, project, section
                    )
                    all_image_records.extend(images)
                    all_chunk_records.extend(chunks)

                elif page_type == "description":
                    chunks = self._process_description_page(
                        page, page_num, project, section
                    )
                    all_chunk_records.extend(chunks)

        # 批量写入Milvus
        if all_image_records:
            self._batch_insert_images(all_image_records)
        if all_chunk_records:
            self._batch_insert_chunks(all_chunk_records)

        print(f"  完成: {len(all_image_records)}张图片, {len(all_chunk_records)}个chunk")

    def _process_description_page(
            self,
            page,
            page_num: int,
            project: str,
            section: str,
    ) -> List[Dict]:

        processor = DescriptionPageProcessor()
        return processor.process(
            page, page_num, project, section, self.embedding_fn
        )
    # ──────────────────────────────────────────
    # 塔位明细表页处理
    # ──────────────────────────────────────────

    def _process_tower_table_page(
            self,
            pdf_path: str,
            page,
            page_num: int,
            project: str,
            section: str
    ) -> Tuple[List, List]:

        image_records = []
        chunk_records = []

        # 1. 整页截图上传
        full_url, thumb_url, obj_name = self._capture_and_upload_page(
            pdf_path, page_num, project, section
        )

        # 2. 解析表格数据
        tower_rows = self._parse_tower_rows(page)
        if not tower_rows:
            return [], []

        # 3. 按行切片上传（每5行一张）
        row_slice_map = self._slice_and_upload_rows(
            pdf_path, page, page_num, tower_rows, project, section
        )
        # row_slice_map: {(tower_from, tower_to): (url, thumb_url, obj_name)}

        # 4. 整页图片记录
        tower_nos = [r["tower_no"] for r in tower_rows if r.get("tower_no")]
        page_image_id = f"{project}_{section}_page{page_num + 1}_full"
        image_records.append({
            "image_id": page_image_id,
            "full_url": full_url,
            "thumb_url": thumb_url,
            "object_name": obj_name,
            "project": project,
            "section": section,
            "page_num": page_num + 1,
            "table_idx": 0,
            "table_type": "tower_detail",
            "tower_from": tower_nos[0] if tower_nos else "",
            "tower_to": tower_nos[-1] if tower_nos else "",
            "tower_list": tower_nos[:20],
            "img_width": 0,
            "img_height": 0,
            "dummy_vector": [0.0, 0.0, 0.0, 0.0],
        })

        # 5. 行切片图片记录 + 对应chunk
        for (t_from, t_to), (s_url, st_url, s_obj) in row_slice_map.items():
            slice_id = f"{project}_{section}_{t_from}_{t_to}_slice"
            image_records.append({
                "image_id": slice_id,
                "full_url": s_url,
                "thumb_url": st_url,
                "object_name": s_obj,
                "project": project,
                "section": section,
                "page_num": page_num + 1,
                "table_idx": 1,
                "table_type": "tower_detail",
                "tower_from": t_from,
                "tower_to": t_to,
                "tower_list": [],
                "img_width": 0,
                "img_height": 0,
                "dummy_vector": [0.0, 0.0, 0.0, 0.0],
            })

        # 6. 每个塔位生成chunk，关联图片ID
        for row in tower_rows:
            if not row.get("tower_no"):
                continue

            # 找到这个塔位对应的切片图片ID
            related_ids = [page_image_id]  # 整页图始终关联
            for (t_from, t_to) in row_slice_map:
                if self._tower_in_range(row["tower_no"], t_from, t_to):
                    related_ids.append(f"{project}_{section}_{t_from}_{t_to}_slice")

            text = self._format_tower_text(row)
            embedding = self.embedding_fn(text)

            chunk_records.append({
                "chunk_id": f"{project}_{section}_{row['tower_no']}",
                "embedding": embedding,
                "text": text,
                "chunk_type": "tower",
                "project": project,
                "section": section,
                "tower_no": row.get("tower_no", ""),
                "tower_type": row.get("tower_type", ""),
                "is_tension": row.get("is_tension", False),
                "cumul_dist": float(row.get("cumul_dist", 0) or 0),
                "elevation": float(row.get("elevation", 0) or 0),
                "ice_zone": int(row.get("ice_zone", 10) or 10),
                "ground_resist": int(row.get("ground_resist", 30) or 30),
                "no_joint": row.get("no_joint", False),
                "crossing_types": row.get("crossing_types", [])[:20],
                "image_ids": related_ids[:10],
            })

        return image_records, chunk_records

    def _capture_and_upload_page(
            self, pdf_path: str, page_num: int, project: str, section: str
    ) -> Tuple[str, str, str]:
        """整页截图并上传，返回(full_url, thumb_url, object_name)"""
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        mat = fitz.Matrix(200 / 72, 200 / 72)  # 200 dpi
        pix = page.get_pixmap(matrix=mat, alpha=False)
        full_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()

        # 缩略图
        thumb = full_img.copy()
        thumb.thumbnail((800, 1200))

        # 生成对象名
        obj_base = f"{project}/{section}/pages/page{page_num + 1:03d}"
        full_url = self.minio.upload_pil_image(full_img, f"{obj_base}_full.png")
        thumb_url = self.minio.upload_pil_image(thumb, f"{obj_base}_thumb.jpg")

        return full_url, thumb_url, f"{obj_base}_full.png"

    def _slice_and_upload_rows(
            self, pdf_path, page, page_num, tower_rows, project, section,
            rows_per_slice: int = 5
    ) -> Dict:
        """按行切片截图上传，返回 {(from, to): (url, thumb_url, obj_name)}"""

        doc = fitz.open(pdf_path)
        fitz_page = doc[page_num]
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pix = fitz_page.get_pixmap(matrix=mat, alpha=False)
        full_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()

        # 获取每行的y坐标
        tables = page.find_tables()
        if not tables:
            return {}

        table = tables[0]
        scale = 150 / 72
        page_h = page.height
        result = {}

        # 按rows_per_slice分批
        data_rows = [r for r in table.rows if r.bbox[1] > 50]  # 跳过表头

        for i in range(0, len(data_rows), rows_per_slice):
            batch = data_rows[i: i + rows_per_slice]
            if not batch:
                continue

            y_top = batch[0].bbox[1]
            y_bottom = batch[-1].bbox[3]

            # 裁剪
            px_top = max(0, int(y_top * scale) - 3)
            px_bottom = min(full_img.height, int(y_bottom * scale) + 3)
            slice_img = full_img.crop((0, px_top, full_img.width, px_bottom))

            # 找塔号范围
            towers_in_batch = tower_rows[i: i + rows_per_slice]
            nos = [r["tower_no"] for r in towers_in_batch if r.get("tower_no")]
            if not nos:
                continue

            t_from, t_to = nos[0], nos[-1]
            obj_name = f"{project}/{section}/slices/{t_from}_{t_to}.png"

            thumb = slice_img.copy()
            thumb.thumbnail((1200, 400))
            thumb_obj = f"{project}/{section}/slices/{t_from}_{t_to}_thumb.jpg"

            full_url = self.minio.upload_pil_image(slice_img, obj_name)
            thumb_url = self.minio.upload_pil_image(thumb, thumb_obj)

            result[(t_from, t_to)] = (full_url, thumb_url, obj_name)

        return result

    # ──────────────────────────────────────────
    # 批量写入Milvus
    # ──────────────────────────────────────────

    def _batch_insert_images(self, records: List[Dict], batch_size: int = 100):
        col = Collection("table_images")
        for i in range(0, len(records), batch_size):
            batch = records[i: i + batch_size]
            col.insert(batch)
        col.flush()
        print(f"  ✓ 写入 {len(records)} 条图片记录")

    MAX_TEXT_LENGTH = 8192

    def _batch_insert_chunks(self, records: List[Dict], batch_size: int = 100):
        col = Collection("tower_chunks")
        for i in range(0, len(records), batch_size):
            batch = records[i: i + batch_size]
            for record in batch:
                if len(record["text"]) > self.MAX_TEXT_LENGTH:
                    record["text"] = record["text"][:self.MAX_TEXT_LENGTH]
            col.insert(batch)
        col.flush()
        print(f"  ✓ 写入 {len(records)} 条chunk")

    # ──────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────

    def _classify_page(self, page) -> str:
        text = page.extract_text() or ""
        if "杆塔号" in text and "累距" in text:
            return "tower_table"
        elif any(k in text for k in ["说明", "概况", "气象条件", "强制性条文"]):
            return "description"
        return "other"

    def _parse_tower_rows(self, page) -> List[Dict]:
        """解析塔位行，返回结构化数据"""
        tables = page.find_tables()
        if not tables:
            return []

        rows = []
        raw = tables[0].extract()

        for row in raw:
            if not row or not row[0]:
                continue
            tower_no = str(row[0]).strip()
            if not re.match(r'^N\d+', tower_no):
                continue

            crossings = str(row[-2] or "").strip() if len(row) > 2 else ""
            crossing_types = self._parse_crossing_types(crossings)

            rows.append({
                "tower_no": tower_no,
                "site_no": str(row[1] or "").strip() if len(row) > 1 else "",
                "tower_type": str(row[2] or "").strip() if len(row) > 2 else "",
                "is_tension": str(row[1] or "").startswith("J"),
                "cumul_dist": self._safe_float(row[4] if len(row) > 4 else None),
                "elevation": self._safe_float(row[5] if len(row) > 5 else None),
                "ice_zone": self._extract_ice_zone(page),
                "ground_resist": self._safe_int(row[11] if len(row) > 11 else None),
                "no_joint": "不许" in str(row[-1] or ""),
                "crossing_types": crossing_types,
                "crossings_raw": crossings,
            })

        return rows

    def _parse_crossing_types(self, text: str) -> List[str]:
        types = []
        patterns = {
            "500kV直流": r"±500kV|±800kV",
            "高速公路": r"高速",
            "国道省道": r"G\d+国道|S\d+省道|县道",
            "河流": r"河|江|溪",
            "铁路": r"铁路|高铁",
            "110kV电力线": r"110kV",
            "330kV电力线": r"330kV",
            "750kV电力线": r"750kV",
            "10kV电力线": r"10kV",
            "通信线": r"通信线|通讯线",
            "天然气管道": r"管道|燃气",
        }
        for label, pattern in patterns.items():
            if re.search(pattern, text):
                types.append(label)
        return types

    def _format_tower_text(self, row: Dict) -> str:
        parts = [
            f"塔号：{row['tower_no']}",
            f"塔型：{row['tower_type']}",
        ]
        if row.get("cumul_dist"):
            parts.append(f"累距：{row['cumul_dist']}米")
        if row.get("elevation"):
            parts.append(f"高程：{row['elevation']}米")
        if row.get("ice_zone"):
            parts.append(f"冰区：{row['ice_zone']}mm")
        if row.get("ground_resist"):
            parts.append(f"接地电阻：≤{row['ground_resist']}Ω")
        if row.get("crossings_raw"):
            parts.append(f"交叉跨越：{row['crossings_raw']}")
        if row.get("no_joint"):
            parts.append("导线不许接头")
        return "，".join(parts)

    def _tower_in_range(self, tower_no: str, t_from: str, t_to: str) -> bool:
        """判断塔号是否在切片范围内"""

        def num(t):
            m = re.search(r'\d+', t)
            return int(m.group()) if m else 0

        return num(t_from) <= num(tower_no) <= num(t_to)

    def _safe_float(self, val) -> float:
        try:
            return float(str(val).strip())
        except:
            return 0.0

    def _safe_int(self, val) -> int:
        try:
            return int(str(val).strip())
        except:
            return 30

    def _extract_ice_zone(self, page) -> int:
        text = page.extract_text() or ""
        m = re.search(r'冰区[：:]\s*(\d+)mm', text)
        return int(m.group(1)) if m else 15