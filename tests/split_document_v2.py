import pdfplumber
import fitz  # pymupdf
from pathlib import Path
from PIL import Image
import hashlib
import json


def extract_table_images(pdf_path: str, output_dir: str, dpi: int = 200):
    """
    提取PDF中所有表格区域为图片
    策略：pdfplumber检测表格边界 → pymupdf高清截图
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []  # 每张表格图片的元数据

    # 用pdfplumber检测表格位置（它的表格检测比pymupdf准）
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):

            tables = page.find_tables({
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 3,
            })

            if not tables:
                continue

            # 同一页可能有多个表格，逐个裁剪
            for table_idx, table in enumerate(tables):
                bbox_plumber = table.bbox  # (x0, top, x1, bottom) pdfplumber坐标

                # 转成pymupdf坐标系（pdfplumber的top从上，pymupdf的y0从下）
                x0, top, x1, bottom = bbox_plumber
                page_height = page.height

                # 加padding让截图更好看
                padding = 8
                bbox_pdf = fitz.Rect(
                    x0 - padding,
                    top - padding,
                    x1 + padding,
                    bottom + padding
                )

                # 用pymupdf高清截图
                img = render_page_region(pdf_path, page_num, bbox_pdf, dpi=dpi)
                if img is None:
                    continue

                # 文件命名：页码_表格序号_内容hash
                content_hash = hashlib.md5(img.tobytes()).hexdigest()[:8]
                img_filename = f"page{page_num + 1:03d}_table{table_idx + 1}_{content_hash}.png"
                img_path = output_dir / img_filename
                img.save(img_path, "PNG", optimize=True)

                # 提取表格文字（用于建立text→图片的关联）
                table_text = extract_table_text_simple(table)

                results.append({
                    "image_path": str(img_path),
                    "image_filename": img_filename,
                    "page_num": page_num + 1,
                    "table_idx": table_idx + 1,
                    "bbox": list(bbox_plumber),
                    "table_text": table_text,  # 纯文本，用于向量化
                    "row_count": len(table.rows),
                    "col_count": len(table.columns),
                })

                print(f"✓ 第{page_num + 1}页 表格{table_idx + 1}: {img_filename}")

    return results


def render_page_region(pdf_path: str, page_num: int, bbox: fitz.Rect, dpi: int) -> Image:
    """用pymupdf渲染指定区域为高清图片"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # clip参数裁剪到指定区域，matrix控制分辨率
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    clip = bbox & page.rect  # 确保不超出页面边界

    if clip.is_empty:
        return None

    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    doc.close()

    # 转PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def extract_table_text_simple(table) -> str:
    """提取表格文字，用于向量化和关键词索引"""
    rows = table.extract()
    lines = []
    for row in rows:
        cells = [str(c).strip() if c else "" for c in row]
        non_empty = [c for c in cells if c]
        if non_empty:
            lines.append(" | ".join(non_empty))
    return "\n".join(lines)


def handle_wide_table(pdf_path: str, page_num: int, output_dir: str, dpi: int = 150):
    """
    处理超宽表格（塔位明细表跨整页）
    策略：整页截图 + 生成缩略图 + 按行切片
    """
    output_dir = Path(output_dir)
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # 整页高清截图
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    full_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    full_path = output_dir / f"page{page_num + 1:03d}_full.png"
    full_img.save(full_path, "PNG")

    # 生成缩略图（用于预览）
    thumb = full_img.copy()
    thumb.thumbnail((800, 600))
    thumb_path = output_dir / f"page{page_num + 1:03d}_thumb.jpg"
    thumb.save(thumb_path, "JPEG", quality=85)

    # 按行切片：每N行塔位截一张图（便于精确定位）
    row_images = slice_table_by_rows(full_img, page, page_num, output_dir)

    doc.close()
    return {
        "full_image": str(full_path),
        "thumbnail": str(thumb_path),
        "row_slices": row_images,
    }


def slice_table_by_rows(
        full_img: Image,
        page,  # pdfplumber page对象
        page_num: int,
        output_dir: Path,
        rows_per_slice: int = 5,  # 每片包含多少塔位行
        dpi: int = 150
):
    """
    将大表格按行切片，每片对应几个塔位
    用于"精确到某几基塔"的图片引用
    """
    with pdfplumber.open(page.pdf.stream) as pdf:
        plumber_page = pdf.pages[page_num]
        tables = plumber_page.find_tables()
        if not tables:
            return []

    table = tables[0]  # 取最大的表格
    rows = table.rows

    page_h = plumber_page.height
    page_w = plumber_page.width

    # 图片像素 vs PDF点的缩放比
    scale = dpi / 72

    slices = []
    for i in range(0, len(rows), rows_per_slice):
        batch = rows[i: i + rows_per_slice]

        # 获取这批行的y坐标范围
        y_top = batch[0].bbox[1]
        y_bottom = batch[-1].bbox[3]

        # 转像素坐标（注意pdfplumber y轴方向）
        px_top = int(y_top * scale) - 2
        px_bottom = int(y_bottom * scale) + 2
        px_top = max(0, px_top)
        px_bottom = min(full_img.height, px_bottom)

        # 裁剪行片段（保留全宽）
        row_img = full_img.crop((0, px_top, full_img.width, px_bottom))

        # 提取这几行的塔号范围（用于命名和索引）
        tower_nos = []
        for row in batch:
            cells = row.cells
            if cells and cells[0]:
                cell_text = plumber_page.crop(cells[0]).extract_text() or ""
                cell_text = cell_text.strip()
                if cell_text.startswith("N"):
                    tower_nos.append(cell_text.split()[0])

        if not tower_nos:
            continue

        slice_name = f"page{page_num + 1:03d}_rows_{tower_nos[0]}_{tower_nos[-1]}.png"
        slice_path = output_dir / slice_name
        row_img.save(slice_path, "PNG")

        slices.append({
            "image_path": str(slice_path),
            "tower_range": (tower_nos[0], tower_nos[-1]),
            "tower_list": tower_nos,
            "row_start": i,
            "row_end": i + len(batch) - 1,
        })

    return slices