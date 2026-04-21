from typing import Dict

from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility, Index
)

from devmind.split.image_storage import ImageStorage
from devmind.split.tower_pdf_pipline import TowerPDFPipeline


def connect_milvus(host: str = "localhost", port: str = "19530"):
    connections.connect("default", host=host, port=port)
    print("Milvus connected")


def create_collections():
    """创建两个Collection：表格图片 + 塔位chunk"""

    _create_table_image_collection()
    _create_tower_chunk_collection()


def _create_table_image_collection():
    """
    表格图片Collection
    存图片URL和元数据，不存向量（图片通过chunk关联检索）
    """
    if utility.has_collection("table_images"):
        utility.drop_collection("table_images")

    fields = [
        FieldSchema(
            name="image_id",
            dtype=DataType.VARCHAR,
            max_length=128,
            is_primary=True
        ),
        # Milvus不存图片本体，存URL
        FieldSchema(name="full_url", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="thumb_url", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="object_name", dtype=DataType.VARCHAR, max_length=256),

        # 图片描述信息
        FieldSchema(name="project", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="page_num", dtype=DataType.INT16),
        FieldSchema(name="table_idx", dtype=DataType.INT16),
        FieldSchema(name="table_type", dtype=DataType.VARCHAR, max_length=32),
        # tower_detail / segment_summary / type_statistics / weather_table

        # 塔位范围（用于关联查询）
        FieldSchema(name="tower_from", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="tower_to", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(
            name="tower_list",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=20,
            max_length=16
        ),

        # 图片尺寸
        FieldSchema(name="img_width", dtype=DataType.INT32),
        FieldSchema(name="img_height", dtype=DataType.INT32),

        # Milvus Collection必须有向量字段
        # 这里存一个dummy向量（全0），实际不做向量检索
        FieldSchema(
            name="dummy_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=4
        ),
    ]

    schema = CollectionSchema(
        fields,
        description="tower table images",
        enable_dynamic_field=True  # 允许存额外字段
    )
    collection = Collection("table_images", schema)

    # dummy向量建个最简单的索引
    collection.create_index(
        "dummy_vector",
        {"index_type": "FLAT", "metric_type": "L2", "params": {}}
    )
    print("✓ table_images collection created")
    return collection


def _create_tower_chunk_collection():
    """
    塔位chunk Collection
    存向量 + 结构化元数据 + 关联图片ID列表
    """
    if utility.has_collection("tower_chunks"):
        utility.drop_collection("tower_chunks")

    fields = [
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            max_length=128,
            is_primary=True
        ),

        # 向量字段（text-embedding-v4 = 1024维）
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=1024
        ),

        # 文本内容
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),

        # 结构化字段
        FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=32),
        # tower / segment / description
        FieldSchema(name="project", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="tower_no", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="tower_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="is_tension", dtype=DataType.BOOL),
        FieldSchema(name="cumul_dist", dtype=DataType.FLOAT),
        FieldSchema(name="elevation", dtype=DataType.FLOAT),
        FieldSchema(name="ice_zone", dtype=DataType.INT8),
        FieldSchema(name="ground_resist", dtype=DataType.INT8),
        FieldSchema(name="no_joint", dtype=DataType.BOOL),

        FieldSchema(
            name="crossing_types",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=20,
            max_length=32
        ),

        # 关联的图片ID列表（对应table_images的image_id）
        FieldSchema(
            name="image_ids",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=10,
            max_length=128
        ),
    ]

    schema = CollectionSchema(
        fields,
        description="tower position chunks",
        enable_dynamic_field=True
    )
    collection = Collection("tower_chunks", schema)

    # HNSW索引，适合中等规模语义检索
    collection.create_index(
        "embedding",
        {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200}
        }
    )

    # 标量字段建索引，加速过滤
    collection.create_index("section", index_name="idx_section")
    collection.create_index("ice_zone", index_name="idx_ice_zone")
    collection.create_index("is_tension", index_name="idx_tension")
    collection.create_index("chunk_type", index_name="idx_chunk_type")

    print("✓ tower_chunks collection created")
    return collection


def search_towers(query: str, embedding_fn, filters: Dict = None, top_k: int = 5):
    """
    检索塔位chunk，同时返回关联的表格图片URL
    """
    chunk_col = Collection("tower_chunks")
    image_col = Collection("table_images")
    chunk_col.load()
    image_col.load()

    # 构建过滤表达式
    expr_parts = []
    if filters:
        if filters.get("section"):
            expr_parts.append(f'section == "{filters["section"]}"')
        if filters.get("ice_zone"):
            expr_parts.append(f'ice_zone == {filters["ice_zone"]}')
        if filters.get("is_tension") is not None:
            expr_parts.append(f'is_tension == {str(filters["is_tension"]).lower()}')
        if filters.get("no_joint"):
            expr_parts.append('no_joint == true')
    expr = " and ".join(expr_parts) if expr_parts else ""

    # 向量检索
    query_vec = embedding_fn(query)
    results = chunk_col.search(
        data=[query_vec],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        expr=expr if expr else None,
        output_fields=[
            "text", "tower_no", "tower_type", "section",
            "ice_zone", "crossing_types", "no_joint", "image_ids"
        ]
    )

    output = []
    for hit in results[0]:
        item = {
            "score": hit.score,
            "text": hit.entity.get("text"),
            "tower_no": hit.entity.get("tower_no"),
            "tower_type": hit.entity.get("tower_type"),
            "crossing_types": hit.entity.get("crossing_types"),
            "images": []
        }

        # 取关联图片（优先取切片图，再取整页图）
        image_ids = hit.entity.get("image_ids") or []
        if image_ids:
            # 用image_id IN (...)查图片信息
            id_list = '","'.join(image_ids[:5])
            img_expr = f'image_id in ["{id_list}"]'
            img_results = image_col.query(
                expr=img_expr,
                output_fields=["image_id", "full_url", "thumb_url",
                               "tower_from", "tower_to", "table_type"]
            )
            # 切片图优先
            imgs = sorted(
                img_results,
                key=lambda x: 0 if "slice" in x["image_id"] else 1
            )
            item["images"] = [{
                "thumb_url": img["thumb_url"],
                "full_url": img["full_url"],
                "tower_range": f"{img['tower_from']} ~ {img['tower_to']}",
                "type": img["table_type"],
            } for img in imgs]

        output.append(item)

    return output


# ── 使用示例 ──────────────────────────────────────────
if __name__ == "__main__":
    connect_milvus()
    create_collections()

    storage = ImageStorage("localhost:9000", "minio", "miniosecret", "tower-images")

    from openai import OpenAI

    openai_client = OpenAI(
        api_key="sk-901f743555d246408a9492dc96d57caa",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


    def embed(text: str):
        return openai_client.embeddings.create(
            input=text, model="text-embedding-v4"
        ).data[0].embedding


    pipeline = TowerPDFPipeline(storage, embed)
    pipeline.process_pdf(
        "E:\\03-汇能博友\\文档\\2.351-SA06911S-D0102 第6施工标段塔位明细表.pdf",
        project="甘肃浙江800kV",
        section="第6施工标段"
    )

    # 检索
    results = search_towers(
        query="跨越嘉陵江的耐张塔",
        embedding_fn=embed,
        filters={"section": "第6施工标段", "is_tension": True}
    )

    for r in results:
        print(f"\n{r['tower_no']} (score={r['score']:.3f})")
        print(r["text"])
        for img in r["images"]:
            print(f"  图片: {img['thumb_url']}")
            print(f"  范围: {img['tower_range']}")