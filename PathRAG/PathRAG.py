import asyncio
import os
from tqdm.asyncio import tqdm as tqdm_async
import networkx as nx
import numpy as np
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast
from langchain_gigachat.embeddings import GigaChatEmbeddings


from .llm import (
    gpt_4o_mini_complete,
    openai_embedding,
    EmbeddingWrapper,
    custom_embedding
)
from .operate import (
    chunking_by_token_size,
    extract_entities,
    kg_query,
)

from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)




def lazy_external_import(module_name: str, class_name: str):
    """Lazily import a class from an external module based on the package of the caller."""


    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        import importlib


        module = importlib.import_module(module_name, package=package)


        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


Neo4JStorage = lazy_external_import(".kg.neo4j_impl", "Neo4JStorage")
OracleKVStorage = lazy_external_import(".kg.oracle_impl", "OracleKVStorage")
OracleGraphStorage = lazy_external_import(".kg.oracle_impl", "OracleGraphStorage")
OracleVectorDBStorage = lazy_external_import(".kg.oracle_impl", "OracleVectorDBStorage")
MilvusVectorDBStorge = lazy_external_import(".kg.milvus_impl", "MilvusVectorDBStorge")
MongoKVStorage = lazy_external_import(".kg.mongo_impl", "MongoKVStorage")
ChromaVectorDBStorage = lazy_external_import(".kg.chroma_impl", "ChromaVectorDBStorage")
TiDBKVStorage = lazy_external_import(".kg.tidb_impl", "TiDBKVStorage")
TiDBVectorDBStorage = lazy_external_import(".kg.tidb_impl", "TiDBVectorDBStorage")
AGEStorage = lazy_external_import(".kg.age_impl", "AGEStorage")


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:

        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:

        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop

def create_graph_from_chunks(chunks, embeddings):
    """Создает граф знаний из текстовых фрагментов с семантическими связями."""
    graph = nx.DiGraph()
    embeddings_dict = {i: embeddings.embed_query(chunk) for i, chunk in enumerate(chunks)}

    for i, chunk in enumerate(chunks):
        graph.add_node(i, text=chunk)

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            similarity = cosine_similarity(embeddings_dict[i], embeddings_dict[j])
            if similarity > 0.7:
                graph.add_edge(i, j, weight=similarity)

    return graph


def flow_based_pruning(graph, start_node, end_node, alpha=0.8, theta=0.05):
    """Алгоритм обрезки путей: убирает слабые семантические связи и выделяет ключевые пути."""
    resources = {node: 0 for node in graph.nodes}
    resources[start_node] = 1.0

    for node in nx.topological_sort(graph):
        if node == start_node:
            continue
        for predecessor in graph.predecessors(node):
            successors = list(graph.successors(predecessor))
            if len(successors) == 0:
                continue
            resources[node] += alpha * resources[predecessor] / len(successors)

        current_successors = list(graph.successors(node))
        if len(current_successors) == 0:
            resources[node] = 0
        else:
            if resources[node] / len(current_successors) < theta:
                resources[node] = 0

    paths = list(nx.all_simple_paths(graph, start_node, end_node))
    paths_with_scores = [(path, sum(resources[node] for node in path) / len(path)) for path in paths]
    return sorted(paths_with_scores, key=lambda x: x[1], reverse=True)[:5]


def paths_to_text(graph, paths_with_scores):
    """Конвертирует пути графа в читаемый текст."""
    return "\n".join(
        f"Path (Score: {score:.2f}): {' -> '.join(graph.nodes[node]['text'] for node in path)}"
        for path, score in paths_with_scores
    )


def cosine_similarity(vec1, vec2):
    """Вычисляет косинусное сходство между двумя векторами."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


@dataclass
class PathRAG:
    api_key: str
    working_dir: str = field(
        default_factory=lambda: f"./PathRAG_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    embedding_cache_config: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)


    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"


    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500


    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )


    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 3


    llm_model_func: callable = gpt_4o_mini_complete
    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)


    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    enable_llm_cache: bool = True


    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("Authorization_key must be provided when initializing PathRAG.")

        log_file = os.path.join("PathRAG.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")


        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (
            self._get_storage_class()[self.kv_storage]
        )
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[
            self.vector_storage
        ]
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[
            self.graph_storage
        ]
        # ✅ Используем EmbeddingWrapper для обоих случаев
        # Если embedding_func не задан при создании объекта, создаем дефолтный
        if not isinstance(self.embedding_func, EmbeddingWrapper):
            self.embedding_func = EmbeddingWrapper(
                func=custom_embedding,
                dim=1024,
                api_key=self.api_key
            )

        self.graph = nx.DiGraph()
        self.graph_embeddings = self.embedding_func  # Используем ту же функцию и для графа

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache
            else None
        )
        #self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            #self.embedding_func
        #)


        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )


        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    global_config=asdict(self),
                ),
                **self.llm_model_kwargs,
            )
        )

    def _get_storage_class(self) -> Type[BaseGraphStorage]:
        return {

            "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            "MongoKVStorage": MongoKVStorage,
            "TiDBKVStorage": TiDBKVStorage,

            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            "MilvusVectorDBStorge": MilvusVectorDBStorge,
            "ChromaVectorDBStorage": ChromaVectorDBStorage,
            "TiDBVectorDBStorage": TiDBVectorDBStorage,

            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
            "AGEStorage": AGEStorage,

        }

    def insert(self, string_or_strings):

        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        update_storage = False
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            update_storage = True
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks = {}
            for doc_key, doc in tqdm_async(
                new_docs.items(), desc="Chunking documents", unit="doc"
            ):
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            await self.chunks_vdb.upsert(inserting_chunks)

            logger.info("[Entity Extraction]...")
            maybe_new_kg = await extract_entities(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            if update_storage:
                await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def insert_custom_kg(self, custom_kg: dict):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert_custom_kg(custom_kg))

    async def ainsert_custom_kg(self, custom_kg: dict):
        update_storage = False
        try:

            all_chunks_data = {}
            chunk_to_source_map = {}
            for chunk_data in custom_kg.get("chunks", []):
                chunk_content = chunk_data["content"]
                source_id = chunk_data["source_id"]
                chunk_id = compute_mdhash_id(chunk_content.strip(), prefix="chunk-")

                chunk_entry = {"content": chunk_content.strip(), "source_id": source_id}
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if self.chunks_vdb is not None and all_chunks_data:
                await self.chunks_vdb.upsert(all_chunks_data)
            if self.text_chunks is not None and all_chunks_data:
                await self.text_chunks.upsert(all_chunks_data)


            all_entities_data = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = f'"{entity_data["entity_name"].upper()}"'
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")

                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")


                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )


                node_data = {
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                }

                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, node_data=node_data
                )
                node_data["entity_name"] = entity_name
                all_entities_data.append(node_data)
                update_storage = True


            all_relationships_data = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = f'"{relationship_data["src_id"].upper()}"'
                tgt_id = f'"{relationship_data["tgt_id"].upper()}"'
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)

                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")


                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )


                for need_insert_id in [src_id, tgt_id]:
                    if not (
                        await self.chunk_entity_relation_graph.has_node(need_insert_id)
                    ):
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": "UNKNOWN",
                            },
                        )


                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data={
                        "weight": weight,
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                    },
                )
                edge_data = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                }
                all_relationships_data.append(edge_data)
                update_storage = True


            if self.entities_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                        "content": dp["entity_name"] + dp["description"],
                        "entity_name": dp["entity_name"],
                    }
                    for dp in all_entities_data
                }
                await self.entities_vdb.upsert(data_for_vdb)


            if self.relationships_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                        "src_id": dp["src_id"],
                        "tgt_id": dp["tgt_id"],
                        "content": dp["keywords"]
                        + dp["src_id"]
                        + dp["tgt_id"]
                        + dp["description"],
                    }
                    for dp in all_relationships_data
                }
                await self.relationships_vdb.upsert(data_for_vdb)
        finally:
            if update_storage:
                await self._insert_done()

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def ainsert_with_graphs(self, string_or_strings):
        """Вставка данных и построение графа знаний."""
        if isinstance(string_or_strings, str):
            string_or_strings = [string_or_strings]

        new_docs = {
             compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
             for c in string_or_strings
        }

        # Создаем граф знаний
        chunks = [doc["content"] for doc in new_docs.values()]
        self.graph = create_graph_from_chunks(chunks, self.graph_embeddings)

        # Извлекаем семантические пути
        paths_with_scores = flow_based_pruning(self.graph, 0, len(chunks) - 1)
        path_text = paths_to_text(self.graph, paths_with_scores)

        logger.info("🔗 Извлеченные пути из графа знаний:")
        logger.info(path_text)

        # Сохраняем данные в векторную базу
        await self.chunks_vdb.upsert(new_docs)

    async def aquery_with_graphs(self, query: str, param: QueryParam = QueryParam()):
        """Поиск через векторную базу + семантические графы."""
        docs = await self.chunks_vdb.query(query, top_k=10)
        chunks = [doc["content"] for doc in docs]

        # Построение графа из результатов поиска
        self.graph = create_graph_from_chunks(chunks, self.graph_embeddings)
        paths_with_scores = flow_based_pruning(self.graph, 0, len(chunks) - 1)
        path_text = paths_to_text(self.graph, paths_with_scores)

        logger.info("🔍 Найденные семантические пути:")
        logger.info(path_text)

        return f"🔍 Найденные пути:\n{path_text}"

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode in ["hybrid"]:
            response= await kg_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    global_config=asdict(self),
                ),
            )
            print("response all ready")
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response


    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def delete_by_entity(self, entity_name: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        entity_name = f'"{entity_name.upper()}"'

        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
