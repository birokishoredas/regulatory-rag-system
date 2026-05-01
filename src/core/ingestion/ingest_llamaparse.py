import os
import re
import glob
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from src.core.chunking.chunker_llamaparse import create_chunker
from src.infra.models.embedder import create_embedder
from src.core.ingestion.cleaning import DocumentCleaner
from src.infra.db.db_utils import (
    init_db_pool,
    close_db_pool,
    get_document_by_hash,
    get_document_by_source,
    delete_document_and_chunks,
)
from src.infra.db import db_utils
from src.shared.schemas import IngestionConfig, IngestionResult, DocumentChunk, ChunkingConfig
from src.core.parsing.llama_parser import LlamaDocumentParser
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException


load_dotenv()
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


class DocumentIngestionPipeline:
    """
    End-to-end ingestion pipeline using LlamaParse for document parsing.

    Handles parsing, chunking, embedding, and storing documents into PostgreSQL.

    Attributes:
        config (IngestionConfig): Configuration for ingestion and chunking.
        documents_folder (str): Directory containing documents to ingest.
        cleaner (DocumentCleaner): Cleaning component (reserved for extensibility).
        chunker_config (ChunkingConfig): Configuration for chunking behavior.
        chunker (object): Component responsible for document chunking.
        embedder (object): Component responsible for generating embeddings.
        parser (LlamaDocumentParser): Parser for extracting document content.
        _initialized (bool): Tracks database initialization state.
    """

    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "data",
    ):
        """
        Initializes the ingestion pipeline with required components.

        Args:
            config (IngestionConfig): Configuration controlling ingestion parameters.
            documents_folder (str): Path to the folder containing documents.
        """

        self.config = config
        self.documents_folder = documents_folder

        # Cleaner kept for future extensibility (currently unused)
        # self.cleaner = DocumentCleaner()

        # Chunking configuration
        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size
        )

        # Core components
        self.chunker = create_chunker(self.chunker_config)
        self.embedder = create_embedder()
        self.parser = LlamaDocumentParser()

        # Internal state flag
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self):
        """
        Initializes database connections required for ingestion.

        Returns:
            None

        Raises:
            RegulatoryRAGException: If database initialization fails.
        """

        if self._initialized:
            return

        try:
            await init_db_pool()
            self._initialized = True

            log.info("ingestion_pipeline_initialized")

        except Exception as e:
            log.error("db_initialization_failed", error=str(e))
            raise RegulatoryRAGException(e)

    async def close(self):
        """
        Closes database connections used by the pipeline.

        Returns:
            None

        Raises:
            RegulatoryRAGException: If database shutdown fails.
        """

        if not self._initialized:
            return

        try:
            await close_db_pool()
            self._initialized = False

            log.info("ingestion_pipeline_closed")

        except Exception as e:
            log.error("db_shutdown_failed", error=str(e))
            raise RegulatoryRAGException(e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_documents(
        self,
        progress_callback: Optional[callable] = None,
    ) -> List[IngestionResult]:
        """
        Ingests all supported documents from the configured folder.

        Processes documents sequentially and optionally reports progress.

        Args:
            progress_callback (Optional[callable]):
                Function to report progress with signature (current, total).

        Returns:
            List[IngestionResult]: Results of ingestion for each document.
        """

        if not self._initialized:
            await self.initialize()

        document_files = self._find_document_files()

        if not document_files:
            log.warning("no_supported_documents_found")
            return []

        log.info(
            "ingestion_started",
            document_count=len(document_files),
        )

        results: List[IngestionResult] = []

        for idx, file_path in enumerate(document_files, start=1):

            try:
                log.info(
                    "document_processing_started",
                    file=str(file_path),
                    index=idx,
                    total=len(document_files),
                )

                result = await self._ingest_single_document(file_path)
                results.append(result)

                if progress_callback:
                    progress_callback(idx, len(document_files))

            except RegulatoryRAGException as e:
                log.error("document_ingestion_failed", error=str(e))

                results.append(
                    IngestionResult(
                        document_id="",
                        title=Path(file_path).name,
                        chunks_created=0,
                        processing_time_ms=0,
                        errors=[str(e)],
                    )
                )

        log.info("ingestion_completed", processed=len(results))

        return results

    # ------------------------------------------------------------------
    # Core ingestion logic
    # ------------------------------------------------------------------
    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        """
        Processes a single document using structure-aware parsing and dual-index chunking.

        Enhancements:
            - Uses structured parsing for regulatory documents
            - Generates BOTH:
                • Paragraph-level chunks (fine-grained retrieval)
                • Section-level chunks (coarse contextual retrieval)
            - Enables multi-granularity (dual-index) retrieval
            - Falls back safely if structure extraction fails

        Returns:
            IngestionResult
        """

        start_time = datetime.now()

        try:
            # --------------------------------------------------
            # STRUCTURED PARSING
            # --------------------------------------------------
            parsed = await self.parser.parse(
                        file_path,
                        return_both=True,
                    )
            content = parsed["content"]
            sections = parsed["sections"]

            content = self._clean_content(content)

            # Title
            title = self._extract_title(content, file_path)

            content = f"# {title}\n\n{content}"

            # --------------------------------------------------
            # Metadata + hashing
            # --------------------------------------------------
            source = os.path.relpath(file_path, self.documents_folder)
            metadata = self._extract_document_metadata(content, file_path)
            file_hash = self._compute_file_hash(content)

            # --------------------------------------------------
            # Idempotency check
            # --------------------------------------------------
            existing = await get_document_by_hash(file_hash)

            if existing:
                log.info("document_skipped_hash_match", title=title)

                return IngestionResult(
                    document_id=existing["id"],
                    title=title,
                    chunks_created=0,
                    processing_time_ms=0,
                    errors=[],
                )

            # --------------------------------------------------
            # Update existing document
            # --------------------------------------------------
            existing_source = await get_document_by_source(source)

            if existing_source:
                log.info("document_updated_existing_source", title=title)
                await delete_document_and_chunks(existing_source["id"])

            # --------------------------------------------------
            # PARAGRAPH CHUNKS (PRIMARY - REGULATORY)
            # --------------------------------------------------
            paragraph_chunks = await self.chunker.chunk_document(
                content=content,
                sections=sections,
                title=title,
                source=source,
                metadata=metadata,
            )

            # Ensure chunk_type is present (important for retrieval)
            for c in paragraph_chunks:
                c.metadata["chunk_type"] = "paragraph"

            # --------------------------------------------------
            # SECTION CHUNKS (NEW - DUAL INDEX)
            # --------------------------------------------------
            section_chunks: List[DocumentChunk] = []

            if sections:
                for i, section in enumerate(sections):

                    section_text = f"""
    Document: {title}
    Section: {section.get('number', '')} - {section.get('title', '')}

    {self._clean_content(section.get('content', ''))}
    """.strip()

                    section_chunks.append(
                        DocumentChunk(
                            content=section_text,
                            index=10_000 + i,  # avoid index collision
                            start_char=0,
                            end_char=len(section_text),
                            metadata={
                                "title": title,
                                "source": source,
                                "section": section.get("number", ""),
                                "section_title": section.get("title", ""),
                                "chunk_type": "section",   
                                **(metadata or {}),
                            },
                        )
                    )

            # --------------------------------------------------
            # MERGE CHUNKS (DUAL INDEX READY)
            # --------------------------------------------------
            chunks = paragraph_chunks + section_chunks

            if not chunks:
                log.warning("no_chunks_created", title=title)

                return IngestionResult(
                    document_id="",
                    title=title,
                    chunks_created=0,
                    processing_time_ms=0,
                    errors=["No chunks created"],
                )

            log.info(
                "chunking_completed",
                title=title,
                total_chunks=len(chunks),
                paragraph_chunks=len(paragraph_chunks),
                section_chunks=len(section_chunks),
                structured_used=bool(sections),
            )

            # --------------------------------------------------
            # EMBEDDING
            # --------------------------------------------------
            embedded_chunks = await self.embedder.embed_chunks(chunks)

            log.info(
                "embedding_completed",
                title=title,
                chunk_count=len(embedded_chunks),
            )

            # --------------------------------------------------
            # PERSISTENCE
            # --------------------------------------------------
            document_id = await self._save_to_postgres(
                title=title,
                source=source,
                content=content,
                file_hash=file_hash,
                chunks=embedded_chunks,
                metadata=metadata,
            )

            processing_time = (
                datetime.now() - start_time
            ).total_seconds() * 1000

            log.info(
                "document_ingested",
                title=title,
                document_id=document_id,
                duration_ms=int(processing_time),
            )

            return IngestionResult(
                document_id=document_id,
                title=title,
                chunks_created=len(chunks),
                processing_time_ms=processing_time,
                errors=[],
            )

        except Exception as e:
            raise RegulatoryRAGException(e)
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _clean_content(self, text: str) -> str:
        # Remove warning icons / UI artifacts
        text = re.sub(r"⚠️.*?\n", "", text)

        # Remove standalone labels
        text = re.sub(r"Conditions\s*/\s*Exceptions", "", text, flags=re.IGNORECASE)

        # Remove excessive symbols
        text = re.sub(r"[■●►▶]+", "", text)

        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
    def _find_document_files(self) -> List[str]:
        """
        Finds all supported document files in the documents folder.

        Returns:
            List[str]: List of matching file paths.
        """
        files: List[str] = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(
                glob.glob(
                    os.path.join(self.documents_folder, f"**/*{ext}"),
                    recursive=True,
                )
            )
        return files

    def _compute_file_hash(self, content: str) -> str:
        """
        Computes SHA-256 hash of document content.

        Args:
            content (str): Document content.

        Returns:
            str: Hash string.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _extract_title(self, content: str, file_path: str) -> str:
        """
        Extracts document title from file name.

        Args:
            content (str): Document content.
            file_path (str): Path to file.

        Returns:
            str: Extracted title.
        """
        return Path(file_path).stem

    def _extract_document_metadata(
        self,
        content: str,
        file_path: str,
    ) -> Dict[str, Any]:
        """
        Extracts metadata from document content and file.

        Args:
            content (str): Document content.
            file_path (str): File path.

        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        return {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "file_size": len(content),
            "line_count": len(content.splitlines()),
            "word_count": len(content.split()),
            "ingested_at": datetime.now().isoformat(),
        }

    async def _save_to_postgres(
        self,
        title: str,
        source: str,
        content: str,
        file_hash: str,
        chunks: List[DocumentChunk],
        metadata: Dict[str, Any],
    ) -> str:
        """
        Saves document and its chunks into PostgreSQL database.

        Args:
            title (str): Document title.
            source (str): Relative file path.
            content (str): Full document content.
            file_hash (str): Unique content hash.
            chunks (List[DocumentChunk]): List of chunks with embeddings.
            metadata (Dict[str, Any]): Document metadata.

        Returns:
            str: Stored document ID.
        """

        async with db_utils.db_pool.acquire() as conn:
            async with conn.transaction():

                doc = await conn.fetchrow(
                    """
                    INSERT INTO public.documents
                    (title, source, file_hash, content, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (file_hash)
                    DO UPDATE SET updated_at = now()
                    RETURNING id::text
                    """,
                    title,
                    source,
                    file_hash,
                    content,
                    json.dumps(metadata),
                )

                document_id = doc["id"]

                await conn.execute(
                    """
                    DELETE FROM public.chunks
                    WHERE document_id = $1::uuid
                    """,
                    document_id,
                )

                for chunk in chunks:
                    embedding = (
                        "[" + ",".join(map(str, chunk.embedding)) + "]"
                        if chunk.embedding
                        else None
                    )

                    await conn.execute(
                        """
                        INSERT INTO public.chunks
                        (document_id, content, embedding, chunk_index, metadata, token_count)
                        VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                        """,
                        document_id,
                        chunk.content,
                        embedding,
                        chunk.index,
                        json.dumps(chunk.metadata),
                        chunk.token_count,
                    )

                return document_id