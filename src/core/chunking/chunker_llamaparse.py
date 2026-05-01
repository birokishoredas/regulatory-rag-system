from typing import List, Dict, Any, Optional, Union

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.schema import Document

from src.infra.models.model_loader import ModelLoader
from src.shared.schemas import DocumentChunk, ChunkingConfig
from logger import GLOBAL_LOGGER as log


# =========================================================
# Regulatory Chunker (NEW - PRIMARY)
# =========================================================
class RegulatoryChunker:
    """
    Advanced structure-aware chunker for regulatory/legal documents.

    This chunker improves upon basic paragraph splitting by incorporating:
    
    1. Hierarchical Context Enrichment
       - Injects document title and section hierarchy into each chunk
       - Helps retrieval models understand legal scope and structure

    2. Sliding Window Overlap
       - Maintains partial overlap between chunks (~10–20%)
       - Prevents context loss at chunk boundaries

    3. Token-Aware Chunking
       - Ensures chunks are neither too small nor too large
       - Improves embedding quality and retrieval performance

    4. Multi-Paragraph Grouping
       - Groups related paragraphs into semantically meaningful chunks
       - Avoids overly fragmented chunks

    5. Backward Compatibility
       - Maintains same interface and return type as previous implementation
       - Safe to use as drop-in replacement

    Designed for:
        - Regulatory / legal documents
        - Structured parsing outputs (sections, clauses)
        - High-precision RAG systems
    """

    def __init__(
        self,
        max_tokens: int = 400,
        min_tokens: int = 120,
        overlap_ratio: float = 0.15,
    ):
        """
        Initialize chunker configuration.

        Args:
            max_tokens (int):
                Maximum token size per chunk. Prevents overly large chunks.
            
            min_tokens (int):
                Minimum token threshold before emitting a chunk.
                Helps avoid weak, low-signal chunks.
            
            overlap_ratio (float):
                Fraction of previous chunk to retain as overlap (0.1–0.2 recommended).
                Ensures continuity across chunk boundaries.
        """
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_ratio = overlap_ratio

    async def chunk_document(
        self,
        *,
        content: Optional[str] = None,
        sections: Optional[List[Dict[str, Any]]] = None,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[DocumentChunk]:
        """
        Convert structured document sections into enriched, token-balanced chunks.

        This method:
        - Iterates through parsed sections
        - Groups paragraphs into chunks based on token thresholds
        - Injects hierarchical context (document + section)
        - Applies sliding overlap between chunks
        - Preserves positional information for traceability

        Args:
            content (Optional[str]):
                Full raw document text. Used to compute character offsets.
            
            sections (Optional[List[Dict]]):
                Structured sections from parser. Each section must include:
                    - number
                    - title
                    - content
            
            title (str):
                Document title.
            
            source (str):
                Source file path or identifier.
            
            metadata (Optional[Dict]):
                Additional metadata to attach to each chunk.

        Returns:
            List[DocumentChunk]:
                List of enriched chunks with hierarchical context and metadata.

        Behavior:
            - Returns empty list if no sections are available
            - Ensures consistent chunk sizes
            - Maintains backward compatibility with existing pipeline
        """

        if not sections:
            log.warning("regulatory_chunker_no_sections_fallback")
            return []

        chunks: List[DocumentChunk] = []
        chunk_index = 0
        global_pos = 0

        for section in sections:

            section_number = section.get("number", "")
            section_title = section.get("title", "")
            section_content = section.get("content", "")

            paragraphs = [
                p.strip() for p in section_content.split("\n") if p.strip()
            ]

            # -------------------------------
            # Hierarchical context prefix
            # -------------------------------
            prefix = (
                f"Document: {title}\n"
                f"Section: {section_number} - {section_title}\n"
            )

            buffer = []
            buffer_tokens = 0

            for para in paragraphs:

                para_tokens = self._estimate_tokens(para)

                # Emit chunk if exceeding max tokens
                if buffer and (buffer_tokens + para_tokens > self.max_tokens):

                    chunk_text = self._build_chunk(prefix, buffer)

                    start_char = content.find(buffer[0], global_pos) if content else global_pos
                    if start_char == -1:
                        start_char = global_pos

                    end_char = start_char + len(" ".join(buffer))
                    global_pos = end_char

                    chunks.append(
                        DocumentChunk(
                            content=chunk_text,
                            index=chunk_index,
                            start_char=start_char,
                            end_char=end_char,
                            metadata={
                                "title": title,
                                "source": source,
                                "section": section_number,
                                "section_title": section_title,
                                "chunk_method": "regulatory_enhanced",
                                "chunk_type": "paragraph",
                                **(metadata or {}),
                            },
                        )
                    )

                    chunk_index += 1

                    # -------------------------------
                    # Sliding overlap
                    # -------------------------------
                    overlap_size = max(
                        1,
                        int(len(buffer) * self.overlap_ratio)
                    )

                    buffer = buffer[-overlap_size:]
                    buffer_tokens = sum(self._estimate_tokens(p) for p in buffer)

                buffer.append(para)
                buffer_tokens += para_tokens

            # Flush remaining buffer
            if buffer and buffer_tokens >= self.min_tokens:

                chunk_text = self._build_chunk(prefix, buffer)

                start_char = content.find(buffer[0], global_pos) if content else global_pos
                if start_char == -1:
                    start_char = global_pos

                end_char = start_char + len(" ".join(buffer))
                global_pos = end_char

                chunks.append(
                    DocumentChunk(
                        content=chunk_text,
                        index=chunk_index,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={
                            "title": title,
                            "source": source,
                            "section": section_number,
                            "section_title": section_title,
                            "chunk_method": "regulatory_enhanced",
                            **(metadata or {}),
                        },
                    )
                )

                chunk_index += 1

        log.info(
            "regulatory_enhanced_chunking_completed",
            title=title,
            chunk_count=len(chunks),
        )

        return chunks

    # --------------------------------------------------

    def _build_chunk(self, prefix: str, paragraphs: List[str]) -> str:
        """
        Construct final chunk text with hierarchical context.

        Args:
            prefix (str):
                Context prefix including document and section metadata.
            
            paragraphs (List[str]):
                List of grouped paragraphs forming the chunk.

        Returns:
            str:
                Final chunk text with context + content.
        """
        return prefix + "\n" + "\n".join(paragraphs)

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a given text.

        Uses a simple heuristic:
            1 token ≈ 4 characters

        Args:
            text (str): Input text

        Returns:
            int: Estimated token count
        """
        return max(1, len(text) // 4)


# =========================================================
# Semantic Chunker (EXISTING - FALLBACK)
# =========================================================
class SemanticChunker:
    """
    Embedding-based chunker used as fallback when structure parsing fails.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config

        try:
            embedding_client = ModelLoader().load_embeddings()

            embed_model = LangchainEmbedding(
                langchain_embeddings=embedding_client
            )

            self.parser = SemanticSplitterNodeParser(
                embed_model=embed_model,
                buffer_size=1,
                breakpoint_percentile_threshold=95,
            )

            log.info("semantic_chunker_initialized")

        except Exception as e:
            log.error("semantic_chunker_failed", error=str(e))
            raise

    async def chunk_document(
        self,
        *,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[DocumentChunk]:

        if not content.strip():
            return []

        try:
            doc = Document(text=content)
            nodes = self.parser.get_nodes_from_documents([doc])

            chunks: List[DocumentChunk] = []
            start_pos = 0

            for idx, node in enumerate(nodes):

                text = node.text.strip()
                if not text:
                    continue

                found_at = content.find(text, start_pos)
                if found_at == -1:
                    found_at = start_pos

                end_pos = found_at + len(text)
                start_pos = end_pos

                chunks.append(
                    DocumentChunk(
                        content=text,
                        index=idx,
                        start_char=found_at,
                        end_char=end_pos,
                        metadata={
                            "title": title,
                            "source": source,
                            "chunk_method": "semantic",
                            "chunk_type": "paragraph",
                            **(metadata or {}),
                        },
                    )
                )

            log.info(
                "semantic_chunking_completed",
                title=title,
                chunk_count=len(chunks),
            )

            return chunks

        except Exception as e:
            log.warning("semantic_failed_fallback", error=str(e))
            return []


# =========================================================
# Hybrid Chunker (NEW - SMART SWITCH)
# =========================================================
class HybridChunker:
    """
    Hybrid chunker that selects the best strategy:

        1. RegulatoryChunker (if structure exists)
        2. SemanticChunker (fallback)

    This ensures robustness across document types.
    """

    def __init__(self, config: ChunkingConfig):
        self.semantic_chunker = SemanticChunker(config)
        self.regulatory_chunker = RegulatoryChunker()

    async def chunk_document(
        self,
        *,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        sections: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> List[DocumentChunk]:

        # Try structured chunking first
        if sections:
            chunks = await self.regulatory_chunker.chunk_document(
                content=content,
                sections=sections,
                title=title,
                source=source,
                metadata=metadata,
            )

            if chunks:
                return chunks

        # Fallback to semantic
        log.info("fallback_to_semantic_chunking")

        return await self.semantic_chunker.chunk_document(
            content=content,
            title=title,
            source=source,
            metadata=metadata,
        )


# =========================================================
# Factory
# =========================================================
def create_chunker(config: ChunkingConfig):
    """
    Factory to create hybrid chunker.

    Returns:
        HybridChunker
    """
    return HybridChunker(config)