import os
import sys
import asyncio
import re
from typing import List, Dict, Any, Union

from dotenv import load_dotenv
from llama_parse import LlamaParse

from logger import GLOBAL_LOGGER as log
from exception.custom_exception import RegulatoryRAGException


class LlamaDocumentParser:
    """
    Advanced parser for extracting and structuring regulatory documents using LlamaParse.

    Enhancements over basic parser:
        - Preserves document hierarchy (sections, subsections)
        - Supports structured parsing (optional)
        - Maintains backward compatibility (returns raw text by default)

    Attributes:
        api_key (str): API key for Llama Cloud.
        parser (LlamaParse): Markdown-based parsing client.
    """

    SECTION_PATTERN = re.compile(
        r"(?P<number>\d+(\.\d+)*)\.\s+(?P<title>[^\n]+)"
    )

    def __init__(self):
        """
        Initializes the parser with structure-preserving configuration.

        Raises:
            RegulatoryRAGException:
                - If API key is missing
                - If parser initialization fails
        """

        load_dotenv()

        self.api_key = os.getenv("LLAMA_CLOUD_API_KEY")

        if not self.api_key:
            log.error("llama_parser_missing_api_key")

            raise RegulatoryRAGException(
                "LLAMA_CLOUD_API_KEY not set",
                sys,
            )

        try:
            self.parser = LlamaParse(
                api_key=self.api_key,
                result_type="markdown",
                verbose=False,
                parsing_instruction="""
                Preserve all section numbers and headings.
                Do NOT merge numbered sections.
                Keep numbering like 1., 1.1, 1.1.1 explicitly.
                Maintain hierarchy and structure.
                """,
            )

            log.info(
                "llama_parser_initialized",
                mode="structured_markdown",
            )

        except Exception as e:
            log.error(
                "llama_parser_init_failed",
                error=str(e),
            )

            raise RegulatoryRAGException(e, sys)

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    async def parse(
            self,
            file_path: str,
            return_structured: bool = False,
            return_both: bool = False,
        ) -> Union[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Parses a document and optionally returns structured sections.

        Args:
            file_path (str): Path to document
            return_structured (bool):
                - False → returns raw markdown text (default, backward compatible)
                - True → returns structured sections

        Returns:
            Union[str, List[Dict]]:
                Raw text OR structured sections

        Raises:
            RegulatoryRAGException
        """

        if not os.path.exists(file_path):
            raise RegulatoryRAGException(
                f"File not found: {file_path}",
                sys,
            )

        log.info("llama_parsing_started", file=file_path)

        try:
            documents = await asyncio.to_thread(
                self.parser.load_data,
                file_path,
            )

            text = self._join_documents(documents)

            log.info(
                "llama_parsing_completed",
                file=file_path,
                length=len(text),
            )
            sections = self._extract_sections(text)
            if return_both:
                return {
                    "content": text,
                    "sections": sections,
                }
            if return_structured:
                return sections

            return text

        except Exception as e:
            log.error(
                "llama_parsing_failed",
                file=file_path,
                error=str(e),
            )

            raise RegulatoryRAGException(e, sys)

    # -------------------------------------------------
    # Structured Parsing
    # -------------------------------------------------

    def _extract_sections(
        self,
        text: str,
    ) -> List[Dict[str, Any]]:
        """
        Extracts structured sections from markdown text.

        Detects hierarchical numbering patterns such as:
            - 1.
            - 1.1
            - 1.1.2

        Returns:
            List of sections with:
                - number
                - title
                - content
        """

        matches = list(self.SECTION_PATTERN.finditer(text))

        if not matches:
            log.warning("no_sections_detected")
            return []

        sections = []

        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            number = match.group("number")
            title = match.group("title").strip()
            content = text[start:end].strip()

            sections.append(
                {
                    "number": number,
                    "title": title,
                    "content": content,
                }
            )

        return sections

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def _join_documents(
        self,
        documents: List,
    ) -> str:
        """
        Combines parsed document segments into a single text.

        Args:
            documents (List): LlamaParse output

        Returns:
            str: Combined markdown text
        """

        texts = []

        for doc in documents:
            if hasattr(doc, "text") and doc.text:
                texts.append(doc.text)

        return "\n\n".join(texts)