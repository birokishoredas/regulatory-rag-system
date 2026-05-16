from typing import List, Dict
import re
from logger import GLOBAL_LOGGER as log
from src.core.prompts.prompt_library import QUERY_REWRITE_PROMPT_TEMPLATE

class SafeQueryRewriter:
    """
    Safety-first query rewriter for conversational RAG systems.

    Purpose:
        Resolves ambiguous references in user queries using minimal context
        from chat history without introducing hallucinations.
    """

    def __init__(self, llm):
        """
        Initialize SafeQueryRewriter.

        Args:
            llm: Async LLM client supporting `.ainvoke()`
        """
        self.llm = llm

    # -------------------------------------------------

    def should_rewrite(self, query: str) -> bool:
        """
        Detect whether a query requires rewriting.

        Logic:
            Identifies vague references that depend on prior context.

        Args:
            query (str): User query

        Returns:
            bool:
                True → rewrite required
                False → safe to use as-is
        """
        reference_patterns = [
            "this", "that", "it", "they",
            "those", "these",
            "section", "clause", "article",
            "above", "previous"
        ]

        return any(p in query.lower() for p in reference_patterns)

    # -------------------------------------------------

    async def rewrite(
        self,
        query: str,
        chat_history: List[Dict]
    ) -> str:
        """
        Rewrite query by resolving ambiguous references using chat history.

        Strategy:
            1. Skip rewriting if not needed
            2. Extract minimal context (last user + assistant turns)
            3. Use strict prompt to prevent answering
            4. Apply safety filters on output
            5. Fallback to original query if unsafe

        Args:
            query (str): Current user query
            chat_history (List[Dict]):
                Conversation history with roles:
                [{"role": "user"/"assistant", "content": str}]

        Returns:
            str:
                - Rewritten query (safe)
                - Original query (fallback on failure/unsafe)
        """

        # 1. Skip if not needed
        if not self.should_rewrite(query):
            return query

        # 2. Extract minimal context
        last_user = ""
        last_assistant = ""

        for h in reversed(chat_history):
            if h["role"] == "user" and not last_user:
                last_user = h["content"]
            elif h["role"] == "assistant" and not last_assistant:
                last_assistant = h["content"]

        context = f"""
Previous Question: {last_user}
Previous Answer: {self._strip_citations(last_assistant)}
"""

        # 3. Strict prompt
        prompt = QUERY_REWRITE_PROMPT_TEMPLATE.format(
            context=context,
            query=query
        )

        try:
            response = await self.llm.ainvoke(prompt)
            rewritten = response.content.strip()

            # 4. Safety validation
            if self._is_unsafe(rewritten, query):
                log.warning("unsafe_rewrite_detected", rewritten=rewritten)
                return query

            return rewritten

        except Exception as e:
            log.warning("rewrite_failed_fallback", error=str(e))
            return query

    # -------------------------------------------------

    def _is_unsafe(self, rewritten: str, original: str) -> bool:
        """
        Detect whether rewritten query is unsafe.

        Unsafe Conditions:
            - Too long → likely hallucination
            - Contains answer-like assertions
            - Semantically diverges from original query

        Args:
            rewritten (str): LLM-generated rewrite
            original (str): Original query

        Returns:
            bool:
                True → unsafe (fallback to original)
                False → safe rewrite
        """

        # Length explosion check
        if len(rewritten.split()) > len(original.split()) * 2:
            return True

        # Answer-like patterns (indicates hallucination)
        unsafe_patterns = [
            r"\bis\b",
            r"\bare\b",
            r"\bwas\b",
            r"\bmeans\b",
            r"\brefers to\b"
        ]

        if any(re.search(p, rewritten.lower()) for p in unsafe_patterns):
            return True

        # Semantic drift check
        if rewritten.lower() not in original.lower() and len(rewritten.split()) > 10:
            return True

        return False

    # -------------------------------------------------

    def _strip_citations(self, text: str) -> str:
        """
        Remove citation markers from text.

        Example:
            "Regulation applies [1][2]" → "Regulation applies"

        Args:
            text (str): Input text

        Returns:
            str: Cleaned text without citations
        """
        return re.sub(r"\[\d+\]", "", text or "")