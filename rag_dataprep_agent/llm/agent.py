from __future__ import annotations

import json

from openai import OpenAI

from rag_dataprep_agent.llm.prompts import CONTENT_METADATA_PROMPT
from rag_dataprep_agent.models import ContentMetadata, ParsedDocument
from rag_dataprep_agent.tools.content_metadata import extract_content_metadata


class DocumentPrepAgent:
    """Thin LLM wrapper that uses local tools for the rest of the pipeline."""

    def __init__(self, client: OpenAI | None, metadata_model: str) -> None:
        self.client = client
        self.metadata_model = metadata_model

    def extract_content_metadata(self, parsed: ParsedDocument) -> ContentMetadata:
        fallback = extract_content_metadata(parsed)
        if self.client is None or not parsed.text.strip():
            return fallback

        response = self.client.responses.create(
            model=self.metadata_model,
            input=[
                {"role": "system", "content": CONTENT_METADATA_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"PDF metadata: {json.dumps(parsed.pdf_metadata, default=str)}\n\n"
                        f"Document text sample:\n{parsed.text[:12000]}"
                    ),
                },
            ],
            text={"format": {"type": "json_object"}},
        )
        try:
            data = json.loads(response.output_text)
        except json.JSONDecodeError:
            return fallback

        return ContentMetadata(
            title=data.get("title") or fallback.title,
            subject=data.get("subject") or fallback.subject,
            summary=data.get("summary") or fallback.summary,
            keywords=[str(item) for item in data.get("keywords", fallback.keywords)],
            document_metadata={
                "place": (
                    data.get("document_metadata", {}).get("place")
                    or getattr(fallback, "document_metadata", {}).get("place")
                ),
                "date_of_publication": (
                    data.get("document_metadata", {}).get("date_of_publication")
                    or getattr(fallback, "document_metadata", {}).get("date_of_publication")
                ),
            },
            layout_metadata={
                "header": (
                    data.get("layout_metadata", {}).get("header")
                    or getattr(fallback, "layout_metadata", {}).get("header")
                ),
                "footer": (
                    data.get("layout_metadata", {}).get("footer")
                    or getattr(fallback, "layout_metadata", {}).get("footer")
                ),
            },
        )

    def embed_texts(self, texts: list[str], embedding_model: str) -> list[list[float]] | None:
        if self.client is None or not texts:
            return None
        response = self.client.embeddings.create(model=embedding_model, input=texts)
        return [item.embedding for item in response.data]
