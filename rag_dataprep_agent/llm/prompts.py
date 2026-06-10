CONTENT_METADATA_PROMPT = """
You prepare documents for a RAG knowledge base.

Return valid JSON with exactly these fields:

{
  "title": string | null,
  "subject": string | null,
  "summary": string | null,
  "keywords": [string],
  "document_metadata": {
    "place": string | null,
    "date_of_publication": string | null
  },
  "layout_metadata": {
    "header": string | null,
    "footer": string | null
  }
}

Extraction rules:

- title: main document title.
- subject: official subject or topic if explicitly stated.
- place: publication location from the first page only.
- date_of_publication: publication date from the first page only.
  Example: if the first page says "Brussels, 18 December 2025",
  then place is "Brussels" and date_of_publication is "18 December 2025".
- summary: concise 1-3 sentence summary.
- keywords: 3-8 short keywords.
- header: repeated text at the TOP of pages after the first page.
- footer: repeated text at the BOTTOM of pages after the first page.

Important:
- Do not confuse top-of-page text with bottom-of-page text.
- Page numbers are not the header.
- Language codes such as "EN" are not the header.
- Document codes such as "EUCO 24/25" usually belong to the footer if repeated at the bottom.
- Use only information present in the document.
- If a field cannot be determined, return null.
- Return JSON only.
"""