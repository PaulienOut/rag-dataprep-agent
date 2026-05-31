CONTENT_METADATA_PROMPT = """You prepare documents for a RAG knowledge base.
Return concise JSON with these fields:
- title: string or null
- subject: string or null
- summary: 1-3 sentence string
- keywords: list of 3-8 short strings

Only use evidence in the provided document text.
"""
