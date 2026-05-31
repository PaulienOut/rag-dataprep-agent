from rag_dataprep_agent.tools.chunking import chunk_text


def test_chunk_text_uses_overlap_and_stable_ids() -> None:
    text = " ".join(f"word{i}" for i in range(120))

    chunks = chunk_text(text, chunk_size=120, overlap=20)

    assert len(chunks) > 1
    assert chunks[0].chunk_id == "chunk-0001"
    assert chunks[1].start_char < chunks[0].end_char
    assert all(chunk.text for chunk in chunks)
