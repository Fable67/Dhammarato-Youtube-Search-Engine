"""
Basic snapshot/sanity tests for sanghabot.ingest.chunker.chunk_transcript_semantic().

Kept intentionally light: this algorithm was ported near-verbatim from the
old (well-written) chunking.py, so the goal here is a smoke test that the
port didn't break anything structurally -- not a full re-validation of the
chunking algorithm's design.
"""
from sanghabot.ingest.chunker import chunk_transcript_semantic


def _fake_embed_fn(texts: list[str]):
    """Deterministic fake embeddings for testing without hitting any API:
    embeds each text as a simple bag-of-words vector so cosine similarity
    is meaningful for the test transcript below."""
    import numpy as np

    vocab = ["speaker", "meditation", "breath", "jhana", "weather", "rain", "cooking", "recipe"]
    vectors = []
    for text in texts:
        lower = text.lower()
        vec = np.array([lower.count(w) for w in vocab], dtype="float32")
        if vec.sum() == 0:
            vec = np.ones(len(vocab), dtype="float32")
        vectors.append(vec)
    return vectors


def test_chunker_returns_no_boundaries_for_short_uniform_text():
    text = "Speaker A: This is a short talk about meditation. It is brief."
    borders = chunk_transcript_semantic(
        text, refine_with_embeddings=False, min_chunk_chars=1500,
    )
    # Too short to force any split.
    assert borders == []


def test_chunker_forces_split_on_max_chunk_chars():
    # Build text well beyond max_chunk_chars with no strong semantic signal,
    # to exercise the "is_forced" branch.
    sentence = "Speaker A: This is a sentence about meditation and breath. "
    text = sentence * 200  # comfortably > max_chunk_chars default (6000)
    borders = chunk_transcript_semantic(
        text, refine_with_embeddings=False, min_chunk_chars=500, max_chunk_chars=2000,
    )
    assert len(borders) > 0
    # Borders must be strictly increasing and within text bounds.
    assert borders == sorted(borders)
    assert all(0 < b < len(text) for b in borders)


def test_chunker_with_embedding_refinement_does_not_crash():
    text = (
        "Speaker A: Let's talk about meditation and the breath today.\n\n"
        "Speaker A: The jhana states are deep concentration.\n\n"
        "Speaker B: Anyway, unrelated topic: the weather has been rainy lately.\n\n"
        "Speaker B: I've also been cooking a new recipe with lots of vegetables.\n\n"
    ) * 20
    borders = chunk_transcript_semantic(
        text,
        refine_with_embeddings=True,
        batch_embed_fn=_fake_embed_fn,
        min_chunk_chars=200,
        max_chunk_chars=1000,
    )
    assert isinstance(borders, list)
    assert borders == sorted(borders)


def test_chunker_empty_text_returns_empty_list():
    assert chunk_transcript_semantic("", refine_with_embeddings=False) == []
