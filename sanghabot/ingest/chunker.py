"""
Semantic transcript chunking.

Ported from AI_Transcripts/chunking.py largely unchanged -- this was
identified as the best-written, most thoughtfully-designed file in the old
codebase (multi-signal boundary detection: lexical overlap + structural
gaps + embedding similarity + local non-max suppression), and the rewrite
plan explicitly calls for keeping it rather than rewriting it from
scratch. Only cosmetic/typing cleanup applied; the algorithm is unchanged.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable


def chunk_transcript_semantic(
    transcript_text: str,
    # --- lexical chunking (based on sentence count) ---
    window_size: int = 16,
    overlap_threshold: float = 0.3,
    structural_gap_threshold: int = 1,
    # --- chunk size control (based on characters) ---
    min_chunk_chars: int = 1500,
    max_chunk_chars: int = 6000,
    # --- embedding refinement ---
    refine_with_embeddings: bool = True,
    batch_embed_fn: Callable[[list[str]], list] | None = None,
    embedding_window_size: int = 16,
    embedding_batch_size: int = 32,
    # --- local boundary competition ---
    boundary_competition_window: int = 400,
    # --- boundary confidence policy ---
    hard_boundary_confidence: float = 0.65,
    # --- confidence rating weights ---
    lexical_weight: float = 0.35,
    embedding_weight: float = 0.55,
    structural_weight: float = 0.1,
    stopwords: set | None = None,
    debug: bool = False,
) -> list[int]:
    """
    Semantic chunking adapted for structured (speaker-labeled) transcript
    text. Returns a list of character indices where the text should be cut.
    """
    import numpy as np

    if stopwords is None:
        stopwords = set()

    if refine_with_embeddings and batch_embed_fn is None:
        raise ValueError("batch_embed_fn must be provided if refine_with_embeddings=True")

    # ------------------------------------------------------------------
    # Step 1: Segmentation (sentence & speaker aware)
    # ------------------------------------------------------------------
    segments = []
    segment_end_indices = []

    pattern = r'\S.*?(?:[.!?]+(?:\s|\n|$)|(?:\n\n+|$))'
    for match in re.finditer(pattern, transcript_text, re.DOTALL):
        seg = match.group().strip()
        if seg:
            segments.append(seg)
            segment_end_indices.append(match.end())

    if not segments:
        return []

    def tokenize(text: str) -> list[str]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [t for t in tokens if t not in stopwords]

    tokenized = [tokenize(seg) for seg in segments]

    if debug:
        print(f"[DEBUG] Segmented transcript into {len(segments)} blocks.")

    # ------------------------------------------------------------------
    # Step 2: Lexical overlap
    # ------------------------------------------------------------------
    def lexical_overlap(left_tokens, right_tokens) -> float:
        if not left_tokens or not right_tokens:
            return 0.0
        left = Counter(left_tokens)
        right = Counter(right_tokens)
        intersection = sum((left & right).values())
        return intersection / max(sum(left.values()), sum(right.values()))

    candidate_boundaries: dict[int, dict] = {}

    for i in range(1, len(segments)):
        left_range = range(max(0, i - window_size), i)
        right_range = range(i, min(len(segments), i + window_size))

        left_tokens = [t for j in left_range for t in tokenized[j]]
        right_tokens = [t for j in right_range for t in tokenized[j]]

        overlap = lexical_overlap(left_tokens, right_tokens)

        prev_end = segment_end_indices[i - 1]
        curr_start = transcript_text.find(segments[i], prev_end)
        gap_text = transcript_text[prev_end:curr_start]

        structural_gap = ["\n"] * structural_gap_threshold
        has_structural_gap = 1.0 if "".join(structural_gap) in gap_text else 0.0

        if overlap > overlap_threshold or has_structural_gap > 0:
            candidate_boundaries[i] = {
                "lexical_overlap": overlap,
                "structural_gap": has_structural_gap,
                "embedding_similarity": None,
                "char_index": prev_end,
            }
            if debug:
                print(f"[DEBUG] Candidate boundary at segment {i} (char {prev_end}): "
                      f"overlap={overlap:.3f}, structural_gap={has_structural_gap}")

    # ------------------------------------------------------------------
    # Step 3: Embedding refinement (batched)
    # ------------------------------------------------------------------
    if refine_with_embeddings and candidate_boundaries:
        if debug:
            print(f"[DEBUG] Refining {len(candidate_boundaries)} boundaries with embeddings.")

        boundary_texts = []
        boundary_keys = []

        for i in candidate_boundaries.keys():
            left_s = max(0, i - embedding_window_size)
            right_e = min(len(segments), i + embedding_window_size)

            boundary_texts.append(" ".join(segments[left_s:i]))
            boundary_keys.append((i, "left"))
            boundary_texts.append(" ".join(segments[i:right_e]))
            boundary_keys.append((i, "right"))

        embeddings = []
        for start in range(0, len(boundary_texts), embedding_batch_size):
            batch = boundary_texts[start:start + embedding_batch_size]
            embeddings.extend(batch_embed_fn(batch))

        def cosine(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        boundary_vectors: dict[int, dict] = {}
        for (idx, side), vec in zip(boundary_keys, embeddings):
            boundary_vectors.setdefault(idx, {})[side] = vec

        for i, vecs in boundary_vectors.items():
            sim = cosine(vecs["left"], vecs["right"]) if "left" in vecs and "right" in vecs else 1.0
            candidate_boundaries[i]["embedding_similarity"] = sim
            if debug:
                print(f"[DEBUG] Boundary {i}: embedding_similarity={sim:.3f}")

    # ------------------------------------------------------------------
    # Step 4: Confidence scoring
    # ------------------------------------------------------------------
    def compute_confidence(info: dict) -> float:
        lexical_component = 1.0 - info["lexical_overlap"]
        embedding_component = 1.0 - info["embedding_similarity"] if info["embedding_similarity"] is not None else 0.0
        structure_component = info["structural_gap"]

        confidence = (
            lexical_weight * lexical_component
            + embedding_weight * embedding_component
            + structural_weight * structure_component
        )
        return max(0.0, min(1.0, confidence))

    for i, info in candidate_boundaries.items():
        info["confidence"] = compute_confidence(info)
        if debug:
            print(f"[DEBUG] Boundary {i} confidence={info['confidence']:.3f} "
                  f"(lex={1.0 - info['lexical_overlap']:.3f}, "
                  f"emb={1.0 - (info['embedding_similarity'] or 0.0):.3f}, "
                  f"struct={info['structural_gap']:.1f})")

    # ------------------------------------------------------------------
    # Step 5: Local boundary competition (non-max suppression)
    # ------------------------------------------------------------------
    if candidate_boundaries:
        items = sorted(candidate_boundaries.items(), key=lambda x: x[1]["char_index"])
        suppressed = set()
        surviving = {}

        for i, (idx_i, info_i) in enumerate(items):
            if idx_i in suppressed:
                continue

            for j in range(i + 1, len(items)):
                idx_j, info_j = items[j]
                if info_j["char_index"] - info_i["char_index"] > boundary_competition_window:
                    break

                if info_j["confidence"] > info_i["confidence"]:
                    suppressed.add(idx_i)
                    if debug:
                        print(f"[DEBUG] Suppressing boundary at char {info_i['char_index']} "
                              f"(conf={info_i['confidence']:.3f}) in favor of char "
                              f"{info_j['char_index']} (conf={info_j['confidence']:.3f})")
                    break
                else:
                    suppressed.add(idx_j)
                    if debug:
                        print(f"[DEBUG] Suppressing boundary at char {info_j['char_index']} "
                              f"(conf={info_j['confidence']:.3f}) in favor of char "
                              f"{info_i['char_index']} (conf={info_i['confidence']:.3f})")

            if idx_i not in suppressed:
                surviving[idx_i] = info_i
        candidate_boundaries = surviving

    # ------------------------------------------------------------------
    # Step 6: Build final borders
    # ------------------------------------------------------------------
    final_borders = []
    last_border_char = 0

    for i in range(1, len(segments)):
        current_char_count = segment_end_indices[i]
        chars_since_last = current_char_count - last_border_char

        info = candidate_boundaries.get(i)
        confidence = info["confidence"] if info else 0.0

        is_hard = confidence >= hard_boundary_confidence and chars_since_last >= min_chunk_chars
        is_forced = chars_since_last >= max_chunk_chars

        if is_hard or is_forced:
            cut_point = segment_end_indices[i - 1]

            chars_remaining = len(transcript_text) - cut_point
            if 0 < chars_remaining < min_chunk_chars:
                if debug:
                    print(f"[DEBUG] Skipping split at char {cut_point} to avoid tiny final "
                          f"chunk ({chars_remaining} chars remaining).")
                continue

            final_borders.append(cut_point)
            last_border_char = cut_point
            if debug:
                reason = "FORCED" if is_forced else "HARD"
                print(f"[DEBUG] {reason} split at char {cut_point} "
                      f"(confidence={confidence:.3f}, chars_since_last={chars_since_last})")

    if debug:
        print(f"[DEBUG] Generated {len(final_borders)} chunk borders.")

    return final_borders
