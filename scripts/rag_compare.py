#!/usr/bin/env python
"""
Compare RAG model quality across 5 OpenRouter models on the same 6 Dhamma prompts.

Builds the search engine once, then queries each of 6 realistic Dhamma teaching
prompts with each of 5 models using identical retrieved context (top_k=5) to
isolate generation-quality differences. Writes a detailed markdown comparison
report to docs/rag-comparison.md.

Usage:
    python scripts/rag_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REWRITE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REWRITE_ROOT))

from config import settings  # noqa: E402
from sanghabot.rag.generator import (  # noqa: E402
    PermanentGenerationError,
    TransientGenerationError,
)
from sanghabot.rag.pipeline import RagPipeline, build_search_engine  # noqa: E402

# ============================================================================
# COMPARISON CONFIGURATION
# ============================================================================

COMPARISON_PROMPTS = [
    # 1. Mainstream/common teaching: answerable from typical corpus content
    "What is Right Effort in the Noble Eightfold Path, and how does it relate to the Four Great Efforts?",
    
    # 2. More specific/niche: sparse or partial source coverage expected
    "Explain the relationship between concentration (samadhi) and jhana, and how do you know you are in a jhana?",
    
    # 3. Implausibly covered / likely not in corpus: tests "insufficient sources" instruction
    "What is the quantum physics interpretation of consciousness in Buddhist meditation practice?",
    
    # 4. Open-ended practice question: tempts generic advice, needs source grounding
    "I struggle with anger and frustration in my daily life. How should I practice with these emotions?",
    
    # 5. Well-known Dhamma concept from different angle: dependent origination
    "Can you explain dependent origination (paticcasamuppada) and how understanding it leads to liberation?",
    
    # 6. Another open-ended practice question: dealing with grief/loss
    "I'm dealing with the loss of a loved one. How does Buddhist practice help with grief, and what should I do?",
]

COMPARISON_MODELS = [
    "google/gemini-2.5-flash",
    "google/gemini-2.0-flash-001",
    "mistralai/mistral-small-3.2-24b-instruct",
    "openai/gpt-4o-mini",
    "anthropic/claude-haiku-4.5",
]


# ============================================================================
# MAIN COMPARISON LOGIC
# ============================================================================

def format_citations_for_report(citations, title_prefix="") -> str:
    """
    Format citations as a bulleted markdown list with title + relevance %.
    
    Args:
        citations: List of Citation objects from RagAnswer.
        title_prefix: Optional string to prepend to each title.
    
    Returns:
        Markdown-formatted bullet list.
    """
    if not citations:
        return "_(No sources retrieved)_"
    
    lines = []
    for citation in citations:
        title = f"{title_prefix}{citation.title}" if title_prefix else citation.title
        lines.append(f"- {title} ({citation.percentage_score:.1f}%)")
    return "\n".join(lines)


def run_comparison():
    """
    Run the full comparison: build engine, query each (prompt, model) pair,
    write markdown report, and print progress.
    """
    print("=" * 70)
    print("RAG Model Comparison")
    print("=" * 70)
    print()
    
    # Build search engine once
    print("Building search engine (this may take a moment)...")
    engine = build_search_engine()
    pipeline = RagPipeline(engine)
    print("✓ Search engine ready")
    print()
    
    # Prepare results table: results[prompt_idx][model_idx] = (answer_text, citations, error_or_none)
    num_prompts = len(COMPARISON_PROMPTS)
    num_models = len(COMPARISON_MODELS)
    results: list[list[tuple[str, list, str | None]]] = [
        [(None, None, None) for _ in range(num_models)]
        for _ in range(num_prompts)
    ]
    
    total = num_prompts * num_models
    call_idx = 0
    
    # Run all (prompt, model) pairs
    for prompt_idx, prompt in enumerate(COMPARISON_PROMPTS):
        for model_idx, model in enumerate(COMPARISON_MODELS):
            call_idx += 1
            progress = f"[{call_idx}/{total}]"
            prompt_preview = prompt[:60] + "..." if len(prompt) > 60 else prompt
            print(f"{progress} {model:40s} :: {prompt_preview}")
            
            try:
                answer = pipeline.answer(
                    query=prompt,
                    top_k=5,
                    model=model,
                    temperature=settings.rag_temperature,
                    top_p=settings.rag_top_p,
                )
                results[prompt_idx][model_idx] = (answer.answer, answer.citations, None)
            except PermanentGenerationError as e:
                error_msg = f"[PERMANENT ERROR] {str(e)[:200]}"
                results[prompt_idx][model_idx] = (None, [], error_msg)
                print(f"  → {error_msg}")
            except TransientGenerationError as e:
                error_msg = f"[TRANSIENT ERROR - retried & failed] {str(e)[:200]}"
                results[prompt_idx][model_idx] = (None, [], error_msg)
                print(f"  → {error_msg}")
    
    print()
    print("=" * 70)
    print("Writing report to docs/rag-comparison.md...")
    print("=" * 70)
    
    # Write markdown report
    docs_dir = REWRITE_ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    report_path = docs_dir / "rag-comparison.md"
    
    report_lines = [
        "# RAG Model Comparison",
        "",
        "Generated by `scripts/rag_compare.py`. Compares 5 OpenRouter models on 6 realistic",
        "Dhamma queries, using identical retrieved context per query (top_k=5,",
        f"temperature={settings.rag_temperature}, top_p={settings.rag_top_p}) to isolate",
        "generation-quality differences.",
        "",
    ]
    
    # One section per prompt
    for prompt_idx, prompt in enumerate(COMPARISON_PROMPTS):
        report_lines.append(f"## Prompt {prompt_idx + 1}: {prompt}")
        report_lines.append("")
        
        # One subsection per model
        for model_idx, model in enumerate(COMPARISON_MODELS):
            answer_text, citations, error = results[prompt_idx][model_idx]
            
            report_lines.append(f"### {model}")
            report_lines.append("")
            
            if error:
                report_lines.append(f"{error}")
            else:
                report_lines.append(answer_text or "(No answer generated)")
            
            report_lines.append("")
            
            if error:
                report_lines.append(f"**Sources retrieved:** Error (no sources)")
            else:
                num_citations = len(citations)
                report_lines.append(f"**Sources retrieved:** {num_citations}")
                report_lines.append(format_citations_for_report(citations))
            
            report_lines.append("")
    
    report_content = "\n".join(report_lines)
    report_path.write_text(report_content)
    print(f"✓ Report written to {report_path}")
    print()
    print("Done!")


if __name__ == "__main__":
    run_comparison()
