"""
Tests for sanghabot.bot.discord_bot.format_typo_notice() -- the pure
message-formatting function behind the transparent, non-blocking "did you
mean...?" Discord notice. Deliberately does not touch any live discord.py
objects/connections; those are exercised manually/in production, not here.
"""
from sanghabot.bot.discord_bot import format_typo_notice
from sanghabot.models import TermSuggestion


def test_empty_suggestions_produce_empty_string():
    assert format_typo_notice([]) == ""


def test_single_typo_with_suggestion_is_formatted():
    notice = format_typo_notice([TermSuggestion(term="duka", suggestion="dukkha")])
    assert '"duka"' in notice
    assert "dukkha" in notice
    assert "heads up" in notice.lower()


def test_single_typo_without_suggestion_is_formatted():
    notice = format_typo_notice([TermSuggestion(term="xqzwplk", suggestion=None)])
    assert '"xqzwplk"' in notice
    assert "not found in the library" in notice.lower()


def test_multiple_suggestions_all_appear_as_separate_lines():
    suggestions = [
        TermSuggestion(term="duka", suggestion="dukkha"),
        TermSuggestion(term="jhanas", suggestion="jhana"),
        TermSuggestion(term="xqzwplk", suggestion=None),
    ]
    notice = format_typo_notice(suggestions)
    lines = notice.strip().split("\n")
    # 1 header line + 3 term lines.
    assert len(lines) == 4
    assert '"duka"' in notice and "dukkha" in notice
    assert '"jhanas"' in notice and "jhana" in notice
    assert '"xqzwplk"' in notice

    # Plural header wording used when there's more than one flagged term.
    assert "words" in notice.lower()


def test_notice_never_implies_the_search_was_blocked():
    notice = format_typo_notice([TermSuggestion(term="duka", suggestion="dukkha")])
    assert "anyway" in notice.lower()
