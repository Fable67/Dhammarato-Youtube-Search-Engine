"""
Tests for sanghabot.bot.discord_bot.SanghaBot.split_text_for_embeds() --
the marker-aware text splitter used to break long transcript chunks across
multiple Discord embeds without severing a highlighted "**term**" span or
leaving an orphaned/unbalanced "**" marker in one block.

Constructs a SanghaBot instance directly (no real Discord connection is
made -- discord.Client.__init__ only sets up local state) so the method
can be exercised in isolation.
"""
import discord
import pytest

from sanghabot.bot.discord_bot import SanghaBot


@pytest.fixture(scope="module")
def bot():
    intents = discord.Intents.default()
    intents.message_content = True
    return SanghaBot()


def _assert_balanced_markers(blocks: list[str]):
    for block in blocks:
        assert block.count("**") % 2 == 0, f"unbalanced ** markers in block: {block!r}"


def test_short_text_returns_single_block(bot):
    text = "A short transcript chunk."
    blocks = bot.split_text_for_embeds(text, max_len=4000)
    assert blocks == [text]


def test_text_exactly_at_limit_returns_single_block(bot):
    text = "x" * 4000
    blocks = bot.split_text_for_embeds(text, max_len=4000)
    assert blocks == [text]


def test_long_plain_text_splits_into_multiple_blocks(bot):
    text = "word " * 2000  # 10000 chars, no markers
    blocks = bot.split_text_for_embeds(text, max_len=4000)
    assert len(blocks) == 3
    for block in blocks[:-1]:
        assert len(block) <= 4000
    # No block should end or start mid-word (should end/start at whitespace
    # or a word boundary produced by rstrip/lstrip of the split point).
    assert "".join(blocks).replace(" ", "") == text.replace(" ", "")


def test_marker_straddling_boundary_stays_atomic(bot):
    """
    Regression test for the exact bug found during investigation: a
    "**bold**" span landing right at what would otherwise be a 4000-char
    cut point must never be severed, and must never leave an orphaned
    "**" in one block with its pair in another.
    """
    text = "x" * 3998 + "**important**" + "y" * 20
    blocks = bot.split_text_for_embeds(text, max_len=4000)

    _assert_balanced_markers(blocks)
    # The marker must appear whole, in exactly one block.
    assert sum(1 for b in blocks if "**important**" in b) == 1
    # Reconstructing the blocks must reproduce the original text exactly
    # (no characters dropped/duplicated), since this text has no separator
    # whitespace to lstrip away at the join point.
    assert "".join(blocks) == text


def test_many_scattered_markers_all_remain_balanced(bot):
    text = "The teaching of **dukkha** is central. " * 200
    blocks = bot.split_text_for_embeds(text, max_len=1000)
    assert len(blocks) > 1
    _assert_balanced_markers(blocks)
    for block in blocks:
        assert len(block) <= 1000


def test_single_marker_longer_than_max_len_kept_whole(bot):
    # Pathological case: a highlighted phrase so long it alone exceeds
    # max_len. There's no safe way to split inside "**...**", so it must
    # be kept whole even though it overflows the nominal limit.
    long_phrase = "word " * 50
    text = f"**{long_phrase.strip()}**"
    blocks = bot.split_text_for_embeds(text, max_len=50)
    _assert_balanced_markers(blocks)
    assert any(text in b or b in text for b in blocks)
    assert "".join(blocks) == text


def test_no_markers_behaves_like_plain_fixed_width_split_content(bot):
    # Without any "**" markers, output should still reconstruct the
    # original content (modulo whitespace at split points) and respect
    # the max_len budget per block.
    text = "abcdefghij " * 500
    blocks = bot.split_text_for_embeds(text, max_len=200)
    for block in blocks:
        assert len(block) <= 200
    assert "".join(blocks).replace(" ", "") == text.replace(" ", "")


def test_empty_text_returns_single_empty_block(bot):
    assert bot.split_text_for_embeds("", max_len=4000) == [""]
