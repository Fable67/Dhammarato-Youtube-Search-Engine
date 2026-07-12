"""
Discord bot entrypoint.

Fixes vs. AI_Transcripts/discord_bot_channel.py:
  - The Discord token is read from config.settings (backed by .env), never
    hardcoded in source. The old token was committed to git; see
    REWRITE_PLAN.md Section 0 and Section 10 step 1 (rotate it).
  - CombinedSearchEngine is constructed exactly ONCE, in on_ready, and held
    as a module-level singleton for the process lifetime -- not
    reconstructed on every incoming message (the old code's
    `perform_search()` built a brand new CombinedSearchEngine per message,
    which reloaded a 45MB FAISS index and 55MB BM25 pickle from disk every
    single time; see search_engine.log for the resulting 8-14s latencies).

The UX (privacy-choice DM flow, thread creation, embed formatting) is
preserved as-is from the old bot -- it was a genuinely well-thought-out
feature for a sensitive-topic community bot and isn't part of what needed
fixing.
"""
from __future__ import annotations

import asyncio
import logging
import traceback

import discord

from config import settings
from sanghabot.embeddings.client import embed_query
from sanghabot.embeddings.legacy_compat import LEGACY_RUN_MARKER, legacy_embed_query
from sanghabot.engine import CombinedSearchEngine
from sanghabot.search.bm25 import BM25SearchEngine
from sanghabot.search.semantic import SemanticSearchEngine
from sanghabot.storage.db import Database

logger = logging.getLogger("sanghabot")
logging.basicConfig(level=settings.log_level)

# Module-level singleton, set exactly once in on_ready(). This is the
# actual fix for the old code's per-message reconstruction problem: loaded
# once, held for the process lifetime, never touched again except to call
# .search() (which is pure in-memory work -- no disk I/O in the hot path).
engine: CombinedSearchEngine | None = None
db: Database | None = None


def build_engine() -> CombinedSearchEngine:
    database = Database(settings.db_path)

    # The FAISS index must be queried with vectors from the SAME embedding
    # pipeline it was built from, or scores are meaningless (see
    # sanghabot/embeddings/legacy_compat.py). Until REWRITE_PLAN.md Section
    # 11's full re-embed happens and a new run is promoted to active, the
    # active run is the migrated legacy one (query expansion + 1024-dim
    # native embed + average-pool to 512) -- so we must use that exact
    # pipeline here, not a plain native embed_query() call. This check
    # makes the correct choice automatic once a new run IS promoted,
    # instead of silently embedding queries in the wrong vector space.
    active_run = database.get_active_embedding_run()
    if active_run is not None and active_run["run_id"] == LEGACY_RUN_MARKER:
        embed_fn = legacy_embed_query
        logger.info("Active embedding run is the migrated legacy run -- using legacy query-embedding pipeline.")
    else:
        embed_fn = lambda q: embed_query(q, dimension=settings.embedding_dim)  # noqa: E731
        logger.info("Active embedding run is native -- using standard query-embedding client.")

    semantic = SemanticSearchEngine(settings.faiss_index_path, database, embed_fn=embed_fn)
    bm25 = BM25SearchEngine(database, settings.bm25_index_path)
    return CombinedSearchEngine(
        semantic_engine=semantic,
        bm25_engine=bm25,
        top_k_per_engine=settings.search_top_k_per_engine,
        rrf_k_penalty=settings.rrf_k_penalty,
    )


class PrivacyChoiceView(discord.ui.View):
    """DM-only view letting the user choose how a private search is routed."""

    def __init__(self, bot: "SanghaBot", query: str, user_message: discord.Message):
        super().__init__(timeout=300)
        self.bot = bot
        self.query = query
        self.user_message = user_message

    @discord.ui.button(label="\U0001f512 Keep Private here", style=discord.ButtonStyle.primary)
    async def private_search(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content=f"Searching privately for: **{self.query}**...", view=None)
        async with self.user_message.channel.typing():
            await self.bot.perform_search(self.query, self.user_message, is_dm_private=True)

    @discord.ui.button(label="\U0001f4e2 Share Anonymously to Sangha", style=discord.ButtonStyle.success)
    async def public_search(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content=f"Preparing anonymous proxy search for: **{self.query}**...", view=None)
        async with self.user_message.channel.typing():
            await self.bot.perform_search(self.query, self.user_message, is_dm_proxy=True)


class SanghaBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

    async def on_ready(self):
        global engine
        logger.info("Building search engine (one-time, at startup) ...")
        engine = build_engine()
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logger.info(f"Monitoring Channel ID: {settings.channel_id}")

        channel = self.get_channel(settings.channel_id)
        if channel and isinstance(channel, discord.TextChannel):
            if channel.slowmode_delay == 0:
                logger.warning("Slow mode is NOT enabled in the target channel.")
            else:
                logger.info(f"Slow mode is active: {channel.slowmode_delay}s delay.")

    def split_text_for_embeds(self, text: str, max_len: int = 4000) -> list[str]:
        return [text[i:i + max_len] for i in range(0, len(text), max_len)]

    async def perform_search(self, query: str, message: discord.Message, is_dm_private=False, is_dm_proxy=False):
        try:
            if engine is None:
                await message.reply("Search engine is still starting up, please try again shortly.", delete_after=10)
                return

            results = await asyncio.to_thread(engine.search, query, 3)

            if not results:
                await message.reply(f"I couldn't find specific teachings for '{query}'. Maybe try different words?", delete_after=10)
                return

            post_target = None

            if is_dm_proxy:
                target_channel = self.get_channel(settings.channel_id)
                if not target_channel:
                    await message.reply("Sorry, I could not locate the public library channel to post your inquiry.")
                    return
                thread = await target_channel.create_thread(
                    name=f"Anonymous: {query[:15]}",
                    auto_archive_duration=60,
                    type=discord.ChannelType.public_thread,
                )
                await message.reply(f"\U0001f64f Your anonymous search has been posted: {thread.jump_url}")
                await thread.send(f"\U0001f338 **Anonymous Inquiry:** Here are the most relevant teachings for: **{query}**")
                post_target = thread
            elif is_dm_private:
                await message.channel.send(f"\U0001f64f Greetings. Here are the most relevant teachings for your private inquiry: **{query}**")
                post_target = message.channel
            else:
                thread = await message.create_thread(name=f"Search: {query[:20]}", auto_archive_duration=60)
                await thread.send(f"\U0001f64f Greetings. Here are the most relevant teachings for your inquiry: **{query}**")
                post_target = thread

            for i, res in enumerate(results):
                full_chunk_text = res.text.replace("<b>", "**").replace("</b>", "**")

                progression_percent = 0
                if res.video_length_chars and res.video_length_chars > 0 and res.start_char is not None:
                    progression_percent = int((res.start_char / res.video_length_chars) * 100)

                estimated_seconds = int((res.start_char or 0) / 17.0)
                hours = estimated_seconds // 3600
                minutes = (estimated_seconds % 3600) // 60
                secs = estimated_seconds % 60
                time_guess = f"{hours:02d}:{minutes:02d}:{secs:02d}"

                embed = discord.Embed(url=res.url, color=discord.Color.dark_gold())
                embed.add_field(name="Relevance", value=f"{res.percentage_score:.1f}%", inline=True)
                embed.add_field(name="Location", value=f"~{progression_percent}% into transcript (~{time_guess})", inline=True)

                await post_target.send(f"***URL***: {res.url}")
                await post_target.send(embed=embed)
                await post_target.send(
                    f"***BLOG URL***: You can find the full transcript + summary of the video here\n{res.blog_url}"
                )

                for block in self.split_text_for_embeds(full_chunk_text):
                    reading_box = discord.Embed(description=block, color=discord.Color.light_grey())
                    await post_target.send(embed=reading_box)

                if i < len(results) - 1:
                    divider = "\u2501" * 67
                    await post_target.send(divider)
                    await post_target.send(divider)

        except Exception:
            logger.error("Error during search:\n%s", traceback.format_exc())
            await message.reply("An error occurred while searching the library. Please try again later.", delete_after=10)

    async def on_message(self, message: discord.Message):
        if message.author == self.user:
            return

        is_dm = message.guild is None
        if not is_dm and message.channel.id != settings.channel_id:
            return
        if message.mentions or message.role_mentions or "@everyone" in message.content or "@here" in message.content:
            return

        query = message.content.strip()
        if not query:
            return

        if is_dm:
            view = PrivacyChoiceView(self, query, message)
            await message.reply("How would you like to perform this search?", view=view)
        else:
            async with message.channel.typing():
                await self.perform_search(query, message)


def main() -> None:
    if not settings.discord_token or settings.discord_token == "placeholder-not-set-yet":
        raise RuntimeError(
            "DISCORD_TOKEN is not set in .env. Refusing to start. "
            "See .env.example."
        )
    bot = SanghaBot()
    bot.run(settings.discord_token)


if __name__ == "__main__":
    main()
