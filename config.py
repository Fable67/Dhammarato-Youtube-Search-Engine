"""
Single source of configuration truth for the rewrite.

Replaces the old codebase's pattern of scattering hardcoded paths/secrets
across three different files (discord_bot_channel.py, chunk_videos.py,
embed_videos.py each re-read ".openrouterapikey" independently; the Discord
token was hardcoded directly in discord_bot_channel.py and committed to git).

Usage:
    from config import settings
    settings.discord_token
    settings.db_path
"""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Root of the rewrite project (this file's directory).
PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Secrets / bot identity ---------------------------------------
    # No defaults for secrets on purpose: missing them should fail loudly
    # at startup, not silently fall back to something wrong.
    discord_token: str = Field(default="")
    guild_id: int = Field(default=0)
    channel_id: int = Field(default=0)
    openrouter_api_key: str = Field(default="")

    # --- Embedding model -------------------------------------------------
    embedding_model: str = "qwen/qwen3-embedding-4b"
    # New default going forward is 1024 (see REWRITE_PLAN.md Appendix C).
    # Existing legacy data was embedded at 512 dims via a lossy averaging hack;
    # that legacy run is migrated as-is and flagged in embedding_runs, not
    # silently "fixed" to this default.
    embedding_dim: int = 1024

    # --- Storage paths ----------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "index" / "sanghabot.db"

    @property
    def faiss_index_path(self) -> Path:
        return self.data_dir / "index" / "faiss.index"

    @property
    def bm25_index_path(self) -> Path:
        return self.data_dir / "index" / "bm25.pkl"

    @property
    def raw_blogs_dir(self) -> Path:
        return self.data_dir / "raw" / "blogs"

    # --- Search behavior ----------------------------------------------
    # Old code hardcoded 1000 per engine before RRF trimmed to top_k.
    # Default here is lower and is explicitly flagged as needing a profiling
    # pass (see REWRITE_PLAN.md Appendix B) before being trusted blindly.
    search_top_k_per_engine: int = 100
    search_final_k: int = 3
    rrf_k_penalty: int = 60

    # --- Misc -------------------------------------------------------------
    log_level: str = "INFO"


settings = Settings()
