# syntax=docker/dockerfile:1
#
# This image intentionally does NOT bake the application source code
# (sanghabot/, config.py, scripts/) into the image with COPY. Only the
# Python virtualenv (dependencies) is built in. The actual code, along with
# data/index/ and .env, is expected to be bind-mounted into the container
# at `docker run` time (see docker-compose.yml, or the `docker run -v ...`
# examples in README.md).
#
# Why: dependencies (requirements.txt) change rarely and are relatively
# slow to install, so baking them into the image and rebuilding only when
# they change is the expensive/slow part worth caching. Application code
# changes constantly during development -- if it were COPYed in, every
# code edit would require a full image rebuild just to test it. With code
# bind-mounted instead, editing a .py file on the host is reflected inside
# the container immediately; no rebuild needed, just a container restart
# to pick up the change (Python doesn't hot-reload modules on its own).
#
# Trade-off to be aware of: this means the image is NOT a fully
# self-contained artifact -- it depends on the source code being mounted
# alongside it at runtime. That's a deliberate choice for a
# single-maintainer bot with a fast local dev loop. If you ever want a
# "ship one immutable image anywhere" deployment instead, add a COPY step
# back in for sanghabot/, config.py, and scripts/ before the USER line
# below, and drop the corresponding bind mounts from docker-compose.yml.

FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ---------------------------------------------------------------------------
# Runtime stage: slim image with only the Python virtualenv baked in.
# Application code, data/index/, and .env are all bind-mounted at `docker
# run`/`docker compose up` time -- see docker-compose.yml.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# libgomp1 is required at runtime by faiss-cpu (OpenMP support).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Nothing is COPYed here on purpose -- see the block comment at the top of
# this file. /app/sanghabot, /app/config.py, /app/scripts, /app/data, and
# /app/.env are all expected to appear via bind mounts at container start.
RUN mkdir -p /app/data/index && chown -R appuser:appuser /app

USER appuser

CMD ["python", "-m", "sanghabot.bot.discord_bot"]
