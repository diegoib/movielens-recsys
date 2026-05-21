FROM public.ecr.aws/deep-learning-containers/pytorch-training-arm64:2.7.0-gpu-py312-cu128-ubuntu22.04-ec2-v1.49

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl make build-essential gnupg apt-transport-https ca-certificates openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Node.js LTS (required for Claude Code CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Google Cloud CLI (arm64-compatible via apt)
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
       > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install uv for dependency-groups support
RUN pip install --upgrade pip uv

# Keep venv outside /workspace so the bind-mount doesn't shadow it
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml ./
# torch/torchvision already in base image — install only project extras
RUN uv sync --group data --group train --group dev --group test
