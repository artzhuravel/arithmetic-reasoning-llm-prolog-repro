FROM vllm/vllm-openai:v0.12.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers

WORKDIR /workspace

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    swi-prolog \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.container.txt /workspace/requirements.container.txt

# Keep the official vLLM image's validated Python/PyTorch/Transformers/vLLM stack
# and layer only the repo-specific training/runtime dependencies on top.
RUN uv pip install --system -r /workspace/requirements.container.txt

ENTRYPOINT []
CMD ["bash"]
