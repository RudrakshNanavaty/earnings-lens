# Use a standard Python image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies using uv
# We use --frozen if uv.lock is present, but for HF Space we might want to be flexible
RUN uv sync --no-cache

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# Expose the port Gradio will run on
EXPOSE 8000

# Default command to run the Gradio app
CMD ["python", "app.py"]
