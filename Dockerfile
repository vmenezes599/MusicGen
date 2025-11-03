FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Install system dependencies
RUN apt update && \
    apt install -y curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . ./

# Create user and set permissions
RUN groupadd -g 1000 appgroup && useradd -u 1000 -g appgroup -m appuser
RUN chown -R appuser:appgroup /app

# Create cache directories with proper permissions
RUN mkdir -p /tmp/triton_cache && chmod 777 /tmp/triton_cache
RUN mkdir -p /.cache && chmod 777 /.cache

RUN chmod +x /app/health-check.sh

USER appuser

# Expose port
EXPOSE 8189
