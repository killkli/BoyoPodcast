FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy pyproject.toml and poetry.lock if available
COPY src/pyproject.toml .
COPY src/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/main.py .


# Expose the Gradio port
EXPOSE 7860

# Run the application
CMD ["python", "main.py"]
