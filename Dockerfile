FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY options_mas ./options_mas
COPY config ./config
RUN pip install --no-cache-dir .
EXPOSE 8000
# Default: synthetic-data replay dashboard (clearly labelled SYNTHETIC). Live trading stays disabled.
CMD ["python", "-m", "options_mas.cli", "dashboard", "--synthetic", "--host", "0.0.0.0", "--port", "8000"]
