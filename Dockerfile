FROM python:3.8.10-slim-buster

ENV PYTHONUNBUFFERED 1

CMD ["python3"]

COPY pyproject.toml poetry.lock

COPY . .

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    netcat=1.* \
    libpq-dev=11.* \
    unixodbc-dev=2.* \
    g++=4:* \
    curl \
 && pip install --no-cache-dir pip==22.0.4 \
 && curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python - \
 && PATH="${PATH}:$HOME/.poetry/bin" \
 && poetry config virtualenvs.create false \
 && poetry install --no-dev --no-root \
 && poetry add psycopg2-binary \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

CMD celery -A app worker -l INFO -P solo -Q training_queue
