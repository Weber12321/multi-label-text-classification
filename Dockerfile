ARG PYTHON_VERSION
ARG PYTORCH_VERSION

FROM python:${PYTHON_VERSION}

ENV PYTHONUNBUFFERED 1

WORKDIR /
COPY pyproject.toml poetry.lock ./
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    curl \
 && pip install --no-cache-dir pip==22.0.4 \
 && curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python - \
 && PATH="${PATH}:$HOME/.poetry/bin" \
 && poetry config virtualenvs.create false \
 && poetry install --no-dev --no-root \
 && poetry add psycopg2-binary \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN rm -rf /var/lib/apt/lists/*

COPY . .
EXPOSE 8000 8501

CMD ['/start.sh', '${PYTORCH_VERSION}']
