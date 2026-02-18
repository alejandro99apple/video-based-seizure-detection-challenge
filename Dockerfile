# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

# prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# keep python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# install dependencies first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy application code
COPY . /app

# create a non-privileged user that the app will run under
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser \
    && mkdir -p /data /output \
    && chown -R appuser:appuser /app /data /output

# switch to the non-privileged user to run the application
USER appuser

# create data volumes
VOLUME ["/data"]
VOLUME ["/output"]

# define environment variables
ENV INPUT=""
ENV OUTPUT=""

# run the application
ENTRYPOINT ["python", "-m", "main"]
