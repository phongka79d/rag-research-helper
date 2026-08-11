"""Safely recover or create local Neo4j credentials, then verify connectivity."""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, set_key
from neo4j import GraphDatabase

ENV_PATH = Path(".env")
KNOWN_NEO4J_CONTAINERS = (
    "rag_research_neo4j",
    "rag-research-helper-neo4j-1",
)
NEO4J_DATA_VOLUME = "rag-research-helper_neo4j_data"
DEFAULTS = {
    "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "OPENAI_API_KEY": "",
    "OPENAI_MODEL": "gpt-5-nano",
    "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
    "OPENAI_EMBEDDING_DIM": "1536",
    "QDRANT_URL": "http://localhost:6333",
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USER": "neo4j",
}


def run_docker(*args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Docker is required to prepare the local databases.") from error


def inspect_container(name: str) -> dict[str, Any] | None:
    result = run_docker("inspect", name)
    if result.returncode:
        return None
    return json.loads(result.stdout)[0]


def find_existing_neo4j() -> dict[str, Any] | None:
    for name in KNOWN_NEO4J_CONTAINERS:
        container = inspect_container(name)
        if container is not None:
            return container
    return None


def volume_has_database_files(volume_name: str) -> bool:
    if run_docker("volume", "inspect", volume_name).returncode:
        return False
    result = run_docker(
        "run",
        "--rm",
        "-v",
        f"{volume_name}:/data",
        "alpine:3.20",
        "sh",
        "-c",
        "test -d /data/databases && test -n \"$(ls -A /data/databases 2>/dev/null)\"",
    )
    return result.returncode == 0


def has_existing_data(container: dict[str, Any] | None) -> bool:
    if container:
        data_mount = next(
            (
                mount
                for mount in container.get("Mounts", [])
                if mount.get("Destination") == "/data"
            ),
            None,
        )
        if data_mount and data_mount.get("Type") == "volume":
            return volume_has_database_files(data_mount.get("Name", ""))
        if data_mount and data_mount.get("Type") == "bind":
            databases = Path(data_mount["Source"]) / "databases"
            return databases.is_dir() and any(databases.iterdir())
    return volume_has_database_files(NEO4J_DATA_VOLUME)


def password_from_container(container: dict[str, Any]) -> str:
    environment = container.get("Config", {}).get("Env", [])
    auth = next((item[11:] for item in environment if item.startswith("NEO4J_AUTH=")), "")
    user, separator, password = auth.partition("/")
    if not separator or not user or not password or password.lower() == "none":
        return ""
    return password


def set_env_value(key: str, value: str) -> None:
    ENV_PATH.touch(exist_ok=True)
    set_key(str(ENV_PATH), key, value, quote_mode="never")


def ensure_defaults() -> dict[str, str | None]:
    values = dotenv_values(ENV_PATH)
    for key, value in DEFAULTS.items():
        if values.get(key) is None:
            set_env_value(key, value)
    return dotenv_values(ENV_PATH)


def start_databases() -> None:
    result = run_docker("compose", "up", "-d", "qdrant", "neo4j")
    if result.returncode:
        raise RuntimeError("Docker Compose could not start Qdrant and Neo4j.")


def verify_connection(uri: str, user: str, password: str) -> None:
    deadline = time.monotonic() + 30
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            driver.verify_connectivity()
            return
        except Exception as error:
            last_error = error
            time.sleep(1)
        finally:
            driver.close()
    raise RuntimeError(
        "Neo4j credential could not be verified. Check the running database and local .env."
    ) from last_error


def existing_data_error() -> RuntimeError:
    return RuntimeError(
        "Existing Neo4j data detected but password cannot be verified. "
        "Provide/reset the Neo4j password, then rerun setup."
    )


def prepare_environment(start: bool = False) -> None:
    values = ensure_defaults()
    password = values.get("NEO4J_PASSWORD") or ""
    uri = values.get("NEO4J_URI") or DEFAULTS["NEO4J_URI"]
    user = values.get("NEO4J_USER") or DEFAULTS["NEO4J_USER"]
    container = find_existing_neo4j()
    data_exists = has_existing_data(container)

    if password:
        if start:
            start_databases()
        verify_connection(uri, user, password)
        print("Neo4j connectivity verified; credential is stored only in .env.")
        return

    if data_exists:
        if container is None:
            raise existing_data_error()
        candidate = password_from_container(container)
        if not candidate:
            raise existing_data_error()
        try:
            verify_connection(uri, user, candidate)
        except RuntimeError as error:
            raise existing_data_error() from error
        set_env_value("NEO4J_PASSWORD", candidate)
        print("Neo4j connectivity verified; credential is stored only in .env.")
        return

    password = secrets.token_urlsafe(18)
    set_env_value("NEO4J_PASSWORD", password)
    if start:
        start_databases()
    verify_connection(uri, user, password)
    print("Neo4j connectivity verified; credential is stored only in .env.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        action="store_true",
        help="start the project Qdrant and Neo4j containers before verification",
    )
    args = parser.parse_args()
    try:
        prepare_environment(start=args.start)
    except RuntimeError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
