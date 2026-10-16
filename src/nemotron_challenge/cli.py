"""Commands for setup checks, source data, and unfinished milestones."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any

from . import tasks
from .clients import ChatClient, ParseClient, RetrievalClient, ServiceError, WikiClient
from .config import ConfigurationError, Settings


def doctor(settings: Settings, *, live: bool) -> tuple[dict[str, Any], int]:
    checks = []
    for package in ("httpx", "pydantic", "python-dotenv", "lancedb", "pyarrow", "huggingface-hub"):
        try:
            version = importlib.metadata.version(package)
            checks.append({"name": package, "status": "ok", "version": version})
        except importlib.metadata.PackageNotFoundError:
            checks.append({"name": package, "status": "missing"})
    settings_present = {
        "wikipedia": bool(settings.wikipedia_api_url), "parse": bool(settings.parse_base_url),
        "retrieval": bool(settings.retrieval_base_url),
        "super": bool(settings.super_base_url and settings.super_model and settings.super_api_key),
    }
    if not live:
        checks += [{"name": name, "status": "not_tested", "configured": present}
                   for name, present in settings_present.items()]
    else:
        operations = {
            "wikipedia": (WikiClient, (settings.wikipedia_api_url,), "corpus"),
            "parse": (ParseClient, (settings.parse_base_url, settings.parse_api_key), "verify_model"),
            "retrieval": (RetrievalClient, (settings.retrieval_base_url, settings.retrieval_api_key), "verify_models"),
            "super": (ChatClient, (settings.super_base_url, settings.super_api_key, settings.super_model), "verify_model"),
        }
        for name, (client_type, args, operation) in operations.items():
            if not settings_present[name]:
                checks.append({"name": name, "status": "missing", "detail": "configure this resource in .env"})
                continue
            try:
                with client_type(*args) as client:
                    getattr(client, operation)()
                checks.append({"name": name, "status": "ok", "scope": "catalog/readiness; inference not tested"})
            except (ServiceError, ConfigurationError, ValueError) as exc:
                checks.append({"name": name, "status": "failed", "detail": str(exc)})
    failed = any(check["status"] in {"failed", "missing"} for check in checks)
    return {
        "mode": "live" if live else "offline", "checks": checks,
        "challenge_stages": "not implemented", "inference": "not tested",
    }, int(failed)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="nemotron-challenge", description="Nemotron Agentic AI Challenge scaffold")
    sub = result.add_subparsers(dest="command", required=True)
    check = sub.add_parser("doctor", help="check local setup or remote catalogs")
    mode = check.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="check packages without network access")
    mode.add_argument("--live", action="store_true", help="check resource catalogs and readiness; does not run inference")
    sub.add_parser("stages", help="list unfinished challenge milestones")
    run = sub.add_parser("run", help="run a team milestone")
    run.add_argument("stage", choices=tasks.STAGES)
    run.add_argument("--work-dir", type=Path, default=Path("artifacts"))
    data = sub.add_parser("data", help="inspect or fetch the pinned source scans")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    data_sub.add_parser("show")
    fetch = data_sub.add_parser("fetch")
    fetch.add_argument("--output-dir", type=Path, default=Path("data/raw/britannica"))
    fetch.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "stages":
            result = [{"stage": stage, "status": "TODO", "guide": f"docs/milestones.md#{stage}"} for stage in tasks.STAGES]
        elif args.command == "data":
            from .data import fetch_sample, load_manifest

            result = load_manifest() if args.data_command == "show" else fetch_sample(args.output_dir, args.cache_dir)
        else:
            settings = Settings.from_env()
            if args.command == "doctor":
                result, status = doctor(settings, live=args.live)
                print(json.dumps(result, indent=2))
                return status
            result = getattr(tasks, args.stage)(settings, args.work_dir)
        print(json.dumps(result, indent=2))
        return 0
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (ConfigurationError, ServiceError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
