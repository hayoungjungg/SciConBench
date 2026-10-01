#!/usr/bin/env python3
"""Stage 9 backfill: query newly added models on core + every released rolling cohort.

Query only (no atomic facts / precision / recall). Uses the normal
task_run_queries path, so pending DOIs follow the evaluate-once policy and a
rerun resumes where the previous one stopped. Models in different providers
run as concurrent lanes inside one process.

Credentials (this script only; the monthly pipeline is unchanged):
  openai  gpt-6.1-sol         pipeline default (SALAME_*)
  claude  claude-opus-5-5     SPI_HAYOUNG_DASHBOARD_AZURE_OPENAI_KEY, via the
                              resource's Foundry /anthropic endpoint
  gemini  gemini-3.8-flash    pipeline default (Vertex env)
  azure   DeepSeek-V4.1-Flash pipeline default (azure_openai_key_models)

--key-env / --url-env replace the openai/azure credentials, or (with
--providers claude alone) the Claude resource, with another Azure key/base-URL
env-var pair; --model replaces the model name. --offset K --limit N restricts
the run to eval DOIs K+1..K+N.

Examples:
  python scripts/bulk_query_new_models.py --providers openai,claude --smoke
  python scripts/bulk_query_new_models.py --providers gemini,azure
  python scripts/bulk_query_new_models.py --providers azure \\
      --key-env SPI_HAYOUNG_DASHBOARD_AZURE_OPENAI_KEY --url-env SPI_HAYOUNG_DASHBOARD_OPENAI_BASE_URL
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scicon-track"), str(ROOT / "scripts")]

os.environ.setdefault(
    "PREFECT_HOME", f"/tmp/{os.environ.get('USER', 'user')}-prefect"
)

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import run_workflow
from config import query_cfg
from data_collection.utils import previous_year_month
from db.utils import get_eval_dois
from run_workflow import _env_api_key, _pending_dois_for_model, task_run_queries

NEW_MODELS: dict[str, str] = {
    "openai": "gpt-6.1-sol",
    "claude": "claude-opus-5-5",
    "gemini": "gemini-3.8-flash",
    "azure": "DeepSeek-V4.1-Flash",
}
CONFIG_LABEL = "tools_filter"

_default_resolve = run_workflow._resolve_query_credentials
_azure_openai_override: tuple[str, str] | None = None
_claude_creds: tuple[str, str] = (
    "SPI_HAYOUNG_DASHBOARD_AZURE_OPENAI_KEY", "SPI_HAYOUNG_DASHBOARD_OPENAI_BASE_URL",
)


def _required(name: str) -> str:
    value = _env_api_key(name)
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def _claude_resource() -> str:
    return urlparse(_required(_claude_creds[1])).netloc.split(".")[0]


def _claude_foundry_url() -> str:
    return f"https://{_claude_resource()}.services.ai.azure.com/anthropic"


def _resolve_credentials(provider, model, *, lane_name=None):
    if provider in ("openai", "azure") and _azure_openai_override:
        key_env, url_env = _azure_openai_override
        return _required(key_env), _required(url_env), os.environ.get("OPENAI_API_VERSION")
    if provider == "claude":
        return _required(_claude_creds[0]), _claude_foundry_url(), None
    return _default_resolve(provider, model, lane_name=lane_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--providers", required=True,
                        help=f"Comma-separated subset of {sorted(NEW_MODELS)}")
    parser.add_argument("--run-month", default=None,
                        help="Latest closed month / run_month label (default: previous calendar month)")
    parser.add_argument("--smoke", action="store_true",
                        help="Query a single DOI that is pending for every selected model, then stop")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Restrict to N eval DOIs, skipping ones already answered (single provider only)")
    parser.add_argument("--offset", type=int, default=0, metavar="K",
                        help="With --limit: start at eval DOI index K (e.g. --offset 80 --limit 20 = DOIs 81-100)")
    parser.add_argument("--model", default=None,
                        help="Model name to use instead of the default (single provider only)")
    parser.add_argument("--key-env", default=None, metavar="VAR",
                        help="Env var holding the Azure OpenAI key for openai/azure (needs --url-env)")
    parser.add_argument("--url-env", default=None, metavar="VAR",
                        help="Env var holding the matching Azure OpenAI base URL")
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    unknown = [p for p in providers if p not in NEW_MODELS]
    if unknown:
        raise SystemExit(f"Unknown provider(s) {unknown}; choose from {sorted(NEW_MODELS)}")
    if args.limit is not None and len(providers) != 1:
        raise SystemExit("--limit needs exactly one provider")
    if args.model:
        if len(providers) != 1:
            raise SystemExit("--model needs exactly one provider")
        NEW_MODELS[providers[0]] = args.model
    if bool(args.key_env) != bool(args.url_env):
        raise SystemExit("--key-env and --url-env must be given together")
    if args.key_env:
        global _azure_openai_override, _claude_creds
        if providers == ["claude"]:
            _claude_creds = (args.key_env, args.url_env)
        elif set(providers) <= {"openai", "azure"}:
            _azure_openai_override = (args.key_env, args.url_env)
        else:
            raise SystemExit("--key-env/--url-env apply to claude alone, or to openai/azure")
    if args.offset and args.limit is None:
        raise SystemExit("--offset needs --limit")
    run_month = args.run_month or previous_year_month()

    # create_provider() prefers AZURE_ANTHROPIC_RESOURCE_NAME over base_url, and
    # importing sciconharness reloads .env (without override), so set it rather
    # than unsetting it.
    if "claude" in providers:
        os.environ["AZURE_ANTHROPIC_RESOURCE_NAME"] = _claude_resource()
        os.environ.pop("ANTHROPIC_FOUNDRY_RESOURCE", None)
    run_workflow._resolve_query_credentials = _resolve_credentials

    # Every released rolling cohort, not just the latest rolling_panel_months.
    query_cfg.rolling_panel_months = 1200
    query_cfg.default_models = {p: [NEW_MODELS[p]] for p in providers}

    eval_dois, held_back = get_eval_dois(run_month)
    pending = {
        p: _pending_dois_for_model(
            provider=p, model=NEW_MODELS[p], eval_dois=eval_dois,
            run_month=run_month, config_label=CONFIG_LABEL,
        )
        for p in providers
    }

    print("=" * 60)
    print(f"Stage 9 backfill: run_month={run_month} smoke={args.smoke}")
    print(f"  eval DOIs (core + all released rolling): {len(eval_dois)}; held back: {len(held_back)}")
    for p in providers:
        _key, base_url, _ver = _resolve_credentials(p, NEW_MODELS[p])
        host = urlparse(base_url).netloc if base_url else "env default"
        print(f"  {p}/{NEW_MODELS[p]}: pending {len(pending[p])}/{len(eval_dois)} endpoint={host}")
    print("=" * 60)

    if args.smoke:
        common = sorted(set.intersection(*(set(v) for v in pending.values())))
        if not common:
            raise SystemExit("No DOI is pending for every selected model; nothing to smoke-test.")
        dois = [common[0]]
        print(f"Smoke DOI: {dois[0]}")
    elif args.limit is not None:
        dois = eval_dois[args.offset: args.offset + args.limit]
        todo = [d for d in dois if d in set(pending[providers[0]])]
        print(
            f"Limit: eval DOIs {args.offset + 1}-{args.offset + len(dois)} ({dois[0] if dois else '-'} .. "
            f"{dois[-1] if dois else '-'}); {len(todo)} still pending"
        )
        if not todo:
            print("Nothing pending.")
            return
    else:
        dois = eval_dois
        if not any(pending.values()):
            print("Nothing pending.")
            return

    asyncio.run(task_run_queries.fn(dois, run_month=run_month, providers=providers))

    for p in providers:
        left = _pending_dois_for_model(
            provider=p, model=NEW_MODELS[p], eval_dois=dois,
            run_month=run_month, config_label=CONFIG_LABEL,
        )
        print(f"Done {p}/{NEW_MODELS[p]}: {len(dois) - len(left)}/{len(dois)} answered")


if __name__ == "__main__":
    main()
