#!/usr/bin/env python3
"""
vsctl_precheck — verify all service prerequisites for an e2e vsctl video
generation run in one shot, instead of curling each endpoint by hand.

Usage:
    uv run python scripts/vsctl_precheck.py

Exit code is 0 iff every check passes. Prints one PASS/FAIL line per check
plus a short reason on failure. Consult .claude/skills/vsctl-e2e-video/SKILL.md
for what each service is for.
"""

import os
import subprocess
import sys

import requests

VS_API_URL = os.environ.get("VS_API_URL", "http://localhost:8000")
LTX_SERVER_URL = os.environ.get("LTX_SERVER_URL", "http://gx10-cbc5:8082")
SD_SERVER_URL = os.environ.get("SD_SERVER_URL", "http://gx10-cbc5:1234")
JUDGE_LLM_URL = os.environ.get("JUDGE_LLM_URL", "http://100.114.182.89:8081")
ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")

TIMEOUT = 5

results = []


def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    results.append((name, ok, detail))


def check_rest_api():
    r = requests.get(f"{VS_API_URL}/health", timeout=TIMEOUT)
    return r.status_code == 200, f"status={r.status_code}"


def check_ltx_server():
    r = requests.get(f"{LTX_SERVER_URL}/health", timeout=TIMEOUT)
    if r.status_code != 200:
        return False, f"status={r.status_code}"
    loaded = r.json().get("runtime_loaded")
    return bool(loaded), f"runtime_loaded={loaded}"


def check_sd_server():
    r = requests.get(f"{SD_SERVER_URL}/v1/models", timeout=TIMEOUT)
    return r.status_code == 200, f"status={r.status_code}"


def check_judge_llm():
    r = requests.get(f"{JUDGE_LLM_URL}/v1/models", timeout=TIMEOUT)
    return r.status_code == 200, f"status={r.status_code}"


def check_compose_services():
    out = subprocess.run(
        ["docker", "compose", "ps", "mysql", "minio", "--format", "{{.Service}} {{.State}}"],
        capture_output=True, text=True, timeout=15,
    )
    if out.returncode != 0:
        return False, out.stderr.strip()
    lines = [l for l in out.stdout.strip().splitlines() if l.strip()]
    states = dict(l.split(maxsplit=1) for l in lines)
    missing = [s for s in ("mysql", "minio") if s not in states]
    not_running = [s for s, st in states.items() if "running" not in st.lower()]
    if missing or not_running:
        return False, f"missing={missing} not_running={not_running}"
    return True, ", ".join(f"{k}={v}" for k, v in states.items())


def check_anthropic_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True, "set in environment"
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            for line in f:
                if line.strip().startswith("ANTHROPIC_API_KEY=") and line.strip() != "ANTHROPIC_API_KEY=":
                    return True, "set in .env"
    return False, "not found in env or .env"


check("REST API", check_rest_api)
check("LTX video server", check_ltx_server)
check("Stable Diffusion server", check_sd_server)
check("Judge/scene LLM", check_judge_llm)
check("MySQL + MinIO (compose)", check_compose_services)
check("ANTHROPIC_API_KEY", check_anthropic_key)

width = max(len(n) for n, _, _ in results)
all_ok = True
for name, ok, detail in results:
    status = "PASS" if ok else "FAIL"
    all_ok &= ok
    print(f"[{status}] {name:<{width}}  {detail}")

if not all_ok:
    print(
        "\nOne or more checks failed. See .claude/skills/vsctl-e2e-video/SKILL.md "
        "section 0 for what each service does and how to bring it up.",
        file=sys.stderr,
    )
    sys.exit(1)

print("\nAll prerequisite checks passed.")
