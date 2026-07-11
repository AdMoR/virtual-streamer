#!/usr/bin/env python3
"""
vsctl — self-discovering CLI client for the Virtual Streamer API.

Instead of hardcoding endpoints (MCP-style), vsctl reads the live OpenAPI
schema from the running API, so any agent (or human) can *discover* what the
system can do and call it. New endpoints appear automatically.

Usage:
    vsctl ops [filter]                      List available operations
    vsctl describe <operationId>            Show params/body schema for one operation
    vsctl call <operationId> [options]      Invoke an operation
        -p name=value                       Path/query parameter (repeatable)
        --json '{"k": "v"}'                 Request body (JSON string or @file.json)
        --api http://host:8000              API base (default $VS_API_URL or localhost:8000)

Examples:
    vsctl ops candidate
    vsctl describe select_candidate_api_v1_candidates__candidate_id__select_post
    vsctl call list_stories_api_v1_stories_get -p limit=5
    vsctl call recompose_story_api_v1_stories__story_id__recompose_post \
        -p story_id=STORY_UUID --json '{}'

Requires: requests (already a project dependency).
"""

import argparse
import json
import os
import re
import sys

import requests

DEFAULT_API = os.environ.get("VS_API_URL", "http://localhost:8000")


def load_schema(api_base: str) -> dict:
    r = requests.get(f"{api_base}/openapi.json", timeout=10)
    r.raise_for_status()
    return r.json()


def iter_operations(schema: dict):
    for path, methods in schema.get("paths", {}).items():
        for method, op in methods.items():
            if method.lower() not in ("get", "post", "put", "delete", "patch"):
                continue
            yield {
                "id": op.get("operationId") or f"{method}_{path}",
                "method": method.upper(),
                "path": path,
                "summary": op.get("summary", ""),
                "tags": op.get("tags", []),
                "op": op,
            }


def find_operation(schema: dict, op_id: str) -> dict:
    ops = list(iter_operations(schema))
    exact = [o for o in ops if o["id"] == op_id]
    if exact:
        return exact[0]
    partial = [o for o in ops if op_id.lower() in o["id"].lower()]
    if len(partial) == 1:
        return partial[0]
    if partial:
        sys.exit(
            f"Ambiguous operation '{op_id}'. Matches:\n  "
            + "\n  ".join(o["id"] for o in partial)
        )
    sys.exit(f"Unknown operation '{op_id}'. Run: vsctl ops")


def cmd_ops(args):
    schema = load_schema(args.api)
    filt = (args.filter or "").lower()
    for o in iter_operations(schema):
        line = f"{o['method']:6} {o['path']:60} {o['id']}"
        if not filt or filt in line.lower() or filt in o["summary"].lower():
            print(line)
            if o["summary"]:
                print(f"       └─ {o['summary']}")


def cmd_describe(args):
    schema = load_schema(args.api)
    o = find_operation(schema, args.operation)
    print(f"{o['method']} {o['path']}")
    print(f"operationId: {o['id']}")
    if o["summary"]:
        print(f"summary: {o['summary']}")
    if o["op"].get("description"):
        print(f"\n{o['op']['description'].strip()}\n")
    params = o["op"].get("parameters", [])
    if params:
        print("parameters:")
        for p in params:
            req = " (required)" if p.get("required") else ""
            typ = p.get("schema", {}).get("type", "any")
            print(f"  -p {p['name']}=<{typ}>  [{p['in']}]{req}")
    body = o["op"].get("requestBody", {})
    if body:
        content = body.get("content", {}).get("application/json", {})
        ref = content.get("schema", {}).get("$ref", "")
        if ref:
            name = ref.rsplit("/", 1)[-1]
            model = schema.get("components", {}).get("schemas", {}).get(name, {})
            print(f"body ({name}):")
            print(json.dumps(model, indent=2))
        else:
            print("body: (see schema)")
            print(json.dumps(content.get("schema", {}), indent=2))


def cmd_call(args):
    schema = load_schema(args.api)
    o = find_operation(schema, args.operation)

    kv = {}
    for item in args.param or []:
        if "=" not in item:
            sys.exit(f"Bad -p value '{item}' (expected name=value)")
        k, v = item.split("=", 1)
        kv[k] = v

    # Fill path parameters, remaining kv become query params
    path = o["path"]
    for name in re.findall(r"\{(\w+)\}", path):
        if name not in kv:
            sys.exit(f"Missing path parameter: -p {name}=...")
        path = path.replace("{" + name + "}", kv.pop(name))

    body = None
    if args.json:
        raw = args.json
        if raw.startswith("@"):
            raw = open(raw[1:]).read()
        body = json.loads(raw)
    elif o["method"] in ("POST", "PUT", "PATCH") and o["op"].get("requestBody"):
        body = {}

    r = requests.request(
        o["method"],
        f"{args.api}{path}",
        params=kv or None,
        json=body,
        timeout=args.timeout,
    )
    try:
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    except ValueError:
        print(r.text)
    if not r.ok:
        sys.exit(f"HTTP {r.status_code}")


def main():
    ap = argparse.ArgumentParser(prog="vsctl", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default=DEFAULT_API, help=f"API base URL (default {DEFAULT_API})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ops = sub.add_parser("ops", help="List available operations")
    p_ops.add_argument("filter", nargs="?", help="Substring filter")
    p_ops.set_defaults(fn=cmd_ops)

    p_desc = sub.add_parser("describe", help="Describe one operation")
    p_desc.add_argument("operation")
    p_desc.set_defaults(fn=cmd_describe)

    p_call = sub.add_parser("call", help="Invoke an operation")
    p_call.add_argument("operation")
    p_call.add_argument("-p", "--param", action="append", help="name=value (path or query)")
    p_call.add_argument("--json", help="JSON request body (string or @file)")
    p_call.add_argument("--timeout", type=float, default=60.0)
    p_call.set_defaults(fn=cmd_call)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
