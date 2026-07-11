#!/usr/bin/env python3
"""
vsctl — curated, discoverable CLI client for the Virtual Streamer generation API.

Exposes only the *generic* generation operations (video, script, voice, image,
entities, review) under short stable names, so an agent driving a regeneration
loop only needs a small operation surface. Parameter/body schemas are still
discovered live from the API's OpenAPI document — nothing is hardcoded beyond
the operation registry below.

Usage:
    vsctl ops [filter]              List curated operations (--all for the raw API)
    vsctl describe <op>             Show params + body schema for one operation
    vsctl call <op> [options]       Invoke an operation
        -p name=value               Path/query parameter (repeatable)
        --json '{"k": "v"}'         Request body (JSON string or @file.json)
        --api http://host:8000      API base (default $VS_API_URL or localhost:8000)

Typical regeneration loop:
    vsctl call list-stories -p limit=5
    vsctl call get-story-scenes -p story_id=UUID
    vsctl call list-candidates -p story_id=UUID -p scene_id=UUID
    vsctl call regenerate-scene -p story_id=UUID -p scene_id=UUID --json '{"max_candidates": 3}'
    vsctl call job-status -p job_id=UUID
    vsctl call recompose-story -p story_id=UUID --json '{}'
"""

import argparse
import json
import os
import re
import sys

import requests

DEFAULT_API = os.environ.get("VS_API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Curated operation registry: short name → (METHOD, path, summary)
# Only generic generation capabilities belong here — no character-specific
# agents (jesus…), streaming, playlist, twitch or admin endpoints.
# ---------------------------------------------------------------------------

OPERATIONS: dict[str, tuple[str, str, str]] = {
    # ── Generation ──────────────────────────────────────────────────────────
    "generate-video": (
        "POST", "/api/v1/video-generation/generate",
        "Full video from a title + story template (script → images → LTX takes with seed hunting → concat). Returns job_id.",
    ),
    "generate-video-from-script": (
        "POST", "/api/v1/video-generation/generate-from-script",
        "Full video from a pre-built/edited script (scenes + locations from generate-script). Returns job_id.",
    ),
    "generate-clip": (
        "POST", "/api/v1/video-generation/single-clip",
        "One raw LTX clip (t2v/i2v/talking-head) — multipart form with optional image/audio files. Returns job_id.",
    ),
    "generate-script": (
        "POST", "/api/v1/story-pipeline/run",
        "Script generation from a title + story template: raw story → locations → detailed scenes.",
    ),
    "generate-voice": (
        "POST", "/api/v1/tts/generate",
        "Voice audio for a text line using a character's cloned voice (character_id + text).",
    ),
    "generate-scene-image": (
        "POST", "/api/v1/location-generation/generate-image",
        "Image for a location + optional character (Stable Diffusion). Returns base64 PNG.",
    ),
    "generate-location": (
        "POST", "/api/v1/location-generation/generate",
        "Generate a new location entity (description + base image) for a story template.",
    ),
    "regenerate-location-image": (
        "POST", "/api/v1/location-generation/{location_id}/regenerate-image",
        "Regenerate the base image of an existing location.",
    ),
    "generate-story-template": (
        "POST", "/api/v1/story-template-generation/generate",
        "Generate a story template from a free-text show concept.",
    ),
    # ── Entities ────────────────────────────────────────────────────────────
    "create-character":     ("POST", "/api/v1/characters", "Create a character (identity images + voice samples added separately)."),
    "list-characters":      ("GET",  "/api/v1/characters", "List characters."),
    "get-character":        ("GET",  "/api/v1/characters/{character_id}", "Get one character."),
    "create-location":      ("POST", "/api/v1/locations", "Create a location entity."),
    "list-locations":       ("GET",  "/api/v1/locations", "List locations (filter by story_template_id)."),
    "get-location":         ("GET",  "/api/v1/locations/{location_id}", "Get one location."),
    "create-story-template": ("POST", "/api/v1/story-templates", "Create a story template."),
    "list-story-templates": ("GET",  "/api/v1/story-templates", "List story templates."),
    "get-story-template":   ("GET",  "/api/v1/story-templates/{template_id}", "Get one story template."),
    # ── Review / regeneration loop ──────────────────────────────────────────
    "list-stories":     ("GET",  "/api/v1/stories", "List generated stories (id, title, status, final_video_key)."),
    "get-story":        ("GET",  "/api/v1/stories/{story_id}", "Get one story incl. raw agent output for replay."),
    "get-story-scenes": ("GET",  "/api/v1/stories/{story_id}/scenes", "List scenes of a story."),
    "list-candidates":  ("GET",  "/api/v1/stories/{story_id}/scenes/{scene_id}/candidates", "All judged takes of a scene (seed, verdict, selected)."),
    "select-candidate": ("POST", "/api/v1/candidates/{candidate_id}/select", "Override the judge: use this take. Recompose afterwards."),
    "submit-feedback":  ("POST", "/api/v1/candidates/{candidate_id}/feedback", "Human/agent preference label on a take (improves the judge)."),
    "export-feedback":  ("GET",  "/api/v1/judge-feedback/export", "Judge-vs-human labels for judge tuning."),
    "regenerate-scene": ("POST", "/api/v1/stories/{story_id}/scenes/{scene_id}/regenerate", "Fresh seed hunt for one scene. Returns job_id."),
    "recompose-story":  ("POST", "/api/v1/stories/{story_id}/recompose", "Rebuild the final video from selected takes. Returns job_id."),
    # ── Utilities ───────────────────────────────────────────────────────────
    "job-status": ("GET", "/api/v1/jobs/{job_id}", "Status/result of a background generation job."),
    "presign":    ("GET", "/api/v1/storage/presign", "Presigned URL for a MinIO key (view videos/images)."),
}


def load_schema(api_base: str) -> dict:
    r = requests.get(f"{api_base}/openapi.json", timeout=10)
    r.raise_for_status()
    return r.json()


def resolve(op_name: str) -> tuple[str, str, str]:
    if op_name in OPERATIONS:
        return OPERATIONS[op_name]
    matches = [k for k in OPERATIONS if op_name.lower() in k]
    if len(matches) == 1:
        return OPERATIONS[matches[0]]
    if matches:
        sys.exit(f"Ambiguous operation '{op_name}'. Matches: {', '.join(matches)}")
    sys.exit(f"Unknown operation '{op_name}'. Run: vsctl ops")


def schema_entry(schema: dict, method: str, path: str) -> dict:
    return schema.get("paths", {}).get(path, {}).get(method.lower(), {})


def cmd_ops(args):
    if args.all:
        schema = load_schema(args.api)
        for path, methods in schema.get("paths", {}).items():
            for method, op in methods.items():
                if method.lower() in ("get", "post", "put", "delete", "patch"):
                    print(f"{method.upper():6} {path:60} {op.get('summary','')}")
        return
    filt = (args.filter or "").lower()
    for name, (method, path, summary) in OPERATIONS.items():
        if filt and filt not in name and filt not in summary.lower():
            continue
        print(f"{name:28} {method:5} {path}")
        print(f"{'':28} └─ {summary}")


def cmd_describe(args):
    method, path, summary = resolve(args.operation)
    print(f"{method} {path}\n{summary}\n")
    schema = load_schema(args.api)
    op = schema_entry(schema, method, path)
    if not op:
        print("(operation not found in live OpenAPI schema — is the API up to date?)")
        return
    if op.get("description"):
        print(op["description"].strip() + "\n")
    for p in op.get("parameters", []):
        req = " (required)" if p.get("required") else ""
        typ = p.get("schema", {}).get("type", "any")
        print(f"  -p {p['name']}=<{typ}>  [{p['in']}]{req}")
    body = op.get("requestBody", {})
    if body:
        content = body.get("content", {})
        json_content = content.get("application/json", {})
        ref = json_content.get("schema", {}).get("$ref", "")
        if ref:
            name = ref.rsplit("/", 1)[-1]
            model = schema.get("components", {}).get("schemas", {}).get(name, {})
            print(f"body ({name}):")
            print(json.dumps(model, indent=2))
        elif "multipart/form-data" in content:
            print("body: multipart/form-data")
            print(json.dumps(content["multipart/form-data"].get("schema", {}), indent=2))
        else:
            print("body:")
            print(json.dumps(json_content.get("schema", {}), indent=2))


def cmd_call(args):
    method, path, _ = resolve(args.operation)

    kv = {}
    for item in args.param or []:
        if "=" not in item:
            sys.exit(f"Bad -p value '{item}' (expected name=value)")
        k, v = item.split("=", 1)
        kv[k] = v

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
    elif method in ("POST", "PUT", "PATCH"):
        body = {}

    r = requests.request(
        method, f"{args.api}{path}", params=kv or None, json=body, timeout=args.timeout
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

    p_ops = sub.add_parser("ops", help="List curated operations")
    p_ops.add_argument("filter", nargs="?", help="Substring filter")
    p_ops.add_argument("--all", action="store_true", help="List the raw OpenAPI surface instead")
    p_ops.set_defaults(fn=cmd_ops)

    p_desc = sub.add_parser("describe", help="Describe one operation")
    p_desc.add_argument("operation")
    p_desc.set_defaults(fn=cmd_describe)

    p_call = sub.add_parser("call", help="Invoke an operation")
    p_call.add_argument("operation")
    p_call.add_argument("-p", "--param", action="append", help="name=value (path or query)")
    p_call.add_argument("--json", help="JSON request body (string or @file)")
    p_call.add_argument("--timeout", type=float, default=120.0)
    p_call.set_defaults(fn=cmd_call)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
