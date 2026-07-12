# SGLang Diffusion: Disaggregated Flux.2-klein Setup (Experimental)

> **Status**: Not integrated into the application. This documents a standalone
> SGLang Diffusion deployment across the home GPU fleet, evaluated as a
> potential future backend for image generation. It does not replace
> `StableDiffusionCppClient` or `WanGPLTXClient` (see [Remote Services](./remote-services.md))
> unless/until adopted.

## Why

The current image/video backends (`StableDiffusionCppClient`, `WanGPLTXClient`)
are both hardcoded to a single host, `gx10-cbc5` (the DGX Spark), behind a
plain HTTP/OpenAI-compatible API. There is no per-GPU-model allocation logic —
the RTX 4080 and RTX 4060 hosts are idle for this workload.

SGLang Diffusion supports splitting a diffusion pipeline into independent
`encoder` / `denoiser` / `decoder` roles that can run as separate processes on
separate machines, coordinated by a `DiffusionServer` head process. This
lets pipeline stages be placed on whichever GPU fits them, instead of
requiring the whole model to fit on one card.

## Hardware

| Host | Tailscale hostname | Tailscale IP | GPU | VRAM |
|---|---|---|---|---|
| DGX Spark | `gx10-cbc5` | `100.114.182.89` | GB10 (Blackwell, ARM64/sm_121a) | 128 GB unified |
| Desktop | `amor-ms-7e02-1` | `100.82.249.12` | RTX 4080 | 16 GB |
| (unused for this setup) | — | — | RTX 4060 | 16 GB |

Inter-host transport is Tailscale (WireGuard overlay) — there is no RDMA/InfiniBand
fabric between these machines.

## Model: Flux.2-klein-9B

VRAM footprint per pipeline stage (source: user-supplied breakdown, not yet
independently verified against the HF model card — see note below):

| Component | FP16 | FP8 |
|---|---|---|
| DiT transformer (9B, denoiser) | ~18 GB | ~9 GB |
| T5-XXL text encoder | ~9.6 GB | ~4.8 GB |
| CLIP-L text encoder | ~0.4 GB | ~0.2 GB |
| VAE (decoder) | ~0.2 GB | ~0.2 GB |
| Activations @ 1024×1024 | ~1–2 GB | ~1–2 GB |

> **Unverified**: the HF model card for `black-forest-labs/FLUX.2-klein-9B`
> describes an "8B Qwen3 text embedder" rather than T5-XXL + CLIP-L. This is
> a real discrepancy between sources that was not resolved — check
> `text_encoder/config.json` in the actual downloaded repo before sizing the
> encoder role's memory budget.

## Role placement

Denoiser and decoder are colocated on the 4080; only the encoder runs on the
Spark. This is deliberate: the denoiser→decoder transfer carries the full
latent tensor (the heaviest payload in the pipeline) while encoder→denoiser
only carries small text-embedding vectors. Colocating the heavy transfer
keeps it off the network; only the cheap transfer crosses the Tailscale link.

| Role | Host | Precision | Why |
|---|---|---|---|
| `encoder` (T5-XXL + CLIP-L) | Spark | FP16 | Spark has 128 GB, precision is irrelevant there |
| `denoiser` (DiT 9B) | RTX 4080 | **FP8** | FP16 (~18 GB) does not fit in 16 GB; FP8 (~9 GB) does |
| `decoder` (VAE) | RTX 4080 | FP8 checkpoint | Trivial footprint (~0.2–2.2 GB); colocated with denoiser |
| `server` (DiffusionServer head) | Spark | — | Lightweight control-plane router only |

Combined 4080 usage (denoiser + decoder, FP8): ~9.2–11.2 GB, leaving
~5–7 GB headroom for a single request in flight. This headroom shrinks with
higher resolution or batch size — recheck the activations line before
scaling either up.

## Transport: TCP instead of RDMA

SGLang's diffusion disaggregation transfers tensors via Mooncake Transfer
Engine, which defaults to `protocol="rdma"`. Verified locally (see
[Verification](#verification) below) that Mooncake supports `protocol="tcp"`
and initializes cleanly with no RDMA hardware present. This is controlled via
an environment variable, read directly from
`sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`:

```python
# Default is "rdma"; set MOONCAKE_PROTOCOL=efa on AWS EFA hardware.
protocol = envs.MOONCAKE_PROTOCOL.get()
```

Set `MOONCAKE_PROTOCOL=tcp` in the environment of **every** role process
(encoder, denoiser, decoder, head) — a mismatch leaves one side attempting
RDMA and hanging on handshake. `--disagg-ib-device` should be omitted
entirely under TCP.

## Launch commands

```bash
# --- Spark (gx10-cbc5, 100.114.182.89): text encoder role ---
MOONCAKE_PROTOCOL=tcp sglang serve --model-path black-forest-labs/FLUX.2-klein-9B \
    --disagg-role encoder \
    --disagg-server-addr tcp://100.114.182.89:19655 \
    --scheduler-port 19000 \
    --num-gpus 1 \
    --disagg-p2p-hostname 100.114.182.89

# --- RTX 4080 (amor-ms-7e02-1, 100.82.249.12): denoiser role, FP8 ---
MOONCAKE_PROTOCOL=tcp sglang serve --model-path black-forest-labs/FLUX.2-klein-9b-fp8 \
    --disagg-role denoiser \
    --disagg-server-addr tcp://100.114.182.89:19655 \
    --scheduler-port 19001 \
    --num-gpus 1 \
    --disagg-p2p-hostname 100.82.249.12

# --- RTX 4080 (amor-ms-7e02-1, 100.82.249.12): decoder role ---
MOONCAKE_PROTOCOL=tcp sglang serve --model-path black-forest-labs/FLUX.2-klein-9b-fp8 \
    --disagg-role decoder \
    --disagg-server-addr tcp://100.114.182.89:19655 \
    --scheduler-port 19002 \
    --num-gpus 1 \
    --disagg-p2p-hostname 100.82.249.12

# --- DiffusionServer head (Spark) ---
MOONCAKE_PROTOCOL=tcp sglang serve --model-path black-forest-labs/FLUX.2-klein-9b-fp8 \
    --disagg-role server \
    --encoder-urls  "tcp://100.114.182.89:19000" \
    --denoiser-urls "tcp://100.82.249.12:19001" \
    --decoder-urls  "tcp://100.82.249.12:19002" \
    --host 0.0.0.0 --port 30000 \
    --scheduler-port 19655
```

ZMQ handshakes gracefully regardless of start order — the four processes can
be started in any sequence.

## Open questions / not yet verified

1. **Per-role model path**: the encoder command above points at the FP16
   repo while denoiser/decoder point at the FP8 repo. Not confirmed that
   SGLang's diffusion disaggregation supports a different `--model-path` per
   role — the multi-machine example this was adapted from used one shared
   path for all roles. Check `sglang serve --help` for a per-role override
   before relying on this. If unsupported, run the encoder from the FP8 repo
   too (T5-XXL/CLIP-L weights are unaffected by DiT quantization).
2. **Text encoder architecture** (T5-XXL+CLIP-L vs. 8B Qwen3) — see note
   above.
3. **RTX 4060 host** is not currently part of this layout (not present in
   the Tailscale node list at time of writing — `tailscale status` showed no
   entry for it).
4. **Flux support for `--disagg-role`**: the SGLang doc example this
   layout is adapted from used `Wan-AI/Wan2.1-T2V-14B-Diffusers` (video).
   The role architecture (Text-Encoder → DiT → VAE) is generic across
   SGLang's diffusion pipeline decomposition, so it should apply to Flux,
   but this hasn't been tested end-to-end against real Flux.2-klein weights.

## Verification

Confirmed by installing `mooncake-transfer-engine` in an isolated venv and
calling `TransferEngine.initialize()` directly on the RTX 4080 host:

```python
from mooncake.engine import TransferEngine
engine = TransferEngine()
ret = engine.initialize("localhost:12345", "P2PHANDSHAKE", "tcp", "")
# ret == 0 (success)
# log: "No RDMA devices found, check your device installation" (non-fatal warning)
# log: "TcpTransport: listen on port 15623"
```

Note: importing `mooncake.engine` initially failed with
`ImportError: libibverbs.so.1: cannot open shared object file` — the compiled
extension links against the InfiniBand verbs userspace library unconditionally,
even for TCP-only usage. Installing `libibverbs1` (via `apt`, no RDMA hardware
required) resolved the import.

Then confirmed the `MOONCAKE_PROTOCOL` env var by reading SGLang's source
directly (`sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`
and `sglang/srt/environ.py:442`) rather than relying on docs, since earlier
web research had produced a stale/incorrect claim (an old RFC issue,
[sgl-project/sglang#19512](https://github.com/sgl-project/sglang/issues/19512),
was mistakenly read as "not implemented" when the feature had since shipped).
