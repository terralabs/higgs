"""Higgsfield CLI image/video generation backend.

Wraps the official ``higgsfield`` CLI binary (higgsfield-ai/cli v1.1.8+) as an
:class:`ImageGenProvider` implementation so it shows up in ``hermes tools`` →
Image Generation alongside FAL.ai and OpenAI.

Why this exists
---------------
* FAL is fast/cheap for standard image work but doesn't expose Google's Veo,
  Kling, Seedance, Soul character training, or Higgsfield's video model catalog.
* The Higgsfield CLI is the official first-party tool — using it (rather than
  hand-rolling REST calls) gets us OAuth, model-catalog updates, and CLI-only
  workflows (workflows, marketing-studio, game deployment) for free.

Setup on the host
-----------------
1. Install the binary (any of these):
     brew install higgsfield-ai/tap/higgsfield
     curl -fsSL https://raw.githubusercontent.com/higgsfield-ai/cli/main/install.sh | sh
     npm install -g @higgsfield/cli
2. ``higgsfield auth login``  (browser-based OAuth, tokens cache locally)
3. ``higgsfield workspace set <workspace_id>``  (pick billing workspace)
4. ``hermes plugins enable higgsfield``  (only if installed as user plugin,
   not bundled — bundled auto-loads via ``kind: backend``)
5. ``hermes tools`` → Image Generation → Higgsfield  (pick as active provider)

Plugin key derivation: path ``plugins/image_gen/higgsfield/`` →
``image_gen/higgsfield`` — config goes under ``image_gen.higgsfield.*``.

Known limitations
-----------------
* No image-to-image when the underlying model doesn't accept source images —
  we surface a clean error rather than silently dropping them.
* Video models (Veo, Kling, Seedance) take 1-10 minutes. We block with
  --wait/--wait-timeout up to 15min by default; pass ``wait=False`` to fire-
  and-forget and poll separately. The ``image`` field on a successful
  response will be a ``.mp4``/``.mov``/``.webm`` path — gateway delivery
  handles video attachments natively.
* Some video models return frames-as-JPEG-sequence rather than a single MP4.
  We download the first frame as a poster; check ``raw.outputs`` for the
  full payload.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
# Curated for theWAVE.is content: audio-reactive 3D, festival visuals, mood
# reels, character-driven campaigns. Speed column is a rough guide based on
# measured runtimes (image = 5-30s, video = 1-10min) and credit cost.
#
# When the user passes a model id via image_gen.higgsfield.model that's NOT in
# _MODELS, we still forward it to the CLI — the catalog is for the picker,
# not a hard allowlist. New Higgsfield models added server-side just work.

_MODELS: Dict[str, Dict[str, Any]] = {
    # --- Image models (fast, ~5-30s) ---
    "nano_banana_2": {
        "display": "Nano Banana 2",
        "speed": "~8s",
        "modalities": ["text"],
        "strengths": "Fast, cheap, good general-purpose still",
        "price": "low",
    },
    "nano_banana_2_lite": {
        "display": "Nano Banana 2 Lite",
        "speed": "~4s",
        "modalities": ["text"],
        "strengths": "Cheapest tier — rapid iteration",
        "price": "lowest",
    },
    "flux_2": {
        "display": "FLUX.2",
        "speed": "~25s",
        "modalities": ["text", "image"],
        "strengths": "Highest fidelity stills, strong prompt adherence",
        "price": "medium",
    },
    "gemini_omni_flash": {
        "display": "Gemini Omni Flash",
        "speed": "~12s",
        "modalities": ["text", "image"],
        "strengths": "Strong prompt understanding, good for nuanced briefs",
        "price": "low",
    },
    # --- Video models (slow, 1-10min) ---
    "veo_3_1": {
        "display": "Veo 3.1",
        "speed": "1-4min",
        "modalities": ["text", "image"],
        "strengths": "Google flagship — cinematic, narrative shots",
        "price": "high",
    },
    "kling_v3_0": {
        "display": "Kling v3.0",
        "speed": "2-6min",
        "modalities": ["text", "image"],
        "strengths": "Strong motion, Sora-tier competitor",
        "price": "high",
    },
    "seedance_2_0": {
        "display": "Seedance 2.0",
        "speed": "1-3min",
        "modalities": ["text", "image"],
        "strengths": "Motion-graphics native — good for kinetic type / loops",
        "price": "medium",
    },
    # --- Character ---
    "soul_v2": {
        "display": "Soul V2",
        "speed": "~20s (generation) / trained on soul-id first",
        "modalities": ["text", "image"],
        "strengths": "Face-faithful character consistency across shots",
        "price": "medium + training cost",
    },
}

# Default per modality when the user hasn't picked a model
DEFAULTS = {
    "image": "nano_banana_2",
    "video": "veo_3_1",
}

# Aspect ratio → Higgsfield aspect_ratio string. The CLI accepts freeform
# "W:H" so we pass through whatever resolve_aspect_ratio() gives us.
ASPECT_RATIOS = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}

# Models the wrapper should NOT default to --wait on (they take minutes).
SLOW_MODELS = {"veo_3_1", "kling_v3_0", "seedance_2_0"}

# How long to wait for slow models before timing out.
SLOW_WAIT_TIMEOUT = "15m"


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------

def _bin() -> str:
    """Locate the higgsfield binary. Search order matches npm install paths."""
    candidates = [
        "/usr/local/bin/higgsfield",
        "/opt/homebrew/bin/higgsfield",
        "/usr/bin/higgsfield",
        str(Path.home() / ".local" / "bin" / "higgsfield"),
    ]
    for c in candidates:
        if Path(c).is_file():
            return c
    found = shutil.which("higgsfield") or shutil.which("higgs") or shutil.which("hf")
    if not found:
        raise FileNotFoundError(
            "higgsfield binary not found. Install with one of:\n"
            "  brew install higgsfield-ai/tap/higgsfield\n"
            "  npm install -g @higgsfield/cli\n"
            "  curl -fsSL https://raw.githubusercontent.com/higgsfield-ai/cli/main/install.sh | sh"
        )
    return found


# Minimum supported CLI version. The 1.0.0 release was a major rewrite that
# changed model ids, JSON output shape, and added subcommands (workspace,
# account status, etc.) — pre-1.0 builds will break this plugin in subtle ways.
# Refuse to load with a clear upgrade hint rather than fail at first call.
_MIN_VERSION = (1, 0, 0)


def _check_version() -> None:
    """Verify the installed CLI is >= 1.0.0. Raise with an upgrade hint on mismatch."""
    try:
        out = subprocess.run(
            [_bin(), "version"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not run `higgsfield version`: {exc}. "
            "Reinstall: npm install -g @higgsfield/cli@latest"
        ) from exc
    # Parse "higgsfield 1.1.8 (commit) built ..."  →  [1, 1, 8]
    import re
    m = re.search(r"higgsfield\s+v?(\d+)\.(\d+)\.(\d+)", out)
    if not m:
        # If the binary returned something we don't recognize, don't block —
        # the user might be on a brand-new format we haven't seen yet.
        return
    installed = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if installed < _MIN_VERSION:
        raise RuntimeError(
            f"Higgsfield CLI v{'.'.join(map(str, installed))} is too old. "
            f"This plugin requires v{'.'.join(map(str, _MIN_VERSION))} or newer "
            f"(1.0.0 was a major rewrite). Upgrade with:\n"
            f"  npm install -g @higgsfield/cli@latest\n"
            f"  # or\n"
            f"  brew upgrade higgsfield\n"
            f"Then restart Hermes."
        )


def _higgsfield_home() -> Path:
    """Where the CLI caches credentials and where results can be re-routed."""
    # Mirror $HERMES_HOME convention so user can override
    return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))


# ---------------------------------------------------------------------------
# CLI invocation
# ---------------------------------------------------------------------------

def _run(cmd: List[str], timeout: int = 900) -> Dict[str, Any]:
    """Run higgsfield CLI with --json, parse, return dict. Raises on failure."""
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        # Detect recoverable config issues and re-raise as a typed string the
        # caller can match. We don't import the wrapper's typed exceptions
        # to keep the plugin self-contained.
        if "Not authenticated" in stderr or "No workspace selected" in stderr:
            raise RuntimeError(f"HIGGSFIELD_CONFIG:{stderr}")
        raise RuntimeError(
            f"higgsfield CLI failed (exit {proc.returncode}): {' '.join(cmd)}\n  stderr: {stderr}"
        )
    out = proc.stdout.strip()
    if not out:
        return {}
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"_raw": out}


def _extract_result_url(payload: Dict[str, Any]) -> Optional[str]:
    """Best-effort walker for the CLI's polymorphic response shapes."""
    if isinstance(payload.get("result_url"), str):
        return payload["result_url"]
    for key in ("results", "outputs", "assets", "data"):
        v = payload.get(key)
        if isinstance(v, list) and v:
            first = v[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                for k in ("url", "result_url", "output_url", "href", "signed_url"):
                    if isinstance(first.get(k), str):
                        return first[k]
    if isinstance(payload.get("url"), str):
        return payload["url"]
    if isinstance(payload.get("output"), str):
        return payload["output"]
    return None


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class HiggsfieldImageGenProvider(ImageGenProvider):
    """Higgsfield CLI backend — image, video, character, audio via the official CLI."""

    @property
    def name(self) -> str:
        return "higgsfield"

    @property
    def display_name(self) -> str:
        return "Higgsfield"

    def is_available(self) -> bool:
        # Auth is interactive (`higgsfield auth login`), so we can't sniff
        # for an API key in env. Check binary exists AND version is supported.
        try:
            _bin()
            _check_version()
            return True
        except (FileNotFoundError, RuntimeError):
            return False

    def get_availability_warning(self) -> Optional[str]:
        """Return a user-facing warning if installed but on an old version, else None."""
        try:
            _bin()
        except FileNotFoundError:
            return None  # is_available() handles missing-binary
        import re
        try:
            out = subprocess.run(
                [_bin(), "version"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
        m = re.search(r"higgsfield\s+v?(\d+)\.(\d+)\.(\d+)", out)
        if not m or (int(m.group(1)), int(m.group(2)), int(m.group(3))) >= _MIN_VERSION:
            return None
        return (
            f"⚠️  higgsfield CLI v{m.group(0).split()[1]} detected. "
            f"This plugin is tested against v{'.'.join(map(str, _MIN_VERSION))}+. "
            f"Upgrade: `npm i -g @higgsfield/cli@latest`"
        )

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": meta["price"],
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULTS["image"]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Higgsfield",
            "badge": "paid",
            "tag": "Veo 3.1, Kling v3, Seedance 2.0, FLUX.2, Soul V2, Nano Banana Pro — image/video/character",
            "env_vars": [],  # Auth is browser-based, not env-var
            "setup_steps": [
                "Install v1.0+ CLI: `brew install higgsfield-ai/tap/higgsfield` "
                "or `npm install -g @higgsfield/cli@latest`",
                "Authenticate: `higgsfield auth login` (browser OAuth)",
                "Select billing workspace: `higgsfield workspace set <id>`",
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        # The active model controls the actual surface; report the union so
        # the dynamic tool schema advertises image-to-image for the providers
        # that support it. Per-call routing still picks the right code path.
        return {"modalities": ["text", "image"], "max_reference_images": 9}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider=self.name,
                aspect_ratio=aspect,
            )

        # Resolve model: kwarg override > config (loaded by plugin harness) > default
        model_id = kwargs.get("model") or self._pick_default_model(kwargs)
        meta = _MODELS.get(model_id, {})
        is_slow = model_id in SLOW_MODELS

        # Aspect-ratio forwarding: the CLI accepts freeform W:H. We just
        # pass resolve_aspect_ratio's output as-is.
        aspect_flag = aspect

        # Build the CLI command
        bin_path: str
        try:
            bin_path = _bin()
        except FileNotFoundError as exc:
            return error_response(
                error=str(exc),
                error_type="missing_binary",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        cmd: List[str] = [bin_path, "generate", "create", model_id, "--prompt", prompt]
        cmd += ["--aspect-ratio", aspect_flag]

        # Source images: primary + references
        sources: List[str] = []
        if image_url:
            sources.append(image_url)
        sources.extend(normalize_reference_images(reference_image_urls) or [])
        # Dedupe while preserving order
        seen = set()
        deduped = [s for s in sources if not (s in seen or seen.add(s))]
        if deduped:
            # CLI accepts comma-separated paths for image-references
            cmd += ["--image-references", ",".join(deduped)]

        # Default to --wait for fast models; skip for video so callers can poll
        wait = kwargs.get("wait")
        if wait is None:
            wait = not is_slow
        if wait:
            cmd += ["--wait", "--wait-timeout", SLOW_WAIT_TIMEOUT, "--wait-interval", "5s"]

        # Forward any extra params the caller wants to set (duration, seed,
        # resolution, negative_prompt, output_format, etc.). The CLI takes
        # them as --name value, so we coerce booleans/numbers to strings.
        for k, v in kwargs.items():
            if k in ("model", "wait", "wait_timeout", "wait_interval"):
                continue  # already handled
            if v is None:
                continue
            cmd += [f"--{k.replace('_', '-')}", str(v)]

        cmd += ["--json"]

        t0 = time.time()
        try:
            payload = _run(cmd, timeout=900 if is_slow else 180)
        except RuntimeError as exc:
            msg = str(exc)
            if msg.startswith("HIGGSFIELD_CONFIG:"):
                return error_response(
                    error=(
                        "Higgsfield CLI not configured. Run on the host:\n"
                        "  higgsfield auth login   (browser OAuth)\n"
                        "  higgsfield workspace set <id>\n\n"
                        f"Details: {msg.split(':', 1)[1].strip()}"
                    ),
                    error_type="auth_required",
                    provider=self.name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            logger.debug("higgsfield CLI failed", exc_info=True)
            return error_response(
                error=f"Higgsfield generation failed: {msg}",
                error_type="api_error",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except subprocess.TimeoutExpired:
            return error_response(
                error=f"Higgsfield CLI timed out after {'15m' if is_slow else '3m'} — try wait=False and poll manually",
                error_type="timeout",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        elapsed = round(time.time() - t0, 2)
        job_id = payload.get("id") or payload.get("job_id") or payload.get("jobId")
        status = payload.get("status", "completed" if wait else "queued")

        result_url = _extract_result_url(payload)
        image_ref: Optional[str] = None
        if result_url:
            try:
                # save_url_image writes to $HERMES_HOME/cache/images/<prefix>_<ts>_<uuid>.<ext>
                # and returns a Path. Cast to str for the response contract.
                saved = save_url_image(result_url, prefix=f"higgsfield_{model_id}")
                image_ref = str(saved)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not cache Higgsfield result %s: %s", result_url, exc
                )
                # Fall back to bare URL — gateway will fetch on delivery
                image_ref = result_url

        if not image_ref:
            return error_response(
                error="Higgsfield returned no result URL in the response",
                error_type="empty_response",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
                extra={"raw": _trim_payload(payload)},
            )

        extra: Dict[str, Any] = {
            "elapsed_sec": elapsed,
            "waited": wait,
            "is_video": is_slow,
        }
        if job_id:
            extra["job_id"] = job_id
        if status:
            extra["status"] = status
        if "credits_used" in payload:
            extra["credits_used"] = payload["credits_used"]

        # Source attribution for the first frame (video models)
        if deduped:
            extra["source_image_count"] = len(deduped)

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            modality="image" if deduped else "text",
            extra=extra,
        )

    # ------------------------------------------------------------------ helpers

    def _pick_default_model(self, kwargs: Dict[str, Any]) -> str:
        """Default to image-class model unless caller asked for video via extra params."""
        # The config key image_gen.higgsfield.model is loaded by the harness
        # into kwargs.get('model') before we get here, so this only runs if
        # nothing was set anywhere — fall back to the image default.
        return DEFAULTS["image"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trim_payload(payload: Dict[str, Any], max_bytes: int = 4000) -> Dict[str, Any]:
    """Trim the raw CLI response for inclusion in error_response extra field."""
    try:
        blob = json.dumps(payload)
        if len(blob) <= max_bytes:
            return payload
        # Strip large arrays but keep the keys so the user knows what came back
        trimmed: Dict[str, Any] = {}
        for k, v in payload.items():
            if isinstance(v, (list, str)) and len(json.dumps(v)) > 500:
                trimmed[k] = f"<{type(v).__name__} of len {len(v) if hasattr(v, '__len__') else '?'}>"
            else:
                trimmed[k] = v
        return trimmed
    except Exception:  # noqa: BLE001
        return {"_unserializable": True}


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Plugin entry point — wire ``HiggsfieldImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(HiggsfieldImageGenProvider())
