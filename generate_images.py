#!/usr/bin/env python3
"""Queue approved prompts to a local ComfyUI server asynchronously.

Pipeline phase covered:
5) Read prompts_approved.txt and split by ###
6) Inject prompt text into CLIPTextEncode (or workflow positive prompt node) and queue jobs
   to ComfyUI; stamp SaveImage / SaveVideo filename prefixes when present.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp
from pipeline_settings import get_default_settings_path, load_settings, save_settings

DEFAULT_DELIMITER = "###"
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_WORKFLOW_DIR = Path("comfy_workflows")
DEFAULT_FINAL_WIDTH = 1920
DEFAULT_FINAL_HEIGHT = 1088
DEFAULT_FINAL_FPS = 24
WORKFLOW_PRESETS = {
    "flux2klein": {
        "workflow_file": "flux2Klein_t2i_9b_v01.json",
        "clip_node_id": None,
    },
    "z-image": {
        "workflow_file": "z_image_t2i_v01.json",
        "clip_node_id": None,
    },
    "ltx23": {
        "workflow_file": "LTX-2.3_T2V_Two_Stage_v01.json",
        "clip_node_id": "2483",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue ComfyUI generation jobs (images or LTX video) from approved prompts."
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("prompts_approved.txt"),
        help="Approved prompt file path (default: prompts_approved.txt)",
    )
    parser.add_argument(
        "--payload-template",
        type=Path,
        default=None,
        help="ComfyUI API payload template JSON path (advanced override)",
    )
    parser.add_argument(
        "--render-model",
        choices=list(WORKFLOW_PRESETS.keys()),
        default=None,
        help=(
            "Workflow preset: flux2klein, z-image (still), or ltx23 (LTX-2.3 video). "
            "Ignored if --payload-template is provided."
        ),
    )
    parser.add_argument(
        "--workflow-dir",
        type=Path,
        default=DEFAULT_WORKFLOW_DIR,
        help="Directory containing Comfy workflow JSON presets.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Do not ask interactive model choice; use saved model if available.",
    )
    parser.add_argument(
        "--settings-file",
        type=Path,
        default=get_default_settings_path(),
        help="Path to shared pipeline settings file.",
    )
    parser.add_argument(
        "--ignore-saved-model",
        action="store_true",
        help="Ignore model stored by Phase 1 and use prompt/default instead.",
    )
    parser.add_argument(
        "--comfy-url",
        default=DEFAULT_COMFY_URL,
        help=f"ComfyUI base URL (default: {DEFAULT_COMFY_URL})",
    )
    parser.add_argument(
        "--delimiter",
        default=DEFAULT_DELIMITER,
        help=f"Prompt delimiter line (default: {DEFAULT_DELIMITER})",
    )
    parser.add_argument(
        "--clip-node-id",
        default=None,
        help="Node id for CLIPTextEncode text input (auto-detect first if omitted)",
    )
    parser.add_argument(
        "--text-input-key",
        default="text",
        help='Input field inside CLIPTextEncode node (default: "text")',
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Advanced: full SaveImage filename_prefix override. Example: "
            "'story_runs/my_story/scene'. If omitted, beginner flags are used."
        ),
    )
    parser.add_argument(
        "--output-folder",
        default="story_runs",
        help=(
            "Beginner: subfolder under ComfyUI output directory "
            "(default: story_runs)"
        ),
    )
    parser.add_argument(
        "--run-name",
        default="default_run",
        help=(
            "Beginner: run subfolder name under output-folder "
            "(default: default_run)"
        ),
    )
    parser.add_argument(
        "--file-prefix",
        default="scene",
        help="Beginner: base filename prefix (default: scene)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent queue requests (default: 4)",
    )
    parser.add_argument(
        "--comfy-output-dir",
        type=Path,
        default=None,
        help=(
            "ComfyUI output folder root (contains subfolders like story_runs/...). "
            "Required for LTX final concat unless COMFYUI_OUTPUT_DIR is set or saved in settings."
        ),
    )
    parser.add_argument(
        "--no-concat-ltx-final",
        action="store_true",
        help="For ltx23: do not wait for clips or build one master video (ffmpeg).",
    )
    parser.add_argument(
        "--final-master-name",
        default="master_1920x1088_24fps",
        help="Base filename for assembled LTX master video (default: master_1920x1088_24fps)",
    )
    parser.add_argument(
        "--final-width",
        type=int,
        default=DEFAULT_FINAL_WIDTH,
        help=f"Master video width (default: {DEFAULT_FINAL_WIDTH})",
    )
    parser.add_argument(
        "--final-height",
        type=int,
        default=DEFAULT_FINAL_HEIGHT,
        help=f"Master video height (default: {DEFAULT_FINAL_HEIGHT})",
    )
    parser.add_argument(
        "--final-fps",
        type=float,
        default=float(DEFAULT_FINAL_FPS),
        help=f"Master video frame rate (default: {DEFAULT_FINAL_FPS})",
    )
    parser.add_argument(
        "--ltx-wait-timeout",
        type=float,
        default=7200.0,
        help="Max seconds to wait per LTX clip job in history (default: 7200)",
    )
    parser.add_argument(
        "--no-save-comfy-output-dir",
        action="store_true",
        help="Do not save comfy output dir to settings after interactive entry.",
    )
    return parser.parse_args()


def _sanitize_path_part(value: str, label: str) -> str:
    cleaned = value.strip().replace("\\", "/").strip("/")
    cleaned = re.sub(r"[<>:\"|?*]", "_", cleaned)
    if not cleaned:
        raise ValueError(f"{label} cannot be empty.")
    return cleaned


def resolve_comfy_output_root(
    args: argparse.Namespace,
    selected_model: str,
) -> Path | None:
    """Return ComfyUI output root when LTX master concat is needed."""
    if selected_model != "ltx23" or args.no_concat_ltx_final:
        return None

    candidates: list[Path] = []
    if args.comfy_output_dir is not None:
        candidates.append(args.comfy_output_dir.expanduser())
    env_dir = os.environ.get("COMFYUI_OUTPUT_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    saved = load_settings(args.settings_file).get("comfy_output_dir")
    if isinstance(saved, str) and saved.strip():
        candidates.append(Path(saved.strip()).expanduser())

    for p in candidates:
        try:
            r = p.resolve()
        except OSError:
            continue
        if r.is_dir():
            return r

    if args.non_interactive:
        raise ValueError(
            "LTX master video assembly needs an existing ComfyUI output folder. Set "
            "--comfy-output-dir, or environment variable COMFYUI_OUTPUT_DIR, or add "
            '"comfy_output_dir" to .story_to_comfy_settings.json (see README).'
        )

    entered = input(
        "Enter ComfyUI output folder path (the folder that contains story_runs/...), "
        "e.g. D:\\ComfyUI\\output: "
    ).strip()
    if not entered:
        raise ValueError("ComfyUI output folder is required to assemble the LTX master video.")
    root = Path(entered).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory or path missing: {root}")
    if not args.no_save_comfy_output_dir:
        save_settings(args.settings_file, {"comfy_output_dir": str(root)})
        print(f"Saved comfy output dir to: {args.settings_file}")
    return root


def extract_video_path_from_history(
    history_entry: dict[str, Any],
    comfy_output_root: Path,
) -> Path | None:
    outputs = history_entry.get("outputs")
    if not isinstance(outputs, dict):
        return None
    video_keys = ("gifs", "videos", "images")
    for node_out in outputs.values():
        if not isinstance(node_out, dict):
            continue
        for key in video_keys:
            files = node_out.get(key)
            if not isinstance(files, list):
                continue
            for item in files:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename")
                if not filename or not str(filename).lower().endswith(
                    (".mp4", ".webm", ".mkv", ".mov", ".avi")
                ):
                    continue
                subfolder = str(item.get("subfolder", "") or "").replace("\\", "/").strip("/")
                typ = item.get("type", "output")
                if typ != "output":
                    continue
                rel = Path(subfolder) / filename if subfolder else Path(filename)
                candidate = (comfy_output_root / rel).resolve()
                if candidate.is_file():
                    return candidate
    return None


async def fetch_history_entry(
    session: aiohttp.ClientSession,
    comfy_url: str,
    prompt_id: str,
) -> dict[str, Any] | None:
    base = comfy_url.rstrip("/")
    url = f"{base}/history/{prompt_id}"
    async with session.get(url) as resp:
        if resp.status == 200:
            data = await resp.json()
            if isinstance(data, dict):
                entry = data.get(prompt_id)
                if isinstance(entry, dict):
                    return entry
    async with session.get(f"{base}/history") as resp:
        if resp.status != 200:
            return None
        data = await resp.json()
        if isinstance(data, dict):
            entry = data.get(prompt_id)
            if isinstance(entry, dict):
                return entry
    return None


async def wait_for_ltx_clip_file(
    session: aiohttp.ClientSession,
    comfy_url: str,
    prompt_id: str,
    comfy_output_root: Path,
    clip_index: int,
    max_wait_s: float,
    poll_s: float,
) -> Path:
    if prompt_id in ("unknown", "", None):
        raise RuntimeError(f"Invalid prompt_id for clip {clip_index}")

    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        entry = await fetch_history_entry(session, comfy_url, prompt_id)
        if entry:
            path = extract_video_path_from_history(entry, comfy_output_root)
            if path is not None and path.is_file():
                return path
        await asyncio.sleep(poll_s)
    raise TimeoutError(
        f"Timed out after {max_wait_s:.0f}s waiting for LTX clip {clip_index} "
        f"(prompt_id={prompt_id}). Check ComfyUI for errors."
    )


def run_ffmpeg_concat_clips(
    clip_paths: list[Path],
    output_file: Path,
    width: int,
    height: int,
    fps: float,
) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install ffmpeg and retry to build the master LTX video."
        )
    if len(clip_paths) < 1:
        raise ValueError("No clip files to concatenate.")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    vf = (
        f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )

    fd, list_path = tempfile.mkstemp(suffix="_concat.txt", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for clip in clip_paths:
                ap = clip.resolve().as_posix().replace("'", "'\\''")
                handle.write(f"file '{ap}'\n")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_file),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited {proc.returncode}. stderr:\n{proc.stderr[-4000:]}"
            )
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass


def resolve_output_prefix(args: argparse.Namespace) -> str:
    if args.output_prefix:
        return _sanitize_path_part(args.output_prefix, "--output-prefix")

    folder = _sanitize_path_part(args.output_folder, "--output-folder")
    run_name = args.run_name
    if not args.non_interactive and args.run_name == "default_run":
        user_value = input(
            "Enter output run folder name (press Enter for default_run): "
        ).strip()
        if user_value:
            run_name = user_value
    run_name = _sanitize_path_part(run_name, "--run-name")
    file_prefix = _sanitize_path_part(args.file_prefix, "--file-prefix")
    return f"{folder}/{run_name}/{file_prefix}"


def read_prompts(path: Path, delimiter: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Approved prompts file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    chunks = [chunk.strip() for chunk in raw.split(delimiter)]
    prompts = [chunk for chunk in chunks if chunk]
    if not prompts:
        raise ValueError(f"No prompts found in file: {path}")
    return prompts


def load_template(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Payload template not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "prompt" in data and isinstance(data["prompt"], dict):
        return data
    if all(isinstance(v, dict) for v in data.values()):
        return {"prompt": data}
    raise ValueError(
        "Invalid template JSON. Expected either {'prompt': {...nodes...}} or raw node map."
    )


def resolve_render_model_choice(
    render_model: str | None,
    non_interactive: bool,
    settings_file: Path,
    ignore_saved_model: bool,
) -> tuple[str, str]:
    default_model = "flux2klein"
    saved_model = None
    if not ignore_saved_model:
        saved = load_settings(settings_file).get("last_render_model")
        if isinstance(saved, str) and saved in WORKFLOW_PRESETS:
            saved_model = saved

    if render_model:
        return render_model, "cli"
    if non_interactive:
        if saved_model:
            return saved_model, "settings"
        return default_model, "default"

    prompt_default = saved_model or default_model

    models = list(WORKFLOW_PRESETS.keys())
    print("Choose workflow model preset:")
    for idx, model in enumerate(models, start=1):
        print(f"  {idx}) {model}")
    if saved_model:
        print(f"Saved model from Phase 1: {saved_model}")
    print(f"Press Enter for default: {prompt_default}")
    user_value = input("Model choice: ").strip().lower()
    if not user_value:
        return prompt_default, "settings" if saved_model else "default"
    if user_value in WORKFLOW_PRESETS:
        return user_value, "interactive"
    if user_value.isdigit():
        idx = int(user_value)
        if 1 <= idx <= len(models):
            return models[idx - 1], "interactive"
    raise ValueError("Invalid model choice. Choose flux2klein, z-image, or ltx23.")


def resolve_template_and_clip_node(
    args: argparse.Namespace,
) -> tuple[Path, str | None, str, str]:
    if args.payload_template is not None:
        return args.payload_template, args.clip_node_id, "custom", "custom"

    model, model_source = resolve_render_model_choice(
        render_model=args.render_model,
        non_interactive=args.non_interactive,
        settings_file=args.settings_file,
        ignore_saved_model=args.ignore_saved_model,
    )
    preset = WORKFLOW_PRESETS[model]
    template_path = args.workflow_dir / preset["workflow_file"]
    clip_node = args.clip_node_id or preset["clip_node_id"]
    return template_path, clip_node, model, model_source


def find_clip_node_id(prompt_graph: dict[str, Any], requested_id: str | None) -> str:
    if requested_id:
        if requested_id not in prompt_graph:
            raise KeyError(f"Provided clip-node-id '{requested_id}' not found in graph.")
        return requested_id

    # Prefer explicit positive prompt nodes when metadata is available.
    for node_id, node in prompt_graph.items():
        if node.get("class_type") != "CLIPTextEncode":
            continue
        title = str(node.get("_meta", {}).get("title", "")).lower()
        if "positive" in title:
            return node_id

    for node_id, node in prompt_graph.items():
        if node.get("class_type") == "CLIPTextEncode":
            return node_id
    raise ValueError(
        "Could not auto-detect CLIPTextEncode node. Pass --clip-node-id explicitly."
    )


def inject_prompt(
    payload_template: dict[str, Any],
    prompt_text: str,
    clip_node_id: str,
    text_input_key: str,
    output_prefix: str,
    index: int,
) -> dict[str, Any]:
    payload = copy.deepcopy(payload_template)
    prompt_graph = payload["prompt"]

    clip_node = prompt_graph.get(clip_node_id)
    if not clip_node or "inputs" not in clip_node:
        raise ValueError(f"CLIP node '{clip_node_id}' is missing or malformed.")

    clip_inputs = clip_node["inputs"]
    if text_input_key not in clip_inputs:
        raise KeyError(
            f"Input key '{text_input_key}' not found in CLIP node '{clip_node_id}'."
        )
    clip_inputs[text_input_key] = prompt_text

    # Optional: stamp SaveImage / SaveVideo nodes for deterministic sequence naming.
    seq_prefix = f"{output_prefix}_{index:04d}"
    for node in prompt_graph.values():
        if node.get("class_type") not in ("SaveImage", "SaveVideo"):
            continue
        inputs = node.get("inputs") or {}
        if "filename_prefix" in inputs:
            inputs["filename_prefix"] = seq_prefix

    payload["client_id"] = payload.get("client_id") or str(uuid.uuid4())
    return payload


async def queue_prompt(
    session: aiohttp.ClientSession,
    comfy_url: str,
    payload: dict[str, Any],
    index: int,
) -> tuple[int, str]:
    async with session.post(f"{comfy_url.rstrip('/')}/prompt", json=payload) as resp:
        body = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"Prompt {index} failed ({resp.status}): {body}")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {}
        prompt_id = parsed.get("prompt_id", "unknown")
        return index, str(prompt_id)


async def run(args: argparse.Namespace) -> None:
    prompts = read_prompts(args.prompts, args.delimiter)
    selected_template, selected_clip_node, selected_model, model_source = (
        resolve_template_and_clip_node(args)
    )
    template = load_template(selected_template)
    prompt_graph = template["prompt"]
    clip_node_id = find_clip_node_id(prompt_graph, selected_clip_node)
    resolved_output_prefix = resolve_output_prefix(args)

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    timeout = aiohttp.ClientTimeout(total=60)
    connector = aiohttp.TCPConnector(limit=max(1, args.concurrency))

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def worker(i: int, prompt_text: str) -> tuple[int, str]:
            payload = inject_prompt(
                payload_template=template,
                prompt_text=prompt_text,
                clip_node_id=clip_node_id,
                text_input_key=args.text_input_key,
                output_prefix=resolved_output_prefix,
                index=i,
            )
            async with semaphore:
                return await queue_prompt(session, args.comfy_url, payload, i)

        tasks = [worker(i, prompt) for i, prompt in enumerate(prompts, start=1)]
        results = await asyncio.gather(*tasks)

    for i, prompt_id in sorted(results, key=lambda x: x[0]):
        print(f"Queued prompt {i}: prompt_id={prompt_id}")
    print(f"Workflow preset: {selected_model}")
    print(f"Model source: {model_source}")
    print(f"Workflow file used: {selected_template}")
    print(f"CLIPTextEncode node used: {clip_node_id}")
    print(f"Save filename_prefix used: {resolved_output_prefix}")
    if selected_model == "ltx23":
        print(
            "All clip prompts queued. ComfyUI will write each segment under its output folder "
            "(SaveVideo node)."
        )
        if not args.no_concat_ltx_final:
            comfy_root = resolve_comfy_output_root(args, selected_model)
            history_timeout = aiohttp.ClientTimeout(total=120)
            poll_interval = 2.0
            results_sorted = sorted(results, key=lambda x: x[0])
            async with aiohttp.ClientSession(timeout=history_timeout) as hist_session:
                wait_tasks = [
                    wait_for_ltx_clip_file(
                        hist_session,
                        args.comfy_url,
                        prompt_id,
                        comfy_root,
                        clip_index,
                        args.ltx_wait_timeout,
                        poll_interval,
                    )
                    for clip_index, prompt_id in results_sorted
                ]
                clip_paths_ordered = await asyncio.gather(*wait_tasks)

            master_rel = Path(resolved_output_prefix.replace("\\", "/")).parent
            master_path = (
                comfy_root / master_rel / f"{args.final_master_name}.mp4"
            ).resolve()
            print(
                f"Assembling master video {args.final_width}x{args.final_height} @ "
                f"{args.final_fps} fps with ffmpeg..."
            )
            await asyncio.to_thread(
                run_ffmpeg_concat_clips,
                clip_paths_ordered,
                master_path,
                args.final_width,
                args.final_height,
                args.final_fps,
            )
            print(f"LTX master video written: {master_path}")
    else:
        print(
            "All prompts queued. ComfyUI will write images to its configured output directory."
        )


def main() -> int:
    args = parse_args()
    try:
        asyncio.run(run(args))
        return 0
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
