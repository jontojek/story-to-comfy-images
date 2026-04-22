#!/usr/bin/env python3
"""Generate scene or video-clip prompts from a story using Gemini.

Pipeline phase covered:
1) Read story text
2) Ask Gemini to extract scenes/clips and write prompts tuned for Flux2Klein, Z-image, or LTX-2.3
3) Save prompts separated by ### into prompts_draft.txt
4) Stop for human review
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from google import genai
from google.genai import types
from pipeline_settings import get_default_settings_path, save_settings

DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_DELIMITER = "###"
WORDS_PER_PAGE = 250
PHOTO_STYLE_PRESETS = {
    "editorial": (
        "modern editorial photography look, clean composition, polished commercial "
        "lighting, high dynamic range realism"
    ),
    "documentary": (
        "documentary photojournalism look, natural available light feel, authentic "
        "textures, restrained post-processing"
    ),
    "fashion": (
        "high-end fashion photography look, controlled studio or location lighting, "
        "crisp subject separation, premium tonal contrast"
    ),
    "neo-noir": (
        "neo-noir photographic look, dramatic low-key lighting, deep shadows, selective "
        "color accents, moody contrast"
    ),
    "cinematic-realism": (
        "cinematic still-photography realism, filmic color science, intentional lens "
        "character, grounded physical lighting"
    ),
}
TEXT_OVERLAY_MODES = ("none", "caption-only", "dialogue-only", "mixed")
RENDER_MODELS = ("flux2klein", "z-image", "ltx23")

MODEL_PROMPT_GUIDANCE = {
    "flux2klein": (
        "Target model is FLUX.2 [klein]. Prioritize coherent prose with strong lighting "
        "control and cinematic still-photography realism. Keep prompts concise-to-moderate "
        "length while preserving subject clarity and scene continuity."
    ),
    "z-image": (
        "Target model is Z-image Turbo. Use long, structured, highly explicit positive "
        "prompts with concrete constraints in the prompt body. Emphasize composition, shot "
        "type, subject traits, clothing, environment, lighting, mood, style, and technical "
        "notes. Do not rely on negative-prompt fields; encode quality/safety constraints "
        "directly in positive prose (for example: no logos, no watermark, no random text). "
        "Keep language clear and instructional rather than poetic."
    ),
    "ltx23": (
        "Target model is LTX-2.3 text-to-video. Each output block is one video clip prompt: "
        "write one long, detailed flowing paragraph in present tense, as a shot description "
        "for a cinematographer. Match detail density to implied clip length — short prompts "
        "for long clips under-direct the model. Include: shot scale and angle, environment, "
        "lighting and color, clear action from beginning to end, character looks and "
        "performance through physical cues (not only abstract emotion words), explicit camera "
        "movement and timing, and audio (ambience, music character, speech). For dialogue, "
        "use short quoted phrases with brief acting or pause beats between lines. Prefer "
        "cinematic vocabulary (tracking shot, shallow depth of field, golden hour, rim light). "
        "Avoid contradictory directions, overcrowded actions, and conflicting lighting. "
        "On-screen readable text and logos are unreliable — describe title cards as stylized "
        "typography intent, not guaranteed literal spelling in every frame."
    ),
}

SYSTEM_INSTRUCTION_STILL = """You are a still-image scene extraction and prompt writing engine.

Your job is to convert a narrative story into a dynamic number of distinct scene prompts.
You MUST choose scene count based on story length, narrative beats, emotional pacing, and transitions.
Do not force a fixed number of scenes.

Still-image constraints:
- Every prompt must describe a single frozen moment for one generated image.
- Do not describe camera movement or temporal transitions (no pan, dolly, zoom, tracking shot, montage, "then", "next", or "meanwhile").
- Keep each prompt self-contained as one frame.
- Use cinematic photography language for a single still frame, not video direction.

Photographic realism constraints:
- Write each prompt as if it is a professionally captured photograph.
- Include realistic camera details near the specific-details section, such as camera body class, lens focal length, aperture, shutter speed, ISO, and depth-of-field behavior when appropriate.
- Include physically plausible optical behavior (bokeh, lens compression, subtle film grain, dynamic range, highlight rolloff) only when it matches the scene.
- Prefer concrete photographic phrasing like "captured on a full-frame mirrorless camera with an 85mm prime at f/1.8, ISO 400, 1/250s".
- Keep camera language natural in prose; never use keyword-tag lists.

Character continuity constraints:
- If character guidance is provided, treat those descriptions as mandatory canon for every scene where that character appears.
- Keep facial structure, age cues, hair, and signature wardrobe/accessories consistent across prompts unless the story explicitly changes them.

Text overlay constraints:
- Follow explicit text guidance provided by the user.
- If text overlay is enabled, include short on-image text suggestions in each prompt suitable for a picture-book, photo-essay, or graphic-novel feel.
- Keep overlay text minimal and legible (typically one short line, optionally two short lines).
- Place text guidance naturally in prose (for example: subtle caption at lower third, small dialogue bubble near speaker).
- If text overlay is disabled, explicitly avoid any written words, captions, signs, subtitles, or dialogue text in the generated image.
- Exception: if opening title guidance is provided, the first prompt must include the story title-and-author text at the top in large, legible typography.

Strict FLUX.2 [klein] prompt architecture requirements:
1) Prose over keywords: write flowing natural-language prose only, never comma-separated keyword tags.
2) Structural hierarchy in exact order:
   Main Subject -> Action/Setting -> Specific Details -> Lighting -> Atmosphere
   The order is mandatory and should be obvious in each paragraph.
3) Lighting dominance: include explicit, concrete lighting direction in every prompt.
   Example style only: "soft diffused light from a large window camera-left".
4) Style annotations at the very end of each paragraph:
   Append exactly in this form: "Style: <style>. Mood: <mood>."

Output formatting rules:
- Return only scene prompt paragraphs.
- Separate each paragraph using a single line containing exactly: ###
- Do not number scenes.
- Do not include JSON, markdown bullets, or commentary.
"""

SYSTEM_INSTRUCTION_LTX23 = """You are a cinematic video shot prompt writer for LTX-2.3 text-to-video.

Your job is to convert a narrative story into a dynamic number of distinct video clip prompts.
You MUST choose clip count based on story length, narrative beats, emotional pacing, and transitions.
Do not force a fixed number of clips.

Video and temporal constraints:
- Each prompt describes one continuous video clip as a single flowing paragraph in present tense.
- Describe action as a clear sequence from start to end (what happens first, then next).
- Camera movement is encouraged when it serves the story: push-in, pull-back, pan, tilt, track, handheld drift, static hold, etc.
- Describe how and when the camera moves relative to the subject.

LTX-2.3 prompting style (high detail):
- Long, specific paragraphs outperform short vague ones; align richness of description with how much happens in the clip.
- Establish the shot (scale, angle, lens character), set the scene (environment, lighting, palette, textures, atmosphere), then action, then character performance.
- Express emotion through physical performance cues (posture, eyes, breath, gestures), not only abstract labels like "sad" or "angry".
- Include audio when relevant: ambience, music mood, foley, and speech. Put spoken dialogue in quotation marks; break longer speech into shorter phrases with acting directions or pauses between them.
- Use cinematic language the model understands: golden hour, noir contrast, shallow depth of field, rack focus, etc.

Character continuity:
- If character guidance is provided, treat it as mandatory canon whenever that character appears.

Text and typography:
- Follow the user's text overlay mode for on-screen typography, captions, or subtitles.
- Readable on-screen text can be imperfect in video; describe intent clearly. Avoid relying on tiny illegible text.

Closing line for every clip paragraph (same as still pipeline):
- End each paragraph with exactly: Style: <style>. Mood: <mood>.

Output formatting rules:
- Return only clip prompt paragraphs.
- Separate each paragraph using a single line containing exactly: ###
- Do not number clips.
- Do not include JSON, markdown bullets, or commentary.
"""


def build_system_instruction(render_model: str) -> str:
    if render_model == "ltx23":
        return SYSTEM_INSTRUCTION_LTX23
    return SYSTEM_INSTRUCTION_STILL


USER_TEMPLATE = """Story to analyze:
---
{story_text}
---

Scene count guidance:
{scene_guidance}

Photo style guidance:
{photo_style_guidance}

Character consistency guidance:
{character_guidance}

Text overlay guidance:
{text_overlay_guidance}

Model-specific guidance:
{model_prompt_guidance}

Opening title guidance:
{opening_title_guidance}

Now produce scene prompts that follow all constraints.
Remember to separate each prompt paragraph with a line containing exactly: {delimiter}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scene or LTX video clip prompts from a story text file."
    )
    parser.add_argument(
        "--story",
        type=Path,
        default=Path("story.txt"),
        help="Input story text file path (default: story.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prompts_draft.txt"),
        help="Output draft prompt file path (default: prompts_draft.txt)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--delimiter",
        default=DEFAULT_DELIMITER,
        help=f"Prompt delimiter line (default: {DEFAULT_DELIMITER})",
    )
    parser.add_argument(
        "--target-scenes",
        type=int,
        default=None,
        help=(
            "Optional explicit target scene count. If omitted, script asks interactively "
            "using page/word estimate."
        ),
    )
    parser.add_argument(
        "--scenes-per-page",
        type=float,
        default=3.0,
        help="Used for suggestion only (default: 3.0 prompts per 250-word page)",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help=(
            "Do not ask for scene count interactively. Uses target-scenes if provided, "
            "otherwise uses estimated count from scenes-per-page."
        ),
    )
    parser.add_argument(
        "--photo-style",
        default=None,
        help=(
            "Photo look preset: editorial, documentary, fashion, neo-noir, "
            "cinematic-realism. You can also pass custom text. If omitted, "
            "script asks interactively unless --non-interactive is used."
        ),
    )
    parser.add_argument(
        "--text-overlay",
        choices=["yes", "no"],
        default=None,
        help=(
            "Include short on-image text guidance in prompts ('yes' or 'no'). "
            "If omitted, script asks interactively unless --non-interactive is used."
        ),
    )
    parser.add_argument(
        "--text-overlay-mode",
        choices=list(TEXT_OVERLAY_MODES),
        default=None,
        help=(
            "Granular text mode: none, caption-only, dialogue-only, mixed. "
            "If set, it overrides --text-overlay."
        ),
    )
    parser.add_argument(
        "--render-model",
        choices=list(RENDER_MODELS),
        default=None,
        help=(
            "Target model prompt profile: flux2klein (still), z-image (still), "
            "ltx23 (LTX-2.3 video). If omitted, script asks interactively unless "
            "--non-interactive is used."
        ),
    )
    parser.add_argument(
        "--settings-file",
        type=Path,
        default=get_default_settings_path(),
        help="Path to shared pipeline settings file.",
    )
    parser.add_argument(
        "--no-save-settings",
        action="store_true",
        help="Do not save selected model choice for Phase 2.",
    )
    return parser.parse_args()


def load_story(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Story file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Story file is empty: {path}")
    return text


def extract_opening_title_line(story_text: str) -> str | None:
    for line in story_text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def resolve_story_path(initial_path: Path, non_interactive: bool) -> Path:
    if non_interactive:
        return initial_path

    user_value = input(
        f"Enter input story text file path (press Enter for {initial_path}): "
    ).strip()
    if not user_value:
        return initial_path
    return Path(user_value)


def resolve_render_model_choice(render_model: str | None, non_interactive: bool) -> str:
    default_model = "flux2klein"
    if render_model:
        return render_model
    if non_interactive:
        return default_model

    print("\nChoose target model prompt profile:")
    print("  1) flux2klein (still image)")
    print("  2) z-image (still image)")
    print("  3) ltx23 (LTX-2.3 video)")
    print(f"Press Enter for default: {default_model}")

    user_value = input("Model choice: ").strip().lower()
    if not user_value:
        return default_model
    if user_value in RENDER_MODELS:
        return user_value
    if user_value.isdigit():
        idx = int(user_value)
        if 1 <= idx <= len(RENDER_MODELS):
            return RENDER_MODELS[idx - 1]
    raise ValueError(
        "Invalid model choice. Choose flux2klein, z-image, or ltx23 "
        "(or 1 / 2 / 3)."
    )


def count_words(text: str) -> int:
    return len(text.split())


def estimate_pages(word_count: int) -> float:
    return word_count / WORDS_PER_PAGE if word_count else 0.0


def estimate_scene_count(word_count: int, scenes_per_page: float) -> int:
    pages = estimate_pages(word_count)
    return max(1, int(round(pages * scenes_per_page)))


def resolve_target_scene_count(
    word_count: int,
    scenes_per_page: float,
    target_scenes: int | None,
    non_interactive: bool,
) -> int:
    suggested = estimate_scene_count(word_count, scenes_per_page)
    pages = estimate_pages(word_count)

    if target_scenes is not None:
        if target_scenes < 1:
            raise ValueError("--target-scenes must be at least 1.")
        return target_scenes

    if non_interactive:
        return suggested

    pages_display = math.ceil(pages * 10) / 10 if pages else 0.0
    print(f"Story length: ~{word_count} words (~{pages_display} pages at 250 words/page).")
    print(
        f"Suggested scene count: {suggested} "
        f"(using {scenes_per_page:g} prompts per page)."
    )
    user_value = input(
        "Enter target number of prompts (press Enter to use suggested): "
    ).strip()
    if not user_value:
        return suggested
    if not user_value.isdigit() or int(user_value) < 1:
        raise ValueError("Scene count must be a positive whole number.")
    return int(user_value)


def extract_characters(
    model_name: str,
    api_key: str,
    story_text: str,
) -> list[dict[str, str]]:
    client = genai.Client(api_key=api_key)
    extraction_prompt = f"""Analyze the story below and identify every named or clearly distinct recurring character.

For each character, provide:
- name
- short role identifier (5 words max)

Return ONLY a JSON array with this exact shape:
[{{"name":"Character Name","role":"short role"}}]

Story:
---
{story_text}
---
"""
    response = client.models.generate_content(
        model=model_name,
        contents=extraction_prompt,
    )
    text = (response.text or "").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Fallback for occasional markdown code fences from model output.
        cleaned = text.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return []

    if not isinstance(parsed, list):
        return []

    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        role = str(item.get("role", "")).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"name": name, "role": role or "character"})
    return normalized


def resolve_character_profiles(
    characters: list[dict[str, str]],
    non_interactive: bool,
) -> list[dict[str, str]]:
    if not characters:
        return []

    print("\nDetected story characters:")
    for idx, character in enumerate(characters, start=1):
        print(f"  {idx}) {character['name']} - {character['role']}")

    if non_interactive:
        return []

    print(
        "\nCharacter consistency step:\n"
        "For each character, enter a short visual description so the model keeps them consistent.\n"
        "Tip: include age range, build, hair, face traits, clothing palette, and signature accessories.\n"
        "Press Enter to skip any character."
    )

    profiles: list[dict[str, str]] = []
    for character in characters:
        description = input(
            f"{character['name']} ({character['role']}) description: "
        ).strip()
        if description:
            profiles.append(
                {
                    "name": character["name"],
                    "role": character["role"],
                    "description": description,
                }
            )
    return profiles


def build_character_guidance(character_profiles: list[dict[str, str]]) -> str:
    if not character_profiles:
        return (
            "No custom character profiles supplied. Keep recurring named characters "
            "visually consistent across prompts."
        )

    lines = [
        "Use these fixed character profiles whenever the character appears. "
        "Treat them as continuity canon unless the story explicitly describes a change:"
    ]
    for profile in character_profiles:
        lines.append(
            f"- {profile['name']} ({profile['role']}): {profile['description']}"
        )
    return "\n".join(lines)


def resolve_text_overlay_mode(
    text_overlay: str | None,
    text_overlay_mode: str | None,
    non_interactive: bool,
    render_model: str,
) -> str:
    if text_overlay_mode is not None:
        return text_overlay_mode.strip().lower()

    # Backward compatibility for existing yes/no flag.
    if text_overlay is not None:
        return "mixed" if text_overlay.strip().lower() == "yes" else "none"

    if non_interactive:
        return "none"

    medium = "video clips" if render_model == "ltx23" else "still images"
    print(
        f"\nText overlay mode:\n"
        f"Choose how written text should appear in {medium} for visual narrative."
    )
    print("  1) none (no text in image)")
    print("  2) caption-only (short caption text)")
    print("  3) dialogue-only (short speech/dialogue text)")
    print("  4) mixed (caption and/or dialogue when useful)")
    print("Press Enter for default: none")
    user_value = input("Text mode choice: ").strip().lower()

    if not user_value:
        return "none"
    if user_value in TEXT_OVERLAY_MODES:
        return user_value
    if user_value.isdigit():
        idx = int(user_value)
        if 1 <= idx <= len(TEXT_OVERLAY_MODES):
            return TEXT_OVERLAY_MODES[idx - 1]
    raise ValueError("Invalid text mode. Choose none, caption-only, dialogue-only, or mixed.")


def build_text_overlay_guidance(text_overlay_mode: str, render_model: str) -> str:
    if render_model == "ltx23":
        if text_overlay_mode == "none":
            return (
                "Mode: none (video). Avoid describing burned-in subtitles, lower-thirds, "
                "or readable signage unless required for the opening title card. Spoken "
                "dialogue may still be written in quotation marks as audio performance, not "
                "as on-screen captions. Exception: opening title guidance for clip 1."
            )
        if text_overlay_mode == "caption-only":
            return (
                "Mode: caption-only (video). Describe one brief on-screen caption or title "
                "card line per clip where useful; keep typography large and simple. Note "
                "on-screen text may not render perfectly."
            )
        if text_overlay_mode == "dialogue-only":
            return (
                "Mode: dialogue-only (video). Integrate spoken lines in quotation marks "
                "inside the flowing paragraph with acting beats and pauses between short "
                "phrases (LTX-style). Prefer audible speech over on-screen subtitle text."
            )
        return (
            "Mode: mixed (video). Combine spoken dialogue in quotes with optional brief "
            "on-screen text or caption beats where it serves the story; keep text minimal "
            "and legible when described."
        )

    if text_overlay_mode == "none":
        return (
            "Mode: none. Do not include any on-image written text. Avoid captions, "
            "subtitles, dialogue bubbles, signage text, or typographic elements, "
            "except any required opening title-and-author overlay."
        )
    if text_overlay_mode == "caption-only":
        return (
            "Mode: caption-only. Include one brief on-image caption line per prompt, "
            "kept minimal, legible, and narrative-focused. Do not add dialogue bubbles."
        )
    if text_overlay_mode == "dialogue-only":
        return (
            "Mode: dialogue-only. Include a very short on-image dialogue line when a "
            "speaker is present. Do not add extra caption narration."
        )
    return (
        "Mode: mixed. Include minimal on-image text cues as needed, using a short caption "
        "or brief dialogue line per scene when it helps narrative continuity."
    )


def build_opening_title_guidance(opening_title_line: str | None, render_model: str) -> str:
    if not opening_title_line:
        return "No opening title line provided."
    if render_model == "ltx23":
        return (
            "For clip 1 only, open with a cinematic title-card beat: describe large, bold "
            "opening typography centered or upper-third, reading exactly: "
            f"\"{opening_title_line}\". Treat it as a motion title sequence (fade or resolve "
            "into the first story imagery). On-screen text may be imperfect; prioritize "
            "clear composition and readable scale in the description. Later clips follow "
            "the selected text mode."
        )
    return (
        "For prompt 1 only, include this exact visible top-of-image title text in large type: "
        f"\"{opening_title_line}\". Keep it prominent and readable to establish a graphic-novel "
        "opening frame. For later prompts, follow the selected text mode."
    )


def resolve_photo_style_guidance(photo_style: str, render_model: str) -> str:
    key = photo_style.strip().lower()
    preset = PHOTO_STYLE_PRESETS.get(key)
    if preset:
        base = f"Use '{key}' style: {preset}."
    else:
        base = (
            "Use this custom visual style direction: "
            f"{photo_style.strip()}."
        )
    if render_model == "ltx23":
        return base + " Interpret as motion-picture look, lighting, and color grading for video."
    return base + " Interpret as photorealistic still-imaging look."


def resolve_photo_style_choice(photo_style: str | None, non_interactive: bool) -> str:
    default_style = "cinematic-realism"
    preset_names = list(PHOTO_STYLE_PRESETS.keys())

    if photo_style and photo_style.strip():
        return photo_style.strip()
    if non_interactive:
        return default_style

    print("\nChoose photo style:")
    for idx, name in enumerate(preset_names, start=1):
        print(f"  {idx}) {name}")
    print("  c) custom text")
    print(f"Press Enter for default: {default_style}")

    user_value = input("Style choice: ").strip()
    if not user_value:
        return default_style

    if user_value.lower() == "c":
        custom = input("Enter custom photo style text: ").strip()
        if not custom:
            raise ValueError("Custom photo style cannot be empty.")
        return custom

    if user_value.isdigit():
        idx = int(user_value)
        if 1 <= idx <= len(preset_names):
            return preset_names[idx - 1]
        raise ValueError("Invalid style number selected.")

    lowered = user_value.lower()
    if lowered in PHOTO_STYLE_PRESETS:
        return lowered

    # If user typed non-preset text directly, treat it as custom style.
    return user_value


def build_prompt(
    story_text: str,
    delimiter: str,
    scene_guidance: str,
    photo_style_guidance: str,
    character_guidance: str,
    text_overlay_guidance: str,
    model_prompt_guidance: str,
    opening_title_guidance: str,
) -> str:
    return USER_TEMPLATE.format(
        story_text=story_text,
        delimiter=delimiter,
        scene_guidance=scene_guidance,
        photo_style_guidance=photo_style_guidance,
        character_guidance=character_guidance,
        text_overlay_guidance=text_overlay_guidance,
        model_prompt_guidance=model_prompt_guidance,
        opening_title_guidance=opening_title_guidance,
    )


def generate_prompts(
    model_name: str,
    api_key: str,
    story_text: str,
    delimiter: str,
    target_scene_count: int,
    photo_style: str,
    character_profiles: list[dict[str, str]],
    text_overlay_mode: str,
    render_model: str,
    opening_title_line: str | None,
) -> str:
    client = genai.Client(api_key=api_key)
    unit = "video clip prompts" if render_model == "ltx23" else "scene prompts"
    scene_guidance = (
        f"Aim for exactly {target_scene_count} {unit} unless the story structure "
        "strongly requires plus or minus one for coherence."
    )
    photo_style_guidance = resolve_photo_style_guidance(photo_style, render_model)
    character_guidance = build_character_guidance(character_profiles)
    text_overlay_guidance = build_text_overlay_guidance(text_overlay_mode, render_model)
    model_prompt_guidance = MODEL_PROMPT_GUIDANCE[render_model]
    opening_title_guidance = build_opening_title_guidance(opening_title_line, render_model)
    response = client.models.generate_content(
        model=model_name,
        contents=build_prompt(
            story_text,
            delimiter,
            scene_guidance,
            photo_style_guidance,
            character_guidance,
            text_overlay_guidance,
            model_prompt_guidance,
            opening_title_guidance,
        ),
        config=types.GenerateContentConfig(
            system_instruction=build_system_instruction(render_model),
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response.")

    # Normalize accidental duplicate delimiter spacing.
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == delimiter:
            cleaned_lines.append(delimiter)
        else:
            cleaned_lines.append(line.rstrip())
    cleaned = "\n".join(cleaned_lines).strip()

    if delimiter not in cleaned:
        # Fallback: if model did not obey delimiter rule, keep full text as one prompt.
        cleaned = cleaned + f"\n{delimiter}\n"
    return cleaned


def main() -> int:
    args = parse_args()
    try:
        selected_story_path = resolve_story_path(
            initial_path=args.story,
            non_interactive=args.non_interactive,
        )
        story_text = load_story(selected_story_path)
        opening_title_line = extract_opening_title_line(story_text)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY environment variable.")
        selected_render_model = resolve_render_model_choice(
            render_model=args.render_model,
            non_interactive=args.non_interactive,
        )
        if not args.no_save_settings:
            save_settings(
                args.settings_file,
                {"last_render_model": selected_render_model},
            )
        word_count = count_words(story_text)
        target_scene_count = resolve_target_scene_count(
            word_count=word_count,
            scenes_per_page=args.scenes_per_page,
            target_scenes=args.target_scenes,
            non_interactive=args.non_interactive,
        )
        characters = extract_characters(
            model_name=args.model,
            api_key=api_key,
            story_text=story_text,
        )
        character_profiles = resolve_character_profiles(
            characters=characters,
            non_interactive=args.non_interactive,
        )
        selected_text_mode = resolve_text_overlay_mode(
            text_overlay=args.text_overlay,
            text_overlay_mode=args.text_overlay_mode,
            non_interactive=args.non_interactive,
            render_model=selected_render_model,
        )
        selected_photo_style = resolve_photo_style_choice(
            photo_style=args.photo_style,
            non_interactive=args.non_interactive,
        )
        prompts_text = generate_prompts(
            model_name=args.model,
            api_key=api_key,
            story_text=story_text,
            delimiter=args.delimiter,
            target_scene_count=target_scene_count,
            photo_style=selected_photo_style,
            character_profiles=character_profiles,
            text_overlay_mode=selected_text_mode,
            render_model=selected_render_model,
            opening_title_line=opening_title_line,
        )
        args.output.write_text(prompts_text + "\n", encoding="utf-8")
    except Exception as exc:  # pragma: no cover - CLI guard
        message = str(exc)
        if "not found" in message.lower() and "models/" in message.lower():
            message += (
                "\nTip: this model may be unavailable for your API project. "
                "Try --model gemini-2.5-flash."
            )
        print(f"Error: {message}", file=sys.stderr)
        return 1

    print(f"Wrote draft prompts to: {args.output}")
    print(f"Input story file used: {selected_story_path}")
    print(f"Target prompt count used: {target_scene_count}")
    print(f"Render model prompt profile: {selected_render_model}")
    if opening_title_line:
        print(f"Opening title line: {opening_title_line}")
    if not args.no_save_settings:
        print(f"Saved model choice to: {args.settings_file}")
    print(f"Photo style used: {selected_photo_style}")
    print(f"Character profiles provided: {len(character_profiles)}")
    print(f"Text overlay mode: {selected_text_mode}")
    print(
        "Pipeline paused for human review. Edit the draft and save as prompts_approved.txt."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
