```
   ____  _____       _       ____              _        ____  _  _    ___  _
  / ___||_   _|__   / \     / ___|___  _ __ __| |_   _ / ___|| || |  / _ \| |
  \___ \  | |/ _ \ / _ \   | |   / _ \| '__/ _` | | | | |    | || |_| | | | |
   ___) | | | (_) / ___ \  | |__| (_) | | | (_| | |_| | |___ |__   _| |_| | |___
  |____/  |_|\___/_/   \_\  \____\___/|_|  \__,_|\__, |\____|  |_|  \___/|_____|
                                                 |___/
          story  →  Gemini  →  you  →  ComfyUI  →  images or video
```

# story-to-comfy-images

Turn a plain-text story into **reviewed prompts**, then into **still images** (Flux or Z-Image) or **video clips** (LTX 2.3) using **Google Gemini** and a **local ComfyUI** server.

---

## If you are new here, read this first

| Step | Who does it | What happens |
|------|----------------|----------------|
| 1 | You | Write or paste your story into a `.txt` file. |
| 2 | Script + Gemini | Suggests how many prompts, lists characters, writes draft prompts. |
| 3 | You | Edit `prompts_draft.txt`, save as `prompts_approved.txt`. |
| 4 | Script + ComfyUI | Sends each prompt to ComfyUI and saves images or video segments. |

You do **not** need to be a programmer: you mostly run **two commands** and answer questions in the terminal.

---

## What you need installed

1. **Python 3.10+** (already common if you use ComfyUI).
2. **Google Gemini API key** — from [Google AI Studio](https://aistudio.google.com/) (free tier may apply; billing depends on your account).
3. **ComfyUI** running locally with the **API** enabled (default address used here: `http://127.0.0.1:8188`).
4. **Workflow models and files** that match the JSON files in `comfy_workflows/` (checkpoint names inside the graphs must exist on your machine).
5. **ffmpeg** (only if you use **LTX** and want one long **master** video file after all clips finish). [ffmpeg.org](https://ffmpeg.org/)

---

## Install project dependencies

Open a terminal **in this project folder**, then:

```bash
pip install -r requirements.txt
```

---

## Set your Gemini API key (once per terminal session)

**PowerShell**

```powershell
$env:GEMINI_API_KEY="paste_your_key_here"
```

**Command Prompt (cmd)**

```cmd
set GEMINI_API_KEY=paste_your_key_here
```

To check in cmd:

```cmd
echo %GEMINI_API_KEY%
```

If you see nothing, set the key again. **Never commit your API key** to GitHub — keep it in environment variables only.

---

## Phase 1 — Story to draft prompts

### 1) Put your story in a text file

Use any name you like; the script will **ask for the path** first (default is `story.txt` if you press Enter).

**Tip:** Put the **title and author on the first line**, for example:

```text
Jackals and Arabs by Franz Kafka

Then your story begins here...
```

That first line is used as a **big opening title** on the first still (or as an opening title-card beat for video prompts).

### 2) Run the prompt generator

```bash
python story_to_prompts.py
```

### 3) What the script will ask (in order)

1. **Path to your story** `.txt` file  
2. **Which renderer** the prompts should be optimized for  
   - `flux2klein` — still images, Flux workflow  
   - `z-image` — still images, Z-Image workflow  
   - `ltx23` — **video** clips, LTX 2.3 workflow ([official prompt tips](https://ltx.io/model/model-blog/ltx-2-3-prompt-guide))  
3. **How many prompts** (suggested from length; default is about **3 prompts per 250 words**)  
4. **Characters** — short visual description per character (helps consistency)  
5. **Text on the image** — none, captions only, dialogue only, or mixed  
6. **Photo / look style** — presets or custom wording  

When you finish, you get **`prompts_draft.txt`**. Each prompt is a **paragraph**; prompts are separated by a line containing exactly:

```text
###
```

Your chosen **model** is saved to **`.story_to_comfy_settings.json`** so Phase 2 can match it automatically (you can change it there or with flags).

### 4) Human review (important)

1. Open **`prompts_draft.txt`** in Notepad or any editor.  
2. Fix wording, continuity, or anything you do not like.  
3. Save a copy as **`prompts_approved.txt`** (same `###` separators).

Phase 2 **only** reads `prompts_approved.txt`.

---

## Phase 2 — Send prompts to ComfyUI

### 1) Start ComfyUI

Start ComfyUI the way you usually do, with **network / API** listening (often port **8188**).

### 2) Run the generator

```bash
python generate_images.py
```

### 3) What you may be asked

- **Which workflow preset** — if you did Phase 1 on the same machine, it can **reuse the saved model** from `.story_to_comfy_settings.json`.  
- **Run folder name** — groups files under something like `story_runs/<your_name>/` inside Comfy’s output tree.

Still images and video segments are written wherever **ComfyUI** is configured to save output (often a folder named `output` inside your Comfy install).

### LTX only — one long master video (1920 × 1088 @ 24 fps)

For **`ltx23`**, after **every** clip job has finished, the script can **wait** for Comfy’s history, then run **ffmpeg** to build **one** MP4:

- Resolution **1920×1088**, **24 fps**  
- Default filename pattern: **`master_1920x1088_24fps.mp4`** next to your segment files  

You must tell the script where Comfy’s **output root** is (the folder that already contains paths like `story_runs/...`):

- **Environment variable:** `COMFYUI_OUTPUT_DIR`  
- **Or flag:** `--comfy-output-dir "D:\ComfyUI\output"`  
- **Or:** answer the prompt once; it can save **`comfy_output_dir`** in `.story_to_comfy_settings.json`  

To **only** queue clips and **skip** the master assembly: `--no-concat-ltx-final`

---

## Project layout (what each file is for)

| Path | Role |
|------|------|
| `story_to_prompts.py` | Phase 1 — story → Gemini → `prompts_draft.txt` |
| `generate_images.py` | Phase 2 — `prompts_approved.txt` → ComfyUI queue (+ LTX ffmpeg join) |
| `pipeline_settings.py` | Small helper to read/write `.story_to_comfy_settings.json` |
| `comfy_workflows/` | Ready-made API workflow JSON for Flux, Z-Image, and LTX |
| `comfy_payload_template.json` | Optional; for advanced custom graphs |
| `requirements.txt` | Python packages to install |

---

## Useful command-line options (optional)

**Phase 1 — examples**

```bash
python story_to_prompts.py --render-model z-image --target-scenes 12
python story_to_prompts.py --model gemini-2.5-flash
python story_to_prompts.py --non-interactive
```

**Phase 2 — examples**

```bash
python generate_images.py --render-model flux2klein
python generate_images.py --comfy-url http://127.0.0.1:8188
python generate_images.py --ignore-saved-model
```

See `--help` on each script for the full list.

---

## When something goes wrong

| Symptom | What to try |
|---------|-------------|
| `Missing GEMINI_API_KEY` | Set the key again in **this** terminal window (see above). |
| Gemini `404` / model not found | Run with `--model gemini-2.5-flash` or another model your API project allows. |
| Comfy connection errors | Confirm ComfyUI is running and the URL matches (`--comfy-url`). |
| LTX master step fails | Install **ffmpeg**, put it on **PATH**, and set **`COMFYUI_OUTPUT_DIR`** or **`--comfy-output-dir`** to your real Comfy **output** folder. |
| Workflow errors in Comfy | Open the same JSON in ComfyUI and fix missing models or nodes; checkpoint paths must match your PC. |

---

## Git and privacy (before you push to GitHub)

- **Do not commit** API keys or `.env` files with secrets.  
- **`.story_to_comfy_settings.json`**, **`prompts_draft.txt`**, and **`prompts_approved.txt`** are listed in **`.gitignore`** so they are not committed by mistake.  
- If **`story.txt`** is private, exclude it from the commit in GitHub Desktop or add it to `.gitignore`.
