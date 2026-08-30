# AI Avatar Video Replacement

Upload a video and this tool produces a new version where the person is replaced by a
**synthetic AI avatar**, moving in front of a **new background**, speaking with a
**synthetic voice**. The avatar is a generated or user-supplied *fictional character* —
the pipeline never copies a real person's face or voice into the output.

## How it works

1. **Motion tracking** — MediaPipe Pose extracts a body skeleton from the source video. Only
   motion is used; no facial identity data is kept.
2. **Person segmentation** — U²-Net (via `rembg`) isolates the person's silhouette per frame.
3. **Avatar rendering** — a single synthetic-character reference image (generated from a text
   prompt, or supplied by you) is rendered into each frame's pose using Stable Diffusion +
   ControlNet (OpenPose) + IP-Adapter, so the character's appearance stays consistent across
   frames while following the source motion.
4. **Background compositing** — the rendered avatar is composited onto a new background image
   using the segmentation mask.
5. **Voice replacement** — the original audio is transcribed with Whisper, and the same words
   are re-spoken in a synthetic TTS voice (Coqui TTS, or `pyttsx3` as a lightweight fallback).
6. **Mux** — the new video and new audio are combined into the final output.

## Quick start

```bash
chmod +x setup.sh
./setup.sh
```

## Usage

### CLI

```bash
# Generate a brand-new character from a description
python main.py input.mp4 background.jpg --avatar-prompt "a friendly animated-style host" -o output/result.mp4

# Or reuse a character reference image you already have
python main.py input.mp4 background.jpg --avatar-image character.png -o output/result.mp4
```

```
python main.py --help

  input_video           source video
  background_image      new background image
  -o, --output          output path
  --avatar-image        reference image of a synthetic character
  --avatar-prompt       text description to generate a synthetic character
  --voice               TTS voice name
  --seed                random seed
```

### Web UI

```bash
python app.py
```

Open http://localhost:7860.

## Project structure

```
ai_avatar_replace/
  config.py             # Configuration loader
  pipeline.py           # Orchestrates the full replacement pipeline
  core/
    motion.py            # Pose tracking + person segmentation
    avatar_render.py      # Pose-guided synthetic avatar rendering + background compositing
    voice.py              # Speech-to-text + synthetic text-to-speech
    video_io.py            # Video read/write, audio extraction/muxing
    temporal.py             # Mask smoothing/feathering
main.py                 # CLI entry point
app.py                  # Gradio web UI
config/default.yaml
```

## What this tool deliberately does NOT do

- It does not swap a real person's face onto footage of another real person.
- It does not clone a real person's voice — the output voice is a stock synthetic voice
  reading a transcript, not a copy of anyone's actual voice.
- It carries no facial-identity model (e.g. face recognition/swap networks) — only body pose
  and a plain silhouette mask are extracted from the source video.

## Ethics & legal notice

Even with a fully synthetic avatar, this still produces AI-generated video of a real person's
movements. Use only:

- With **consent** from anyone appearing in the source video
- For **lawful** purposes
- With clear **disclosure** that the output is AI-generated when you share it

Do not use this to impersonate, deceive, or harm anyone.

## License

MIT — see repository for details.
