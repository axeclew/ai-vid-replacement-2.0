from __future__ import annotations

from pathlib import Path


def transcribe(audio_path: str | Path, model_size: str = "base") -> str:
    """Transcribe speech to text using Whisper."""
    import whisper

    model = whisper.load_model(model_size)
    result = model.transcribe(str(audio_path))
    return result["text"].strip()


def synthesize_speech(
    text: str,
    output_path: str | Path,
    engine: str = "coqui",
    voice: str = "default",
) -> Path:
    """
    Generate a new, synthetic voice track reading `text`.

    This deliberately does NOT clone the speaker's real voice — it uses a
    stock/synthetic TTS voice, so the output audio identity is synthetic,
    consistent with the avatar's synthetic visual identity.
    """
    output_path = Path(output_path)

    if engine == "coqui":
        from TTS.api import TTS

        tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False)
        speaker = voice if voice != "default" else tts.speakers[0]
        tts.tts_to_file(text=text, speaker=speaker, file_path=str(output_path))
    else:
        import pyttsx3

        engine_obj = pyttsx3.init()
        if voice != "default":
            for v in engine_obj.getProperty("voices"):
                if voice.lower() in v.name.lower():
                    engine_obj.setProperty("voice", v.id)
                    break
        engine_obj.save_to_file(text, str(output_path))
        engine_obj.runAndWait()

    return output_path
