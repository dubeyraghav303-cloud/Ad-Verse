# pipeline/inference.py
"""
Mock inference pipeline for Ad-Verse.

Purpose:
- Provide a lightweight, runnable placeholder for the final TTS + T2V pipeline.
- Generates a short animated text-based ad video locally (no model weights needed).

Later you can replace `generate_mock_ad()` with actual model-based logic.
"""

import os
import hashlib
import random
import numpy as np
from moviepy.editor import TextClip, ColorClip, CompositeVideoClip, concatenate_videoclips
from moviepy.audio.AudioClip import AudioArrayClip


def _seed_from_text(prompt: str, seed: int) -> int:
    """Generate a reproducible numeric seed from text input."""
    h = hashlib.sha256((prompt + str(seed)).encode()).digest()
    return int.from_bytes(h[:4], "big")


def _split_prompt(prompt: str, n_slides: int = 6):
    """Split the input prompt into textual slides."""
    prompt = prompt.strip()
    if not prompt:
        return ["Your product. Your message.", "Call to action."]

    # split heuristics
    if "," in prompt:
        parts = [p.strip() for p in prompt.split(",") if p.strip()]
    elif " and " in prompt:
        parts = [p.strip() for p in prompt.split(" and ") if p.strip()]
    else:
        words = prompt.split()
        chunk = max(1, len(words) // n_slides)
        parts = [" ".join(words[i:i + chunk]) for i in range(0, len(words), chunk)]
    while len(parts) < n_slides:
        parts.append(parts[-1] + " •")
    return parts[:n_slides]


def _choose_colors(seed_int: int, n: int):
    """Generate pastel background colors."""
    random.seed(seed_int)
    return [(random.randint(40, 230), random.randint(40, 230), random.randint(40, 230)) for _ in range(n)]


def generate_mock_ad(prompt: str,
                     out_dir: str = "outputs",
                     style: str = "Minimal",
                     voice: str = "Neutral",
                     resolution: str = "1280x720",
                     fps: int = 24,
                     seed: int = 42) -> str:
    """
    Generate a 30s mock advertisement video and return its file path.
    This runs locally and is safe on CPU-only systems.
    """
    os.makedirs(out_dir, exist_ok=True)
    seed_int = _seed_from_text(prompt, seed)
    random.seed(seed_int)
    np.random.seed(seed_int)

    width, height = map(int, resolution.split("x"))
    total_duration = 30
    n_slides = 6
    slide_dur = total_duration / n_slides
    slides = _split_prompt(prompt, n_slides)
    colors = _choose_colors(seed_int, n_slides)

    clips = []
    for idx, text in enumerate(slides):
        bg = ColorClip(size=(width, height), color=colors[idx], duration=slide_dur)
        fontsize = int(min(width, height) * 0.09)
        try:
            txt = TextClip(text, fontsize=fontsize, font='Liberation-Sans',
                           color='white', size=(int(width * 0.9), None), method='caption', align='center')
        except Exception:
            txt = TextClip(text, fontsize=fontsize, color='white',
                           size=(int(width * 0.9), None), method='caption', align='center')

        def pos(t, h=height, dur=slide_dur):
            dy = 10 * np.sin(2 * np.pi * (t / dur))
            return ("center", h / 2 + dy)

        clips.append(CompositeVideoClip([bg, txt.set_position(pos)], size=(width, height)).set_duration(slide_dur))

    video = concatenate_videoclips(clips, method="compose", padding=-0.5)

    # Simple ambient "music" for mock feel
    try:
        sr = 22050
        t = np.linspace(0, total_duration, int(sr * total_duration))
        freqs = [220 + (seed_int % 120), 330, 440]
        audio = sum([0.02 * np.sin(2 * np.pi * f * t) for f in freqs]).astype(np.float32)
        audio_clip = AudioArrayClip(audio.reshape((-1, 1)), fps=sr)
        video = video.set_audio(audio_clip)
    except Exception:
        pass

    safe_name = hashlib.sha1((prompt + str(seed)).encode()).hexdigest()[:10]
    out_path = os.path.join(out_dir, f"adverse_mock_{safe_name}.mp4")
    video.write_videofile(out_path, fps=fps, codec="libx264", audio_codec="aac",
                          preset="medium", verbose=False, logger=None)
    return out_path
