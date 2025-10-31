import os
import json
from typing import List

import streamlit as st
from dotenv import load_dotenv

import google.generativeai as genai
import requests

# ElevenLabs SDK (simple generate interface)
try:
    from elevenlabs import generate, set_api_key
except Exception:  # Fallback in case of version differences
    generate = None
    set_api_key = None


# -----------------------------
# Helpers
# -----------------------------

def parse_sentence_list(text: str) -> List[str]:
    """Parse a Python/JSON list of strings from model text output.

    The model is instructed to return a simple Python list like:
    ["Sentence one.", "Sentence two."]
    But sometimes providers wrap content in code fences or add extra text.
    This function extracts the first [...] block and JSON-loads it.
    """
    if not text:
        return []

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []

    snippet = text[start : end + 1]
    # Normalize single quotes to double for JSON parsing if needed
    json_candidate = snippet.replace("'", '"')
    try:
        parsed = json.loads(json_candidate)
        if isinstance(parsed, list):
            # Ensure all items are strings and strip whitespace
            return [str(s).strip() for s in parsed if str(s).strip()]
    except Exception:
        pass
    return []


def extract_keywords(sentence: str, max_keywords: int = 3) -> List[str]:
    """Naive keyword extraction: pick 2-3 longest words excluding common stopwords."""
    stop = {
        "the","a","an","and","or","but","if","then","than","that","this","these","those",
        "in","on","at","for","from","to","with","without","of","by","as","is","are","was","were",
        "it","its","be","been","being","into","about","over","under","up","down","out","so","such",
        "we","you","they","he","she","i","our","your","their"
    }
    words = [w.strip(".,!?;:""'()[]{}-").lower() for w in sentence.split()]
    words = [w for w in words if w and w not in stop and w.isalpha()]
    words = sorted(words, key=len, reverse=True)
    return words[:max_keywords] or ([sentence.split()[0]] if sentence.split() else [])


def pixabay_video_url(api_key: str, query: str) -> str | None:
    """Search Pixabay videos and return a medium/small URL if found."""
    try:
        params = {
            "key": api_key,
            "q": query,
            "safesearch": "true",
            "video_type": "film",
            "per_page": 5,
        }
        resp = requests.get("https://pixabay.com/api/videos/", params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits") or []
        if not hits:
            return None
        # choose first hit; prefer medium then small
        videos = hits[0].get("videos", {})
        for quality in ("medium", "small", "large"):
            if quality in videos and "url" in videos[quality]:
                return videos[quality]["url"]
    except Exception:
        return None
    return None


def download_file(url: str, dest_path: str) -> bool:
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        return False


# -----------------------------
# App
# -----------------------------

def main() -> None:
    st.set_page_config(page_title="AI Video Storyteller", page_icon="🎬")
    st.title("AI Video Storyteller 🎬")

    # Load env
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    eleven_api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    pixabay_api_key = os.getenv("PIXABAY_API_KEY", "").strip()

    # Simple environment checks (shown collapsed)
    with st.expander("Environment status", expanded=False):
        st.write({
            "GEMINI_API_KEY": "✅ set" if bool(gemini_api_key) else "❌ missing",
            "ELEVENLABS_API_KEY": "✅ set" if bool(eleven_api_key) else "❌ missing",
            "PIXABAY_API_KEY": "✅ set" if bool(pixabay_api_key) else "❌ missing (fallback to black clips)",
        })

    # Configure Gemini client
    if not gemini_api_key:
        st.warning("Please set GEMINI_API_KEY in your .env file.")
    else:
        genai.configure(api_key=gemini_api_key)

    # Configure ElevenLabs client (simple interface)
    if not eleven_api_key:
        st.warning("Please set ELEVENLABS_API_KEY in your .env file.")
    else:
        if set_api_key is not None:
            set_api_key(eleven_api_key)

    # UI inputs
    topic = st.text_input("What is your video topic?", placeholder="e.g., The life cycle of a star")
    generate_btn = st.button("Generate Story", type="primary")

    sentences: List[str] = []
    voiceover_bytes: bytes | None = None

    if generate_btn:
        if not topic.strip():
            st.error("Please enter a topic.")
            st.stop()
        if not gemini_api_key or not eleven_api_key:
            st.error("Missing API keys. Please update your .env file and restart.")
            st.stop()

        # 1) Generate script via Gemini
        with st.spinner("Generating script..."):
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "You are a documentary scriptwriter. Create a short, ~30-second,"
                " documentary-style narration script about the topic below."
                " Return ONLY a valid Python/JSON list of individual sentences (no intro, no extra text).\n\n"
                f"Topic: {topic}\n\n"
                "Format example: [\"This is sentence one.\", \"This is sentence two.\"]"
            )
            try:
                response = model.generate_content(prompt)
                model_text = getattr(response, "text", "") or ""
                sentences = parse_sentence_list(model_text)
                if not sentences:
                    raise ValueError("Model did not return a valid sentence list.")
            except Exception as e:
                st.error(f"Failed to generate script: {e}")
                st.stop()

        with st.expander("Generated Script", expanded=True):
            for idx, line in enumerate(sentences, start=1):
                st.write(f"{idx}. {line}")

        # 2) Generate voiceover via ElevenLabs for the entire script
        with st.spinner("Generating voiceover..."):
            full_text = " ".join(sentences)
            try:
                if generate is None:
                    raise RuntimeError(
                        "ElevenLabs SDK interface not available. Ensure 'elevenlabs' is installed."
                    )
                # Use a default voice and model; adjust as desired
                voiceover_bytes = generate(
                    text=full_text,
                    voice="Rachel",
                    model="eleven_multilingual_v2",
                )
            except Exception as e:
                st.error(f"Failed to generate voiceover: {e}")
                st.stop()

        # Save audio to file for future phases and playback
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        audio_path = os.path.join(output_dir, "voiceover.mp3")
        try:
            with open(audio_path, "wb") as f:
                f.write(voiceover_bytes)
        except Exception as e:
            st.warning(f"Could not save audio file: {e}")

        st.success("Voiceover ready!")
        st.audio(voiceover_bytes, format="audio/mpeg")

        # 3) Assemble stock-footage video with subtitles using MoviePy + Pixabay
        with st.spinner("Assembling video..."):
            try:
                try:
                    from moviepy.editor import (
                        AudioFileClip,
                        VideoFileClip,
                        ColorClip,
                        CompositeVideoClip,
                        TextClip,
                        concatenate_videoclips,
                        vfx,
                    )
                except Exception as import_err:
                    st.error(
                        "MoviePy failed to import. Install dependencies and restart:\n"
                        "pip install moviepy imageio-ffmpeg pillow"
                    )
                    raise import_err

                audio_clip = AudioFileClip(audio_path)
                total_duration = float(audio_clip.duration or 0.0)
                if total_duration <= 0.0:
                    raise ValueError("Audio duration is zero.")

                size = (1280, 720)

                # Allocate sentence durations proportional to word counts
                word_counts = [max(1, len(s.split())) for s in sentences] or [1]
                total_words = sum(word_counts)
                proportional_durations = [total_duration * (wc / total_words) for wc in word_counts]
                min_per_sentence = 0.8
                durations = [max(min_per_sentence, d) for d in proportional_durations]
                overflow = (sum(durations) - total_duration)
                if overflow > 0 and len(durations) > 0:
                    durations[-1] = max(min_per_sentence, durations[-1] - overflow)

                clips = []
                tmp_files: List[str] = []
                for idx, (sentence, dur) in enumerate(zip(sentences, durations)):
                    clip = None
                    try:
                        if pixabay_api_key:
                            kws = extract_keywords(sentence)
                            query = "+".join(kws)
                            url = pixabay_video_url(pixabay_api_key, query)
                        else:
                            url = None

                        if url:
                            tmp_path = os.path.join(output_dir, f"px_{idx}.mp4")
                            if download_file(url, tmp_path):
                                tmp_files.append(tmp_path)
                                vclip = VideoFileClip(tmp_path)
                                # Fit to 720p height, then letterbox to 1280 width if needed
                                vclip = vclip.resize(height=size[1])
                                if vclip.w < size[0]:
                                    pad_left = (size[0] - vclip.w) // 2
                                    pad_right = size[0] - vclip.w - pad_left
                                    # Pad with black borders
                                    vclip = vclip.margin(left=pad_left, right=pad_right, color=(0, 0, 0))
                                # Ensure duration
                                if vclip.duration < dur:
                                    vclip = vclip.fx(vfx.loop, duration=dur)
                                clip = vclip.subclip(0, dur)

                        if clip is None:
                            # Fallback to black clip
                            clip = ColorClip(size=size, color=(0, 0, 0), duration=dur)
                    except Exception:
                        clip = ColorClip(size=size, color=(0, 0, 0), duration=dur)

                    clips.append(clip)

                video_track = concatenate_videoclips(clips, method="compose")
                video_track = video_track.set_audio(audio_clip)

                # Build timed subtitles over the concatenated track
                text_clips: List[TextClip] = []
                start_time = 0.0
                for sentence, dur in zip(sentences, durations):
                    try:
                        txt = TextClip(
                            sentence,
                            fontsize=42,
                            color="white",
                            font="DejaVu-Sans",
                            method="caption",
                            size=(int(size[0] * 0.9), None),
                        ).set_position(("center", "bottom")).set_duration(dur).set_start(start_time)
                        text_clips.append(txt)
                    except Exception:
                        pass
                    start_time += float(dur)

                composite = CompositeVideoClip([video_track] + text_clips, size=size)
                video_path = os.path.join(output_dir, "final_video.mp4")
                composite.write_videofile(
                    video_path,
                    fps=24,
                    codec="libx264",
                    audio_codec="aac",
                    verbose=False,
                    logger=None,
                )
                st.success("Video assembled!")
                st.video(video_path)
            except Exception as e:
                st.warning(f"Failed to assemble video: {e}")
            finally:
                # Cleanup resources
                try:
                    composite.close()
                except Exception:
                    pass
                try:
                    video_track.close()
                except Exception:
                    pass
                try:
                    for c in clips:
                        try:
                            c.close()
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    audio_clip.close()
                except Exception:
                    pass
                # Remove temp files
                try:
                    for p in tmp_files:
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                except Exception:
                    pass


if __name__ == "__main__":
    main()
