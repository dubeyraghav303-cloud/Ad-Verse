# app.py
import os
import streamlit as st

# Try to import real pipeline first
try:
    from pipeline.main_pipeline import run_pipeline as real_pipeline
except Exception:
    real_pipeline = None

# Always import mock fallback
from pipeline.inference import generate_mock_ad


st.set_page_config(page_title="Ad-Verse — AI Commercial Director", layout="centered")

st.title("🎬 Ad-Verse — The AI Commercial Director")
st.caption("Transform a single line prompt into a 30-second video ad. (Currently running mock mode)")

# Sidebar controls
st.sidebar.header("🧠 Generation Settings")
prompt = st.sidebar.text_area("Ad Prompt", value="A small cafe introducing artisan coffee with cozy vibes.")
style = st.sidebar.selectbox("Style", ["Minimal", "Cinematic", "Playful", "Corporate"])
voice = st.sidebar.selectbox("Voice", ["Neutral", "Energetic", "Calm"])
resolution = st.sidebar.selectbox("Resolution", ["640x360", "1280x720"], index=1)
fps = st.sidebar.slider("FPS", 12, 30, 24)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)

os.makedirs("outputs", exist_ok=True)

st.markdown("### 🪄 Your Prompt")
st.write(f"_{prompt or '— Enter something cool above —'}_")

if st.button("🎥 Generate 30-second Ad"):
    if not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        with st.spinner("🎬 Running pipeline... please wait..."):
            try:
                if real_pipeline:
                    out_path = real_pipeline(prompt=prompt, out_dir="outputs")
                else:
                    out_path = generate_mock_ad(prompt, "outputs", style, voice, resolution, fps, seed)
            except Exception as e:
                st.warning(f"⚠️ Real pipeline failed, using mock. ({e})")
                out_path = generate_mock_ad(prompt, "outputs", style, voice, resolution, fps, seed)

        st.success("✅ Ad Generated Successfully!")
        st.video(out_path)
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download Video", f, file_name=os.path.basename(out_path), mime="video/mp4")

st.markdown("---")
st.subheader("📂 Recent Outputs")
videos = sorted([f for f in os.listdir("outputs") if f.endswith(".mp4")], reverse=True)
if videos:
    for f in videos[:5]:
        st.write(f"**{f}**")
        st.video(os.path.join("outputs", f))
else:
    st.info("No generated ads yet.")
