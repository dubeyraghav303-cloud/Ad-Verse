import streamlit as st
import os
import time
from pipeline.main_pipeline import run_ad_verse_pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="Ad-Verse AI",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Ad-Verse: The AI Commercial Director")
st.markdown("Turn a single prompt into a fully-produced 30-second video ad. This app is a **blueprint**; it runs a *mock* pipeline. See `README.md` to implement the real AI models.")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("1. Describe Your Product")
    product_prompt = st.text_area(
        "What are you advertising?",
        "A cozy coffee shop called 'The Daily Grind', located in a quiet neighborhood.",
        height=100
    )

    st.header("2. Choose a Vibe")
    ad_vibe = st.selectbox(
        "What's the mood?",
        ("Cozy & Relaxing", "Energetic & Exciting", "Modern & Sleek", "Funny & Quirky")
    )
    
    st.header("3. Select a Voice")
    ad_voice = st.selectbox(
        "Choose a voice style:",
        ("Warm Male", "Professional Female", "Friendly Male", "Calm Female")
    )

    st.divider()
    generate_button = st.button("✨ Generate My Ad!", type="primary", use_container_width=True)

# --- Main Content Area for Output ---
output_container = st.container()
output_container.header("Your Generated Ad")
video_placeholder = output_container.empty()
video_placeholder.info("Your ad will appear here once generated.")

# This callback function lets us update the Streamlit UI from the pipeline
def streamlit_callback(message):
    st.toast(message)
    print(message) # Also print to console

if generate_button:
    if not product_prompt:
        st.error("Please describe your product first!")
    else:
        full_prompt = f"{product_prompt}. Vibe: {ad_vibe}. Voice: {ad_voice}."
        
        # This is where we call the master pipeline
        try:
            with st.spinner("The AI agents are at work..."):
                final_ad_path = run_ad_verse_pipeline(full_prompt, streamlit_callback)
            
            if final_ad_path and os.path.exists(final_ad_path):
                st.success("🎉 Your ad is ready!")
                video_placeholder.video(final_ad_path)
                
                with open(final_ad_path, "rb") as file:
                    st.download_button(
                        label="Download Ad (.mp4)",
                        data=file,
                        file_name="my_adverse_ad.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
            else:
                st.error("The ad generation failed. See console for mock data. (This is expected in the blueprint).")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("This is expected in the blueprint. Implement the AI models in the `/pipeline/` files to fix.")
