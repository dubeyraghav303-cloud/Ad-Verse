import time
import os
from .agent_scriptwriter import generate_script
from .agent_voice_actor import generate_voice
from .agent_director import generate_scenes
from .agent_editor import assemble_video

# Create output directory if it doesn't exist
OUTPUT_DIR = "outputs/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_ad_verse_pipeline(full_prompt, callback):
    """
    Runs the full generative pipeline from start to finish.
    'callback' is a function (like st.toast) to send status updates.
    """
    try:
        # --- Agent 1: Scriptwriter ---
        callback("Agent 1 (Scriptwriter) is writing the script...")
        script_data = generate_script(full_prompt)
        if not script_data:
            raise Exception("Script generation failed.")
        
        full_script = "\n".join([scene['voice_over'] for scene in script_data['scenes']])

        # --- Agent 2: Voice Actor ---
        callback("Agent 2 (Voice Actor) is recording the voice-over...")
        voice_path = generate_voice(full_script)
        if not voice_path or not os.path.exists(voice_path):
            raise Exception("Voice generation failed. (Mock file missing?)")
            
        # --- Agent 3: Director ---
        scene_paths = []
        scene_count = len(script_data['scenes'])
        for i, scene in enumerate(script_data['scenes']):
            callback(f"Agent 3 (Director) is generating scene {i+1}/{scene_count}...")
            scene_path = generate_scenes(scene['visual_prompt'])
            if not scene_path or not os.path.exists(scene_path):
                raise Exception(f"Scene {i+1} generation failed. (Mock file missing?)")
            scene_paths.append(scene_path)

        # --- Agent 4: Editor ---
        callback("Agent 4 (Editor) is assembling the final cut...")
        final_video_path = assemble_video(scene_paths, voice_path)
        if not final_video_path:
            raise Exception("Video assembly failed.")
        
        callback("Pipeline complete!")
        return final_video_path

    except Exception as e:
        print(f"PIPELINE ERROR: {e}")
        return None