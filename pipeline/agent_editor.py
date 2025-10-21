from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import os

def assemble_video(scene_paths, voice_path):
    """
    Stitches all the generated .mp4 clips and the .wav file together.
    """
    print(f"[Agent 4] Assembling video from {len(scene_paths)} clips and 1 audio file.")
    final_output_path = "outputs/final_ad.mp4"
    
    try:
        # Load all video clips
        clips = [VideoFileClip(path) for path in scene_paths]
        
        # Concatenate them into one long clip
        final_clip = concatenate_videoclips(clips)
        
        # Load voice-over
        voice_clip = AudioFileClip(voice_path)
        
        # Check durations
        if final_clip.duration > voice_clip.duration:
            print("Warning: Video is longer than audio. Trimming video.")
            final_clip = final_clip.subclip(0, voice_clip.duration)
        elif voice_clip.duration > final_clip.duration:
            print("Warning: Audio is longer than video. Trimming audio.")
            voice_clip = voice_clip.subclip(0, final_clip.duration)
            
        # Set the audio of the final video
        final_clip = final_clip.set_audio(voice_clip)
        
        # TODO: Add background music
        # 1. Call a new 'agent_music_gen' to get a .mp3
        # 2. Load it as `music_clip = AudioFileClip(...)`
        # 3. Lower its volume: `music_clip = music_clip.volumex(0.1)`
        # 4. Composite it: `final_audio = CompositeAudioClip([voice_clip, music_clip])`
        # 5. Set it: `final_clip = final_clip.set_audio(final_audio)`
        
        # Write the final file
        print(f"Writing final ad to {final_output_path}")
        final_clip.write_videofile(
            final_output_path,
            codec="libx264",       # A high-compatibility codec
            audio_codec="aac",     # A high-compatibility audio codec
            temp_audiofile="outputs/temp-audio.m4a",
            remove_temp=True
        )
        
        # Clean up individual clips
        for clip in clips:
            clip.close()
        voice_clip.close()
        
        return final_output_path
        
    except Exception as e:
        print(f"[Agent 4] ERROR in video assembly: {e}")
        return None