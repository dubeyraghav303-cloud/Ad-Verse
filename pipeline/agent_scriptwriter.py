import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
import re

# --- 1. Load the Model (This runs only ONCE when the app starts) ---
print("[Agent 1] Initializing... This may take a moment.")
model_id = "Qwen/Qwen2-7B-Instruct"

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("Warning: No GPU detected. LLM will be very slow.")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto", # Use float16/bfloat16 if GPU is available
        device_map=device
    )
    
    # Create a transformers pipeline for easy text generation
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device
    )
    print("[Agent 1] Model loaded successfully.")

except Exception as e:
    print(f"[Agent 1] CRITICAL ERROR: Could not load model. {e}")
    print("This project requires a powerful GPU and correct CUDA setup.")
    llm_pipeline = None

# --- Helper function to clean the LLM's output ---
def extract_json_from_text(text):
    """Finds and parses the first valid JSON object in a string."""
    # Look for the first { and the last }
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        print("[Agent 1] Error: No JSON object found in LLM response.")
        return None
    
    try:
        return json.loads(json_match.group(0))
    except json.JSONDecodeError as e:
        print(f"[Agent 1] Error: Could not decode JSON. {e}")
        print(f"Raw response: {json_match.group(0)}")
        return None

# --- 2. The Main Agent Function ---
def generate_script(user_prompt):
    """
    Uses the loaded LLM to generate a structured JSON script for the ad.
    """
    if llm_pipeline is None:
        raise Exception("Agent 1 (LLM) is not initialized. Check for loading errors.")
        
    print(f"[Agent 1] Received prompt: {user_prompt}")
    
    prompt_template = f"""
    You are an expert advertising copywriter. A user needs a 30-second video ad.
    The user's prompt is: "{user_prompt}"
    
    Generate a complete ad script. The script must have exactly 5 scenes.
    The total voice-over script should be about 75 words (for 30 seconds).
    
    For each scene, provide:
    1. "voice_over": A short line of voice-over text.
    2. "visual_prompt": A detailed text-to-video prompt (e.g., "A cinematic, slow-motion close-up of...")
    
    Respond ONLY with a single, valid JSON object in this format:
    {{"scenes": [...]}}
    """
    
    # Format the prompt for the Qwen2 model
    messages = [
        {"role": "system", "content": "You are a helpful assistant that only outputs valid JSON."},
        {"role": "user", "content": prompt_template}
    ]
    
    # --- 3. Run the LLM ---
    print("[Agent 1] Generating script...")
    try:
        # Use 'apply_chat_template' to format the messages correctly
        prompt = llm_pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        outputs = llm_pipeline(
            prompt,
            max_new_tokens=1024,  # Max length of the *output*
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        
        raw_response = outputs[0]["generated_text"]
        # The response will include our prompt, so we need to find the part *after* it
        generated_text = raw_response.split(prompt)[-1]
        
        print("[Agent 1] Raw response received. Parsing JSON...")
        
        # --- 4. Parse the Output ---
        script_data = extract_json_from_text(generated_text)
        
        if script_data and "scenes" in script_data and len(script_data["scenes"]) == 5:
            print("[Agent 1] Script generated and parsed successfully.")
            return script_data
        else:
            print("[Agent 1] Error: Generated JSON is invalid or incomplete.")
            return None

    except Exception as e:
        print(f"[Agent 1] ERROR during generation: {e}")
        return None
