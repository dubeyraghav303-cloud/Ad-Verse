# Ad-Verse: The AI Commercial Director

This project is a multi-agent AI pipeline that generates 30-second video ads from a single text prompt.

## Project Status: Blueprint

This repository contains the complete "blueprint" and "scaffold" for the application. The Streamlit frontend and the pipeline logic are fully defined. The actual AI inference functions are placeholders (`TODO:`) and must be implemented by loading the real open-source models.

## **CRITICAL:** Setup & Installation

This project is extremely resource-intensive.

1.  **Hardware:** A modern NVIDIA GPU with at least **16GB of VRAM** is strongly recommended.
2.  **Environment:**
    * Install Python 3.10+.
    * Install `CUDA` (matching your PyTorch version).
    * Install `FFmpeg` (required by `moviepy`).
    * Create a virtual environment: `python -m venv venv`
    * Activate it: `source venv/bin/activate` (or `.\venv\Scripts\activate` on Windows)
    * Install all libraries: `pip install -r requirements.txt`
3.  **Download Models (The Hard Part):**
    * You will need to download gigabytes of model checkpoints.
    * **LLM:** Go to Hugging Face and download a model like `Qwen3-14B`.
    * **TTS:** Research and download the checkpoints for `Fish Speech` or `Coqui XTTSv2`.
    * **T2V:** Research and download the checkpoints for `Wan-AI/Wan2.2-T2V-A14B` (or the latest state-of-the-art model).
    * Save all model files into the `/models/` directory.

## How to Run the (Blueprint) App

Even without the models, you can run the app to see the frontend. It will print "MOCK" actions to the console.

1.  Activate your virtual environment.
2.  Run Streamlit: `streamlit run app.py`
