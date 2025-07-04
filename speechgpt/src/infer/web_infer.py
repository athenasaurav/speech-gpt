import numpy as np
import time
import time
import os
import numpy as np
import gradio as gr
from utils.speech2unit.speech2unit import Speech2Unit
from src.infer.cli_infer import SpeechGPTInference
import soundfile as sf
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model-name-or-path", type=str, default=f"temp_audio_{int(time.time())}.wav")
parser.add_argument("--lora-weights", type=str, default=None)
parser.add_argument("--s2u-dir", type=str, default="speechgpt/utils/speech2unit/")
parser.add_argument("--vocoder-dir", type=str, default="speechgpt/utils/vocoder/")
parser.add_argument("--output-dir", type=str, default="speechgpt/output/")
parser.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit mode")
parser.add_argument("--input-path", type=str, default=f"temp_audio_{int(time.time())}.wav", help="Input audio file path" )
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

infer = SpeechGPTInference(
    args.model_name_or_path,
    args.lora_weights,
    args.s2u_dir,
    args.vocoder_dir,
    args.output_dir
)

def speech_dialogue(audio):
    sr, data = audio
    sf.write(
        "temp_audio.wav",
        data,
        sr,
    )
    prompts = ["temp_audio.wav"]
    sr, wav = infer(prompts)
    return (sr, wav)


demo = gr.Interface(    
        fn=speech_dialogue, 
        inputs="microphone", 
        outputs="audio", 
        title="SpeechGPT",
        cache_examples=False
        )
demo.launch(share=True)

import numpy as np

# Add this function to convert audio data properly
def fix_audio_format(audio_data, sample_rate):
    # Convert to numpy array if not already
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data)
    
    # Convert to float32 and normalize to [-1, 1] range
    if audio_data.dtype == np.int64:
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int64).max
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int32).max
    elif audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
    
    return sample_rate, audio_data