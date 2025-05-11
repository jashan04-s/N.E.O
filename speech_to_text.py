import whisper
import os
import torch
import time
from test_utils import logTime

# Contributes to initial load time, place elsewhere
MODEL = whisper.load_model("small")

def getTextFromWavFile(audioFilePath):
    if not os.path.isfile(audioFilePath):
        print(f"Error: The file {audioFilePath} does not exist.")
        return

    device = "cpu"    
    start_time = time.time()    
    model = whisper.load_model("small", device=device)
    result = model.transcribe(audioFilePath)
    logTime("Without Cuda", start_time=start_time)
    
    print(result["text"])
    return result
   
