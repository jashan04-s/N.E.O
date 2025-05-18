
import pyaudio
import pvporcupine
import wave
import numpy as np
import time
from speech_to_text import getTextFromWavFile
from test_utils import logTime
import threading
import torch
import os
from dotenv import load_dotenv



load_dotenv()

PICOVOICE_ACCESS_KEY = os.environ.get('PICOVOICE_ACCESS_KEY')

model_path = "models\\Hey-Neo_en_windows_v3_0_0.ppn"


porcupine = pvporcupine.create(keyword_paths= [model_path], access_key=PICOVOICE_ACCESS_KEY)
FORMAT = pyaudio.paInt16
CHANNELS = 1  
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 200
SILENCE_DURATION = 1
OUTPUT_FILE = "recorded_audio.wav"
INPUT_DEVICE_INDEX = 9

def is_silent(data, threshold=SILENCE_THRESHOLD):
    """Check if the audio chunk is silent."""
    audio_data = np.frombuffer(data, dtype=np.int16) 
    return np.max(np.abs(audio_data)) < threshold

def record_audio():
    start_time = time.time()
    audio = pyaudio.PyAudio()   

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    
    # for i in range(audio.get_device_count()):
    #     print(audio.get_device_info_by_index(i))

    logTime("PyAudio setup", start_time=start_time)
    print("Recording...")

    frames = []
    silent_chunks = 0
    while True:
            data = stream.read(CHUNK)
            frames.append(data)
            
            if is_silent(data):
                silent_chunks += 1
            else:
                silent_chunks = 0
            
            if silent_chunks >= int(RATE / CHUNK * SILENCE_DURATION):
                print("Pause detected, stopping recording.")
                break

    print("Recording complete.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(OUTPUT_FILE, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    getTextFromWavFile(OUTPUT_FILE)
    
    print(f"Audio recorded and saved as {OUTPUT_FILE}")

def listen_for_keyword():
    """Thread for listening to keyword (e.g., "NEO")"""
    print("Listening for 'NEO'...")
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, 
                        frames_per_buffer=CHUNK)
    
    while True:
        # Read audio data and process it for keyword detection
        pcm = np.frombuffer(stream.read(porcupine.frame_length), dtype=np.int16)
        
        # Check for keyword detection
        keyword_index = porcupine.process(pcm)
        
        if keyword_index >= 0:
            print("Keyword 'NEO' detected! Starting recording...")
            recording_thread = threading.Thread(target=record_audio)
            recording_thread.start()
            recording_thread.join()  # Wait for the recording to finish
            print("Resuming keyword detection...")

def main():
    print("Cuda availability", torch.cuda.is_available())
    recording_thread = threading.Thread(target=listen_for_keyword)
    recording_thread.start()
    recording_thread.join()

if __name__ =="__main__":
    main()