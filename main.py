import pyaudio
import pvporcupine
import wave
import numpy as np
import time
import threading
import torch
import os
import queue
from dotenv import load_dotenv
from test_utils import logTime
from speech_to_text import getRealTimeTextFromAudio, DS_MODEL 
from threading import Event


load_dotenv()
PICOVOICE_ACCESS_KEY = os.environ.get('PICOVOICE_ACCESS_KEY')

# ========== Audio Setup ==========
model_path = "models\\Hey-Neo_en_windows_v3_0_0.ppn"
porcupine = pvporcupine.create(keyword_paths=[model_path], access_key=PICOVOICE_ACCESS_KEY)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 200
SILENCE_DURATION = 1
INPUT_DEVICE_INDEX = 9

stop_event = Event()

def is_silent(data, threshold=SILENCE_THRESHOLD):
    audio_data = np.frombuffer(data, dtype=np.int16)
    return np.max(np.abs(audio_data)) < threshold

def record_audio():
    """Stream mic input into DeepSpeech in real-time after wake word"""
    start_time = time.time()
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print("Recording and transcribing...")
    logTime("PyAudio setup", start_time=start_time)

    audioQueue = queue.Queue()
    silent_chunks = 0

    # Thread to stream text from the audio queue
    def transcription_thread():
        for text in getRealTimeTextFromAudio(audioQueue, DS_MODEL):
            print(f"\r {text}", end='', flush=True)

    t = threading.Thread(target=transcription_thread)
    t.start()

    # Stream audio to queue
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audioQueue.put(data)

        if is_silent(data):
            silent_chunks += 1
        else:
            silent_chunks = 0

        if silent_chunks >= int(RATE / CHUNK * SILENCE_DURATION):
            print("\n Pause detected, stopping recording.")
            break

    # Stop everything
    stream.stop_stream()
    stream.close()
    audio.terminate()
    audioQueue.put(None)  # Signal end of audio
    t.join()
    print("Transcription complete.")

def listen_for_keyword():
    """Listen for keyword and trigger recording/transcription"""
    print("Listening for wake word 'NEO'...")
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=porcupine.frame_length)

    while not stop_event.is_set():
        pcm = np.frombuffer(stream.read(porcupine.frame_length, exception_on_overflow=False), dtype=np.int16)
        keyword_index = porcupine.process(pcm)

        if keyword_index >= 0:
            print("Wake word 'NEO' detected!")
            record_audio()
            print("Resuming wake word detection...")

def main():
    print("CUDA available:", torch.cuda.is_available())
    thread = threading.Thread(target=listen_for_keyword)
    thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n Stopping...")
        stop_event.set()
        thread.join()

if __name__ == "__main__":
    main()
