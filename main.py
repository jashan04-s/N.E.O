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
from ai_assistant import AIAssistant

load_dotenv()
PICOVOICE_ACCESS_KEY = os.environ.get("PICOVOICE_ACCESS_KEY")

# ========== Audio Setup ==========
model_path = "models\\Hey-Neo_en_windows_v3_0_0.ppn"
porcupine = pvporcupine.create(
    keyword_paths=[model_path], access_key=PICOVOICE_ACCESS_KEY
)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 400
SILENCE_DURATION = 1
INPUT_DEVICE_INDEX = 9

stop_event = Event()


def is_silent(data, threshold=SILENCE_THRESHOLD):
    audio_data = np.frombuffer(data, dtype=np.int16)
    return np.max(np.abs(audio_data)) < threshold


import threading
import queue
import time
import pyaudio
import numpy as np

def record_audio():
    """Stream mic input into DeepSpeech in real-time after wake word, and return final transcription."""
    start_time = time.time()
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print("Recording and transcribing...")
    logTime("PyAudio setup", start_time=start_time)

    audioQueue = queue.Queue()
    result_queue = queue.Queue()  # To receive final transcription
    silent_chunks = 0

    # Thread to stream text from the audio queue
    def transcription_thread():
        final_text = None
        for text, is_final in getRealTimeTextFromAudio(audioQueue, DS_MODEL):
            print(f"\r{text}", end="", flush=True)
            if is_final:
                final_text = text
                print("\nFinal Output:", final_text)
        result_queue.put(final_text)  # Send final output to main thread

    t = threading.Thread(target=transcription_thread, name="TranscriptionThread")
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
            print("\nPause detected, stopping recording.")
            break

    # Stop everything
    stream.stop_stream()
    stream.close()
    audio.terminate()
    audioQueue.put(None)  # Signal end of audio
    t.join()

    # Get final output from transcription thread
    final_output = result_queue.get()
    print("Transcription complete.")
    return final_output

def listen_for_keyword():
    """Listen for keyword and trigger recording/transcription"""
    print("Listening for wake word 'NEO'...")
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=porcupine.frame_length,
    )

    while not stop_event.is_set():
        pcm = np.frombuffer(
            stream.read(porcupine.frame_length, exception_on_overflow=False),
            dtype=np.int16,
        )
        keyword_index = porcupine.process(pcm)

        if keyword_index >= 0:
            print("Wake word 'NEO' detected!")
            message = record_audio()
            print("Final Sentence", message)
            askNEO(message)
            print("Resuming wake word detection...")


def askNEO(message):
    neo = AIAssistant("Neo", "gpt-4o-mini", "Your assistant for scheduling, web searches, and general assistance with British sass.")
    response = neo.talk_to_assistant(message)
    if response:
        print("NEO says:", response)
    else:
        print("NEO did not respond.")

def main():
    print("CUDA available:", torch.cuda.is_available())
    thread = threading.Thread(target=listen_for_keyword, name="KeywordListener")
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
