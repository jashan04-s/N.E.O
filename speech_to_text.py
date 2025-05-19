import deepspeech
import os
import time
import numpy as np
import queue
from test_utils import logTime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILE = os.path.join(MODEL_DIR, "deepspeech-0.9.3-models.pbmm")
SCORER_FILE = os.path.join(MODEL_DIR, "deepspeech-0.9.3-models.scorer")

if not os.path.isfile(MODEL_FILE):
    raise FileNotFoundError(f"Model file not found at {MODEL_FILE}")
if not os.path.isfile(SCORER_FILE):
    raise FileNotFoundError(f"Scorer file not found at {SCORER_FILE}")

print("Loading DeepSpeech model...")


DS_MODEL = deepspeech.Model(MODEL_FILE)
DS_MODEL.enableExternalScorer(SCORER_FILE)

print("DeepSpeech model loaded.")


def getRealTimeTextFromAudio(audioQueue, model = DS_MODEL, sample_rate=16000):
    """
    Streams audio from a queue and yields intermediate transcriptions using DeepSpeech.

    Args:
        audioQueue (queue.Queue): Queue containing raw audio data (int16 format).
        model (deepspeech.Model): An initialized DeepSpeech model.
        sample_rate (int): Expected sample rate (default is 16000 for DeepSpeech).

    Yields:
        str: Real-time transcribed text.
    """
    stream = model.createStream()

    while True:
        data = audioQueue.get()
        if data is None:
            break  # Exit on signal

        audio = np.frombuffer(data, dtype=np.int16)
        stream.feedAudioContent(audio)

        # Yield intermediate result
        yield stream.intermediateDecode()

    # After final chunk
    yield stream.finishStream()


# ======= Whisper Alternative (Commented) =======
# import whisper
# def getTextFromWavFile(audioFilePath):
#     if not os.path.isfile(audioFilePath):
#         print(f"Error: The file {audioFilePath} does not exist.")
#         return

#     device = "cpu"
#     start_time = time.time()
#     model = whisper.load_model("small", device=device)
#     result = model.transcribe(audioFilePath)
#     logTime("Without Cuda", start_time=start_time)

#     print(result["text"])
#     return result
