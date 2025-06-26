from gtts import gTTS

def synthesize_speech(text, output_file, language_code="en", tld="co.uk"):
    """
    Synthesizes speech from the input text using gTTS and saves it to an output file.

    Args:
        text (str): The text to be converted to speech.
        output_file (str): The path to the output MP3 file.
        language_code (str): The language code for the voice (default is "en").
        tld (str): Top-level domain to specify regional accent (e.g., "co.uk" for British English).
    """
    tts = gTTS(text=text, lang=language_code, tld=tld)
    tts.save(output_file)
