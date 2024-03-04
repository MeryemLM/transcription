import streamlit as st
import assemblyai as aai
from transformers import pipeline
import tempfile
import openai

st.set_page_config(layout="wide")


def transcribe_audio(audio_path):
    # Configuration de l'API AssemblyAI
    aai.settings.api_key = "146c7980fa5a4b6c872033d97234500b"

    # Création d'un transcriber
    transcriber = aai.Transcriber()
    # Configuration de la transcription
    config = aai.TranscriptionConfig(language_code="fr", speaker_labels=True, speakers_expected=2)
    # Transcription de l'audio
    transcript = transcriber.transcribe(audio_path, config)
    return transcript


if __name__ == "__main__":
    main()
