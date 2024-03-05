import streamlit as st
import assemblyai as aai
from transformers import pipeline
import tempfile
import openai
import io

# Set page configuration
st.set_page_config(layout="wide")

# Sidebar input for AssemblyAI API key
aai.settings.api_key = st.sidebar.text_input('Veuillez insérer la clée fournie pour transcription', type='password')

# Function to transcribe audio using AssemblyAI
def transcribe_audio(audio_path):
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(language_code="fr", speaker_labels=True, speakers_expected=2)
    transcript = transcriber.transcribe(audio_path, config)
    return transcript

# Sidebar input for OpenAI API key
openai.api_key = st.sidebar.text_input('Veuillez insérer la clée fournie pour démonstration', type='password')

# Function to analyze emotion using OpenAI
def analyze_emotion(text):
    try:
        content = f"Please analyze the following text to detect the underlying emotion. Return the detected emotion in French (e.g., 'Neutre', 'Frustration', 'Colère', etc.). If no specific emotion is detected, please respond with 'Neutre'. The text for analysis is: \n{text}"
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": content}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            max_tokens=100
        )
        emotion_result = response['choices'][0]['message']['content']
        emotion = emotion_result.split(":")[-1].strip()
        return emotion
    except Exception as e:
        print(f"Erreur lors de l'analyse de l'émotion : {e}")
        return None

# Function to analyze voice emotion using OpenAI
def analyze_voix(text):
    try:
        content = f"Please analyze the following text to detect the underlying emotion. Return the detected emotion as 'Positive', 'Negative', or 'Neutral'. If no specific emotion is detected, please respond with 'Neutral'. The text for analysis is: \n{text}"
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": content}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            max_tokens=100
        )
        emotion_result = response['choices'][0]['message']['content']
        emotion0 = emotion_result.split(":")[-1].strip()
        return emotion0
    except Exception as e:
        print(f"Erreur lors de l'analyse de l'émotion : {e}")
        return None

# Streamlit app
def main():
    st.markdown("<h1 style='text-align:center;font-size:xx-large;color: #B01817;'>Transcription audio et analyse émotionnelle</h1>", unsafe_allow_html=True)
    st.sidebar.image("logo2.jpg", use_column_width=True)

    uploaded_file = st.file_uploader("Téléverser un fichier audio", type=["mp3", "wav"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_path = tmp_file.name

        if st.button("Transcription"):
            transcript = transcribe_audio(audio_path)
            for utterance in transcript.utterances:
                st.write(f"<span style='color: #922B21;'>Speaker {utterance.speaker}:</span> {utterance.text}", unsafe_allow_html=True)

        if st.button("Émotion basée sur le texte") and uploaded_file:
            transcript = transcribe_audio(audio_path)
            for utterance in transcript.utterances:
                st.write(f"<span style='color: blue;'>Speaker {utterance.speaker}:</span> {utterance.text}", unsafe_allow_html=True)
                sentiment = analyze_emotion(utterance.text)
                st.write("Emotion détectée : ", sentiment)

        if st.button("Émotion basée sur la voix") and uploaded_file:
            transcript = transcribe_audio(audio_path)
            for utterance in transcript.utterances:
                st.write(f"<span style='color: blue;'>Speaker {utterance.speaker}:</span> {utterance.text}", unsafe_allow_html=True)
                sentiment0 = analyze_voix(utterance.text)
                st.write("Sentiment : ", sentiment0)

    else:
        st.write("Veuillez uploader un fichier audio pour commencer la transcription.")     

if __name__ == "__main__":
    main()
