import streamlit as st
import assemblyai as aai
from transformers import pipeline
import tempfile
import openai
import io

st.set_page_config(layout="wide")

aai.settings.api_key = st.sidebar.text_input('Veuillez insérer la clée fournie pour transcription', type='password')
#aai.settings.api_key = "146c7980fa5a4b6c872033d97234500b"
def transcribe_audio(audio_path):
    # Configuration de l'API AssemblyAI
    #aai.settings.api_key = "146c7980fa5a4b6c872033d97234500b"

    # Création d'un transcriber
    transcriber = aai.Transcriber()
    # Configuration de la transcription
    config = aai.TranscriptionConfig(language_code="fr", speaker_labels=True, speakers_expected=2)
    # Transcription de l'audio
    transcript = transcriber.transcribe(audio_path, config)
    return transcript



# Set up OpenAI API key

openai.api_key = st.sidebar.text_input('Veuillez insérer la clée fournie pour démonstration', type='password')

def analyze_emotion(text):
   try:
       #content = f"peux tu me donner seulement une émotion exacte sans commentaire, et si tu ne détecte pas une émotion met 'neutre' sans commentaires : par exemple 'Neutre ou frustration ou colère, ...'?\n{text}"
       content = f"Please analyze the following text to detect the underlying emotion. Return the detected emotion in French (e.g., 'Neutre', 'Frustration', 'Colère', etc.). If no specific emotion is detected, please respond with 'Neutre'. The text for analysis is: \n{text}"
       messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": content}]
       response = openai.ChatCompletion.create(
           model="gpt-3.5-turbo-16k",
           messages=messages,
           max_tokens=100
       )
       emotion_result = response['choices'][0]['message']['content']
       # Extraire l'émotion de la phrase complète
       emotion = emotion_result.split(":")[-1].strip()
       return emotion
   except Exception as e:
       print(f"Erreur lors de l'analyse de l'émotion : {e}")
       return None

def analyze_voix(text):
   try:
       #content = f"peux tu me donner seulement une émotion exacte sans commentaire, et si tu ne détecte pas une émotion met 'neutre' sans commentaires : par exemple 'Neutre ou frustration ou colère, ...'?\n{text}"
       content = f"Please analyze the following text to detect the underlying emotion. Return the detected emotion as 'Positive', 'Negative', or 'Neutral'. If no specific emotion is detected, please respond with 'Neutral'. The text for analysis is: \n{text}"

       messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": content}]
       response = openai.ChatCompletion.create(
           model="gpt-3.5-turbo-16k",
           messages=messages,
           max_tokens=100
       )
       emotion_result = response['choices'][0]['message']['content']
       # Extraire l'émotion de la phrase complète
       emotion0 = emotion_result.split(":")[-1].strip()
       return emotion0
   except Exception as e:
       print(f"Erreur lors de l'analyse de l'émotion : {e}")
       return None
 

# Streamlit app
def main():
     
    # Titre de l'application
    st.markdown("<h1 style='text-align:center;font-size:xx-large;color: #B01817;'>Transcription audio et analyse émotionnelle</h1>", unsafe_allow_html=True)
    #st.title("Analyse de la transcription audio")
    st.sidebar.image("logo2.jpg", use_column_width=True)

  

    #option = st.sidebar.selectbox("Option ", ["Téléverser un fichier audio", "Utiliser le chemin du fichier audio"])

    #if option == "Téléverser un fichier audio":
        
        # Ajouter un composant pour uploader un fichier audio
    uploaded_file = st.file_uploader("Téléverser un fichier audio", type=["mp3", "wav"])
    #audio_data = io.BytesIO(uploaded_file.read())
    #st.audio(audio_data, format='audio/wav')
    # Lire le fichier audio téléversé et le jouer
    

        # Créer une rangée pour les boutons "Transcription" et "Emotion"
    button_col1, button_col2, button_col3 = st.columns(3)

        # Vérifier si un fichier a été uploadé
    if uploaded_file is not None:
        
            # Créer un fichier temporaire pour enregistrer l'audio uploadé
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_path = tmp_file.name
            st.audio(open(audio_path, "rb").read(), format="audio/wav")

            # Boutons pour la transcription et l'analyse de l'émotion
        if button_col1.button("Transcription") :

                # Transcription de l'audio
            transcript = transcribe_audio(audio_path)

            for utterance in transcript.utterances:
                        st.write(f"<span style='color: #922B21;'>Speaker {utterance.speaker}:</span> {utterance.text}", unsafe_allow_html=True)
                            
            # Bouton "Emotion"
        if button_col2.button("Émotion basée sur le texte") and uploaded_file:

                # Transcription de l'audio
            transcript = transcribe_audio(audio_path)

            for utterance in transcript.utterances:
                st.write(f"<span style='color: blue;'>Speaker {utterance.speaker}:</span> {utterance.text}", unsafe_allow_html=True)
                    
                sentiment = analyze_emotion(utterance.text)
                    # Affichage du résultat de l'analyse de sentiment
                st.write("Emotion détectée : ", sentiment)

        if button_col3.button("Émotion basée sur la voix") and uploaded_file:
                # Transcription de l'audio
            transcript = transcribe_audio(audio_path)

            for utterance in transcript.utterances:
                st.write(f"<span style='color: blue;'>Speaker {utterance.speaker}:</span> {utterance.text}", unsafe_allow_html=True)
            
                    # Transcription de l'audio
                sentiment0 = analyze_voix(utterance.text)
                st.write("Sentiment : ", sentiment0)
                 

    else:
            # Message indiquant à l'utilisateur d'uploader un fichier
        st.write("Veuillez uploader un fichier audio pour commencer la transcription.")     
            
    

if __name__ == "__main__":
    main()
