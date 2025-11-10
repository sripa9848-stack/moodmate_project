import streamlit as st
import pandas as pd
import numpy as np
import cv2 
from typing import Union

# DeepFace is the core library for emotion analysis
try:
    from deepface import DeepFace
except ImportError:
    st.error("DeepFace library not found. Please run: pip install deepface opencv-python tensorflow")
    DeepFace = None 
    
# CONFIGURATION & FILE PATHS
MUSIC_DATA_PATH = 'spotify_tracks.csv' 
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'] 

# CUSTOM TAMIL PLAYLIST WITH DIRECT SPOTIFY LINKS 
CUSTOM_PLAYLIST = {
    'happy': [
        {'name': 'Jalabulajangu', 'artist': 'D. Imman', 'mood': 'Joyful & Energetic', 'spotify_url': 'https://open.spotify.com/album/3cXSg3K9z6fZAd1lWmqvrc?si=-e_Jq1xVSsavhaYo5rylOQ%0A_source=copy-link'},
        {'name': 'En Jannal Vantha', 'artist': 'A.R. Rahman', 'mood': 'Joyful & Energetic', 'spotify_url': 'https://open.spotify.com/track/04fYoQSUTfPhVjyW8aXkix?si=FzULlkSETx6_7hBdcAPGxQ%0A_source=copy-link'},
        {'name': 'Mazhai Vara Poguthae', 'artist': 'Yuvan Shankar Raja', 'mood': 'Joyful & Energetic', 'spotify_url': 'https://open.spotify.com/track/5ich0wSxDxMryEjwZo6XzG?si=odc9z8hJT3-l-vFuKZOwmw%0A_source=copy-link'},
        {'name': 'Jigidi Killadi', 'artist': 'D. Imman', 'mood': 'Joyful & Energetic', 'spotify_url': 'https://open.spotify.com/track/0OgazYID6CUlSeiAoeuf5B?si=UVesFrwpQ1yVSCpLmAkTGA%0A_source=copy-link'},
        {'name': 'Bujji', 'artist': 'D. Imman', 'mood': 'Joyful & Energetic', 'spotify_url': 'https://open.spotify.com/track/6rJhjykEkCqm3H64uALhhf?si=XGWwPOB8SPutsQ1ypDt49g%0A_source=copy-link'},
    ],
    'sad': [
        {'name': 'Kanavae Kanavae', 'artist': 'S. Thaman', 'mood': 'Emotional & Reflective', 'spotify_url': 'https://open.spotify.com/track/6sPW145Y3LSXd4p1LjBOcS?si=LMrZBgevRUCLWwoAFsVGDg%0A_source=copy-link'},
        {'name': 'Po Nee Po', 'artist': 'Anirudh Ravichander', 'mood': 'Emotional & Reflective', 'spotify_url': 'https://open.spotify.com/track/6zdikHQs2PBFgGIZBxqOeV?si=wD81JrRrSsK688cdLCG2CQ%0A_source=copy-link'},
        {'name': 'Imaye Imayae', 'artist': 'G.V. Prakash Kumar', 'mood': 'Emotional & Reflective', 'spotify_url': 'https://open.spotify.com/track/6H1hnJRQOGQ9djPL3jNu9G?si=Ug-8TzDUS0m3hIl5u1lBQQ%0A_source=copy-link'},
        {'name': 'Pirai Thedum', 'artist': 'G.V. Prakash Kumar', 'mood': 'Emotional & Reflective', 'spotify_url': 'https://open.spotify.com/track/4Dd5XLOdAAmURIZSLThPvH?si=1kw4t7b7QqmG-UL_UvJnKg%0A_source=copy-link'},
        {'name': 'Marappadhillai Nenjae', 'artist': 'D. Imman', 'mood': 'Emotional & Reflective', 'spotify_url': 'https://open.spotify.com/track/4Ndcwn2iAt1MdU6lpw24ZQ?si=FoywoipjSi6eTGuTeE0yMg%0A_source=copy-link'},
    ],
    'angry': [
        {'name': 'Veera Soora', 'artist': 'Hiphop Tamizha', 'mood': 'High Energy & Action', 'spotify_url': 'https://open.spotify.com/track/0lNr0bkRupchqckGeSMSpZ?si=hVvcbxVkQLGaGeD2CJzLWw%0A_source=copy-link'},
        {'name': 'Hey Mama', 'artist': 'Anirudh Ravichander', 'mood': 'High Energy & Action', 'spotify_url': 'https://open.spotify.com/track/32E2AGkk15IU9JMZUGs7Ih?si=303AfnOERASED3mBw4VuIQ%0A_source=copy-link'},
        {'name': 'Theemai Than Vellum', 'artist': 'Harris Jayaraj', 'mood': 'High Energy & Action', 'spotify_url': 'https://open.spotify.com/track/3F7IiHst8X2KXwu4oBkywY?si=Biqn-EVjSiyBPWp8njzKfA%0A_source=copy-link'},
        {'name': 'Thee Thalapathy', 'artist': 'S. Thaman', 'mood': 'High Energy & Action', 'spotify_url': 'https://open.spotify.com/track/0qgy62WThodLxylEu0Dbnc?si=4nc7naODS_-bJVcnTBa3Mg%0A_source=copy-link'},
        {'name': 'Mavanae', 'artist': 'Anirudh Ravichander', 'mood': 'High Energy & Action', 'spotify_url': 'https://open.spotify.com/track/6ffEHNJBFnr68xNVoRwPf2?si=gCZerbPaSuutt05T6TiVag%0A_source=copy-link'},
    ],
    'fear': [ 
        {'name': 'Ratsasan Theme', 'artist': 'Ghibran', 'mood': 'Intense & Calming Score', 'spotify_url': 'https://open.spotify.com/track/43lstkuASEtAyoDEVoynup?si=zKRc8SOZSTCJo8pm0ckzrg%0A_source=copy-link'},
        {'name': 'Keerthana Accident', 'artist': 'Ghibran', 'mood': 'Intense & Calming Score', 'spotify_url': 'https://open.spotify.com/track/4gHtsX6jjUzYhOrLCkCFTN?si=cxBc4yMeTn2ZkYsyZyiAAw%0A_source=copy-link'},
        {'name': 'Daughter Feels Background Score', 'artist': 'A.R. Rahman', 'mood': 'Intense & Calming Score', 'spotify_url': 'https://open.spotify.com/track/1EBFhYnJhpiWx3lxQ2u59q?si=VG90vX1FS4mtYG5aQXSjWg%0A_source=copy-link'},
        {'name': 'Krishna & Nila Unconditional Love', 'artist': 'Ghibran', 'mood': 'Intense & Calming Score', 'spotify_url': 'https://open.spotify.com/track/78uWxjIuuAyo9ROsoEfBfn?si=ZWf5X_lHTTacO-URamJXew%0A_source=copy-link'},
    ],
    'surprise': [
        {'name': 'Adhaaru Adhaaru', 'artist': 'Anirudh Ravichander', 'mood': 'Excitement & Fun', 'spotify_url': 'https://open.spotify.com/track/0Ps7lWZdj1NAuyAVumclyC?si=XLBxdmfWTB6NhuGQ0oUV-g%0A_source=copy-link'},
        {'name': 'Aaluma Doluma', 'artist': 'A.R. Rahman', 'mood': 'Excitement & Fun', 'spotify_url': 'https://open.spotify.com/track/1Do2hDE0etMakAEQbyOd4L?si=OINJln3oS3GbjOkPJwQrUw%0A_source=copy-link'},
        {'name': 'Singari', 'artist': 'Sai Abhyanakkar', 'mood': 'Excitement & Fun', 'spotify_url': 'https://open.spotify.com/track/2iG9pZ6bkfVqXzjuax7J8Z?si=8DYWbbR5T1qFKm5CASmrdQ%0A_source=copy-link'},
    ],
    'disgust': [ 
        {'name': 'Die with A Smile', 'artist': 'Lady Gaga', 'mood': 'Pensive & Neutralizing', 'spotify_url': 'https://open.spotify.com/track/2plbrEY59IikOBgBGLjaoe?si=m_pateXQTBGqaI4UBGso8A%0A_source=copy-link'},
        {'name': 'Aasai Oru Pulveli', 'artist': 'Yuvan Shankar Raja', 'mood': 'Pensive & Neutralizing', 'spotify_url': 'https://open.spotify.com/track/3eLF7e1ODaSX82yKUOuV3i?si=NSctfxBXTZ2sXxLzf8Mirg%0A_source=copy-link'},
        {'name': 'Unna Nenachu', 'artist': 'Illayaraja', 'mood': 'Pensive & Neutralizing', 'spotify_url': 'https://open.spotify.com/track/5KW8kOHmXqkDEdj6JyP3dJ?si=Co8ieKqaR_aKdltNYK-LgA%0A_source=copy-link'},
    ],
    'neutral': [ 
        {'name': 'Malare', 'artist': 'Rajesh Murugesan', 'mood': 'Gentle & Background', 'spotify_url': 'https://open.spotify.com/track/4Hvf9xIeJWp5p9FkJerQhN?si=M4Pu4YrTT_KLlvc4xFVXsA%0A_source=copy-link'},
        {'name': 'Konji Pesida Venaam', 'artist': 'Harris Jayaraj', 'mood': 'Gentle & Background', 'spotify_url': 'https://open.spotify.com/track/5QoCBy9eUHYkeWpEpzAud9?si=cFgfLBj8TwuuHljmPPdJAg%0A_source=copy-link'},
        {'name': 'The Life of Ram', 'artist': 'Govind Vasantha', 'mood': 'Gentle & Background', 'spotify_url': 'https://open.spotify.com/track/3hXX5v2JaVqDR3aeQcPAU9?si=YBHuFmwsTtCOq2C-tbP0WA%0A_source=copy-link'},
    ]
}


# GLOBAL DATA AND MODEL LOADING
@st.cache_data
def load_music_data():
    try:
        df = pd.read_csv(MUSIC_DATA_PATH, encoding='latin-1') 
        df.rename(columns={'track_name': 'name', 'track_artists': 'artists', 'track_genre': 'genre', 'track_popularity': 'popularity'}, inplace=True)
        return df
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame() 

try:
    music_df = load_music_data()
except Exception:
    music_df = pd.DataFrame() 


# EMOTION PREDICTION BY DEEPFACE 

def get_emotion_from_input(input_data: Union[st.runtime.uploaded_file_manager.UploadedFile, str], input_type: str, detector_backend: str = 'opencv'):
    
    # IMAGE ANALYSIS (DEEPFACE)
    if input_type == 'image' and DeepFace is not None:
        if not input_data: return "neutral" 

        try:
            input_data.seek(0) 
            image_bytes = input_data.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_array is None:
                st.error("Image decode failed.")
                return "neutral" 

            results = DeepFace.analyze(
                img_array, 
                actions=['emotion'], 
                enforce_detection=True, 
                detector_backend=detector_backend 
            )
            
            detected_emotion = results[0]['dominant_emotion']
            st.success(f"Emotion detected: **{detected_emotion.capitalize()}**")
            return detected_emotion
            
        except ValueError as e:
            if "Face could not be detected" in str(e):
                st.error("Face not detected. Ensure your face is clearly visible and well-lit.")
            else:
                st.error(f"Error: {e}")
            return "neutral"
            
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return "neutral"
    
    # TEXT EMOTION ANALYSIS 
    elif input_type == 'text':
        text = str(input_data).lower()
        st.info("Using Rule-Based Text Sentiment Detection.")

        # Simple keyword matching
    if any(word in text for word in ["angry", "stress", "furious", "hate", "kill", "mad", "irritated", "frustrated", "rage", "livid", "annoyed", "pissed", "venom", "betray"]): 
        return "angry"
    if any(word in text for word in ["disgust", "nasty", "sick", "ew", "chi", "gross", "repulsed", "ugh", "vomit", "foul", "awful", "trash", "bad smell"]): 
        return "disgust"
    if any(word in text for word in ["scared", "fear", "anxious", "worried", "terrified", "panic", "horror", "nervous", "dread", "frightened", "alone", "ghost"]): 
        return "fear"
    if any(word in text for word in ["happy", "great", "joy", "amazing", "love", "enthusiastic", "excited", "fantastic", "wonderful", "glad", "cheer", "bliss", "good", "excellent", "fun", "celebrate"]): 
        return "happy"
    if any(word in text for word in ["sad", "depress", "miserable", "down", "unhappy", "cry", "grief", "lonely", "tear", "bad", "sorrow", "pain", "miss", "lost"]): 
        return "sad"
    if any(word in text for word in ["shock", "wow", "surprise", "unexpect", "gasp", "astonish", "amazing", "stunning", "sudden", "holy", "unbelievable"]): 
        return "surprise"

    return "neutral"


# MUSIC RECOMMENDATION FUNCTION 

def get_music_recommendation(detected_emotion: str) -> pd.DataFrame:
    
    detected_emotion = detected_emotion.lower()
    
    song_list = CUSTOM_PLAYLIST.get(detected_emotion, [])
    
    if not song_list:
        return pd.DataFrame() 

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(song_list)
    
    # Add genre label
    df['genre'] = 'Custom Tamil Playlist'
    
    df.rename(columns={'artist': 'artists'}, inplace=True)

    # Return the necessary columns
    return df[['name', 'artists', 'mood', 'genre', 'spotify_url']]


# STREAMLIT FRONTEND 

def init_session_state():
    if 'detected_emotion' not in st.session_state: st.session_state.detected_emotion = None
    if 'image_key' not in st.session_state: st.session_state.image_key = 0
    if 'detected_emotion_text' not in st.session_state: st.session_state.detected_emotion_text = None

def main():
    init_session_state()
    st.set_page_config(page_title="MoodMate Music App 🎶", layout="wide") 
    
    st.title("MoodMate: Emotion-Based Music Recommender 🎶🫠💃🏻")
    
    model_status = "Model ACTIVE" if DeepFace else "Model INACTIVE (DeepFace not available)"
    st.markdown(f'<p style="text-align:center; color:grey;">{model_status}</p>', unsafe_allow_html=True)
    st.markdown("---")
    

    input_choice = st.radio(
        "Choose your mood detection method:", 
        ('Image Input (Camera or File)', 'Text Sentiment Analysis'),
        key='input_method',
        horizontal=True
    )

    emotion_output = None 
    
    col1, col2 = st.columns([1, 1])

    with col1:
        if input_choice.startswith('Image Input'):
            
            st.session_state.detected_emotion_text = None
            
            image_mode = st.radio(
                "Choose image source:", 
                ('Upload File', 'Use Camera '), 
                key='image_mode_radio', 
                horizontal=True,
                on_change=lambda: st.session_state.update(detected_emotion=None) 
            )
            
            st.markdown("---")
            
            input_data = None
            detector_to_use = 'opencv' 
            if image_mode == 'Upload File':
                uploaded_file = st.file_uploader("Upload an image of your face", type=['jpg', 'jpeg', 'png'])
                input_data = uploaded_file
                if input_data: 
                    st.image(input_data, caption='Image Uploaded.', width=300) 
                    detector_to_use = 'opencv'
                    
            else: 
                camera_file = st.camera_input("Click 'Take Photo' to capture your current mood:", key=f"camera_{st.session_state.image_key}")
                input_data = camera_file
                detector_to_use = 'mtcnn' 
            
            if st.button('Analyze Emotion', use_container_width=True, key='analyze_image_btn'):
                if DeepFace is None:
                    st.error("Cannot analyze image: DeepFace is not available.")
                elif input_data:
                    with st.spinner(f'Analyzing image using {detector_to_use.capitalize()}...'):
                        st.session_state.detected_emotion = get_emotion_from_input(input_data, 'image', detector_backend=detector_to_use)

                        if image_mode == 'Use Camera ':
                             st.session_state.image_key += 1
                             st.rerun() 
                else:
                    st.error("Please upload a file or take a photo first.")
                    st.session_state.detected_emotion = None

            emotion_output = st.session_state.detected_emotion
            
        elif input_choice.startswith('Text Sentiment'):
            
            st.session_state.detected_emotion = None
            
            user_text = st.text_area("Tell me how you are feeling today:", placeholder="I love Youuu!!!")
            
            if st.button('Analyze Text Emotion', use_container_width=True, key='analyze_text_btn') and user_text:
                with st.spinner('Analyzing text...'):
                    st.session_state.detected_emotion_text = get_emotion_from_input(user_text, 'text')

            emotion_output = st.session_state.detected_emotion_text

    with col2:
        if emotion_output:
            display_emotion = emotion_output.capitalize()
            
            st.subheader(f"Detected Emotion: {display_emotion}")

            music_recommendations = get_music_recommendation(emotion_output)
            
            if not music_recommendations.empty:
                target_mood = music_recommendations['mood'].iloc[0]
                st.info(f"Generating custom Tamil songs aligned with a **{target_mood}** mood.")

                st.subheader("Your Tamil Playlist 🎶💃🏻")
                # USE INDIVIDUAL CARDS
                for index, row in music_recommendations.iterrows():
                    song_title = row['name']
                    artist = row['artists']
                    spotify_url = row['spotify_url']
                    target_mood_item = row['mood']
                    
                    # Using st.markdown with custom HTML for a individual cards
                    st.markdown(f"""
                    <div style='
                        padding: 12px; 
                        border-left: 5px solid #1DB954; 
                        border-radius: 6px; 
                        margin-bottom: 10px; 
                        background-color: #f8f8fa; 
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    '>
                        <p style='margin: 0; font-size: 1.1em;'>
                            🎵 **{song_title}** <span style='font-style: italic; color: #555;'>by {artist}</span>
                        </p>
                        <p style='margin: 5px 0 10px 0; font-size: 0.9em; color: #777;'>
                            Mood Alignment: <span style='color:#007bff; font-weight: bold;'>{target_mood_item}</span>
                        </p>
                        <a href="{spotify_url}" target="_blank" style='text-decoration: none;'>
                            <button style='
                                background-color: #1DB954; 
                                color: white; 
                                padding: 7px 15px; 
                                border: none; 
                                border-radius: 20px; 
                                cursor: pointer;
                                font-weight: 600;
                            '>
                                🎶 Play on Spotify (Direct Link)
                            </button>
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.caption("This link uses the direct URL you provided to open the song immediately on Spotify.")

                # END OF DISPLAY
            else:
                st.warning("No specific music found for this mood.")
#ENTRY POINT OF AN APPLICATION
if __name__ == '__main__':
    main()