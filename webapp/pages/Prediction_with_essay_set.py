import streamlit as st
import pandas as pd
from src.webapp_features import *
import joblib

# st.set_page_config(initial_sidebar_state="collapsed")
model = joblib.load("../src/models/xgb_regressor.joblib")
st.set_page_config(layout = "wide")

sample_essay = """
As soon as the patient room door opened, the worst stench I have ever encountered hit me square in the face. Though I had never smelled it before, I knew instinctively what it was: rotting flesh. A small, elderly woman sat in a wheelchair, dressed in a hospital gown and draped in blankets from the neck down with only her gauze-wrapped right leg peering out from under the green material. Dr. Q began unwrapping the leg, and there was no way to be prepared for what I saw next: gangrene-rotted tissue and blackened, dead toes.

Never before had I seen anything this gruesome-as even open surgery paled in comparison. These past two years of shadowing doctors in the operating room have been important for me in solidifying my commitment to pursue medicine, but this situation proved that time in the operating room alone did not quite provide a complete, accurate perspective of a surgeon's occupation. Doctors in the operating room are calm, cool, and collected, making textbook incisions with machine-like, detached precision. It is a profession founded solely on skill and technique-or so I thought. This grisly experience exposed an entirely different side of this profession I hope to pursue.

Feeling the tug of nausea in my stomach, I forced my gaze from the terrifying wound onto the hopeful face of the ailing woman, seeking to objectively analyze the situation as Dr. Q was struggling to do himself. Slowly and with obvious difficulty, Dr. Q explained that an infection this severe calls for an AKA: Above the Knee Amputation. In the slow, grave silence that ensued, I reflected on how this desperate patient's very life rests in the hands of a man who has dedicated his entire life to making such difficult decisions as these. I marveled at the compassion in Dr. Q's promise that this aggressive approach would save the woman's life. The patient wiped her watery eyes and smiled a long, sad smile. “I trust you, Doc. I trust you.” She shook Dr. Q's hand, and the doctor and I left the room.

Back in his office, Dr. Q addressed my obvious state of contemplation: “This is the hardest part about what we do as surgeons,” he said, sincerely. “We hurt to heal, and often times people cannot understand that. However, knowing that I'm saving lives every time I operate makes the stress completely worth it.”

Suddenly, everything fell into place for me. This completely different perspective broadened my understanding of the surgical field and changed my initial perception of who and what a surgeon was. I not only want to help those who are ill and injured, but also to be entrusted with difficult decisions the occupation entails. Discovering that surgery is also a moral vocation beyond the generic application of a trained skill set encouraged me. I now understand surgeons to be much more complex practitioners of medicine, and I am certain that this is the field for me.
"""

def display_essay(user_input):
    if user_input != "":
        st.session_state["essay"] = user_input
        st.session_state["flesch_reading_ease"] = round(get_flesch_reading_ease(user_input),2)
        st.session_state["gunning_fog"] = round(get_gunning_fog(user_input),2)
        st.session_state["automated_readability_index"] = round(get_automated_readability_index(user_input),2)
        st.session_state["smog_index"] = round(get_smog_index(user_input),2)
        st.session_state["flesch_kincaid_grade"] = round(get_flesch_kincaid_grade(user_input),2)
        st.session_state["coleman_liau_index"] = round(get_coleman_liau_index(user_input),2)
        st.session_state["dale_chall_readability_score"] = round(get_dale_chall_readability_score(user_input),2)
        st.session_state["difficult_words"] = round(get_difficult_words(user_input),2)
        st.session_state["linsear_write_formula"] = round(get_linsear_write_formula(user_input),2)

        st.session_state["count_characters"] = count_characters(user_input)
        st.session_state["count_words"] = count_words(user_input)
        st.session_state["count_sentences"] = count_sentences(user_input)
        st.session_state["count_syllables"] = count_syllables(user_input)
        st.session_state["count_punctuation"] = count_punctuation(user_input)
        st.session_state["count_stopwords"] = count_stop_words(user_input)
        st.session_state["lexical_diversity"] = round(calculate_lexical_diversity(user_input),2)
        data = {"essay": user_input}
        df = pd.DataFrame(data, index=[0])
        df = feature_engineering(df)
        df = df.drop(["essay"], axis=1)
        print(df.columns)
        predictions = model.predict(df)
        print(predictions)
        
    else:
        st.error("Your essay is too short to be analyzed", icon="⚠️")

def reset():
    for key in st.session_state.keys():
        st.session_state.pop(key)

if "essay" not in st.session_state:
    st.session_state["essay"] = ""
if "flesch_reading_ease" not in st.session_state:
    st.session_state["flesch_reading_ease"] = None
if "gunning_fog" not in st.session_state:
    st.session_state["gunning_fog"] = None
if "automated_readability_index" not in st.session_state:
    st.session_state["automated_readability_index"] = None
if "smog_index" not in st.session_state:
    st.session_state["smog_index"] = None
if "flesch_kincaid_grade" not in st.session_state:
    st.session_state["flesch_kincaid_grade"] = None
if "coleman_liau_index" not in st.session_state:
    st.session_state["coleman_liau_index"] = None
if "dale_chall_readability_score" not in st.session_state:
    st.session_state["dale_chall_readability_score"] = None
if "difficult_words" not in st.session_state:
    st.session_state["difficult_words"] = None
if "linsear_write_formula" not in st.session_state:
    st.session_state["linsear_write_formula"] = None
if "predicted_essay_set" not in st.session_state:
    st.session_state["predicted_essay_set"] = None
if "predicted_grade" not in st.session_state:
    st.session_state["predicted_grade"] = None

if "count_characters" not in st.session_state:
    st.session_state["count_characters"] = None
if "count_words" not in st.session_state:
    st.session_state["count_words"] = None
if "count_sentences" not in st.session_state:
    st.session_state["count_sentences"] = None
if "count_syllables" not in st.session_state:
    st.session_state["count_syllables"] = None
if "count_stopwords" not in st.session_state:
    st.session_state["count_stopwords"] = None
if "lexical_diversity" not in st.session_state:
    st.session_state["lexical_diversity"] = None
if "count_punctuation" not in st.session_state:
    st.session_state["count_punctuation"] = None

######

col1, col2 = st.columns([3,2], gap="medium")

with col1:
    if st.session_state["essay"] == "":
        user_input = st.text_area(label="Write your essay here...", height=500)
        col1_1, col1_2 = st.columns([3,1], gap="medium")
        with col1_1:
            load_essay = st.button("Load a sample essay", on_click=display_essay, args=(sample_essay,))
        with col1_2:
            validation_button = st.button("Predict grade", on_click=display_essay, args=(user_input,))
    else:
        user_input = st.markdown(st.session_state.essay)
        col1_1, col1_2 = st.columns([3,1], gap="medium")
        with col1_2:
            validation_button = st.button("Reset", on_click=reset)

with col2:
    tab1, tab2, tab3 = st.tabs(["Readability", "Text analysis", "Grading"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Flesh Reading Ease", value=st.session_state["flesch_reading_ease"], help="Measures text readability using a 100-point scale; higher scores indicate easier reading. Based on sentence length and word syllables")
            st.metric(label="Gunning Fog", value=st.session_state["gunning_fog"], help="Estimates the years of formal education needed to understand a text on a first reading. Higher scores suggest more complex writing")
            st.metric(label="Automated Readability Index", value=st.session_state["automated_readability_index"], help="Calculates the US grade level needed to comprehend a text. Based on characters per word and words per sentence")
            st.metric(label="Linsear Write Formula", value=st.session_state["linsear_write_formula"], help="Determines text readability, giving a score corresponding to a US grade level. It's based on sentence length and easy versus hard words")
            st.metric(label="SMOG index", value=st.session_state["smog_index"], help="Estimates the years of education needed to understand a piece of writing. Based on the number of complex words in three 10-sentence samples")
        with col2:
            st.metric(label="Flesh Kincaid Grade", value=st.session_state["flesch_kincaid_grade"], help="Assesses text readability by US school grade level. It considers sentence length and the number of syllables per word")
            st.metric(label="Coleman Liau Index", value=st.session_state["coleman_liau_index"], help="Predicts the US grade level required to understand a text, based on the number of letters per 100 words and sentences per 100 words")
            st.metric(label="Dale Chall Readability Index", value=st.session_state["dale_chall_readability_score"], help="Evaluates text complexity by comparing words to a list of easy words and analyzing sentence length. Scores correspond to US grade levels")
            st.metric(label="Difficult words", value=st.session_state["difficult_words"], help="Counts the number of words not on a predefined list of commonly understood words, indicating text complexity")
            m = st.markdown("""
                <style>
                div.stButton > button:first-child {
                    background-color: rgb(204, 49, 49);
                }
                </style>""", unsafe_allow_html=True)
            b = st.button("Go to methodology and sources")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Count of Characters", value=st.session_state["count_characters"], help="Count each character, excluding spaces")
            st.metric(label="Count of Words", value=st.session_state["count_words"])
            st.metric(label="Count of Sentences", value=st.session_state["count_sentences"])
            st.metric(label="Count of Syllables", value=st.session_state["count_syllables"])
            st.metric(label="Count of punctuation", value=st.session_state["count_punctuation"])
        with col2:
            if st.session_state["count_characters"] == None or st.session_state["count_words"] == None or st.session_state["count_sentences"] == None:
                characters_per_words = None
                words_per_sentences = None
            else:
                characters_per_words = round(st.session_state["count_characters"]/st.session_state["count_words"],2)
                words_per_sentences = round(st.session_state["count_words"]/st.session_state["count_sentences"],2)
            st.metric(label="Characters/Words", value=characters_per_words)
            st.metric(label="Words/Sentences", value=words_per_sentences)
            st.metric(label="Count of stopwords", value=st.session_state["count_stopwords"])
            st.metric(label="Lexical diversity", value=st.session_state["lexical_diversity"], help="The ratio of unique words on the entire essay")
    with tab3:
        st.metric("Predicted Essay Set", value=st.session_state["predicted_essay_set"])
        st.metric("Predicted Grade", value=st.session_state["predicted_grade"])