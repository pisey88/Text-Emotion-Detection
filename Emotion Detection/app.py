import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os
from text_cleaner import clean_text
import string
import re

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pipeline_logistic_regression.pkl")
pipe_rf = joblib.load(open(MODEL_PATH, "rb"))


emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}

def predict_emotions(docx):
    results = pipe_rf.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_rf.predict_proba([docx])
    return results


def main():
    st.title("Text Emotion Detection 🤖")
    st.subheader("Detect emotions in user-inputted text with ML")

    with st.form(key='emotion_form'):
        raw_text = st.text_area("Type your text here 👇", height=150)
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        if raw_text.strip() == "":
            st.warning("Please enter some text before submitting.")
        else:
            col1, col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.subheader("Original Text")
                st.write(raw_text)

                st.subheader("Prediction")
                emoji_icon = emotions_emoji_dict.get(prediction, "❓")
                st.markdown(f"**Prediction:**  `{prediction}` {emoji_icon}")
                st.markdown(f"**Confidence:**  `{np.max(probability) * 100:.2f}%`")

            with col2:
                st.subheader("Prediction Probability")

                # Format probabilities
                proba_df = pd.DataFrame(probability, columns=pipe_rf.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotions", "Probability"]

                chart = (
                    alt.Chart(proba_df_clean)
                    .mark_bar()
                    .encode(
                        x=alt.X("Emotions", sort="-y"),
                        y=alt.Y("Probability", scale=alt.Scale(domain=[0, 1])),
                        color="Emotions"
                    )
                    .properties(width=400, height=300)
                )

                st.altair_chart(chart, use_container_width=True)

if __name__ == '__main__':
    main()