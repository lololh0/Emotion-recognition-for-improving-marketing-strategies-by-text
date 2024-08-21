import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load the model
model_path = r"C:\Users\lenovo\Videos\ERIMS Pr\2024(ERIMS).pkl"
pipe_lr = joblib.load(model_path)

# Define emotions_emoji_dict
emotions_emoji_dict = {
    "boredom": "😒",
    "anger": "😠",
    "worry": "😬",
    "hate": "😣",
    "happiness": "🤗🙂",
    "relief": "😌",
    "neutral": "😐",
    "fun": "😅",
    "empty": "🙄",
    "sadness": "😔",
    "love": "🥰",
    "surprise": "😮",
    "enthusiasm": "😀"
}

# Define predict_emotions function
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

# Define get_prediction_proba function
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    st.set_page_config(page_title="Emotion Recognition for Improving Marketing Strategies")
    st.markdown("<h1 style='text-align: center; font-size: 31px; color: #218b60 '>"
                "𝑬𝒎𝒐𝒕𝒊𝒐𝒏 𝑹𝒆𝒄𝒐𝒈𝒏𝒊𝒕𝒊𝒐𝒏 𝒇𝒐𝒓 𝑰𝒎𝒑𝒓𝒐𝒗𝒊𝒏𝒈 𝑴𝒂𝒓𝒌𝒆𝒕𝒊𝒏𝒈 𝑺𝒕𝒓𝒂𝒕𝒆𝒈𝒊𝒆𝒔</h1>",
                unsafe_allow_html=True)

    st.markdown(
        "<h2 style='text-align: left; font-size: 35px; color: #fff;'></h2>",
        unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: left; font-size: 15px; color: #c06b3e;'>Enter your comments or feedback here...</h2>",
                unsafe_allow_html=True)

    with st.form(key='my_form'):
        raw_text = st.text_area("Enter your text here:", label_visibility="collapsed")
        submit_text = st.form_submit_button(label='Start analyzing')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.markdown('<p style="color: #c06b3e;"> The original typed text:</p>', unsafe_allow_html=True)
            st.write(raw_text)

            print("Prediction:", prediction)  # Debugging output

            if prediction in emotions_emoji_dict:
                emoji_icon = emotions_emoji_dict[prediction]
                st.markdown('<p style="color: #c06b3e;">The emotion recognized is:</p>', unsafe_allow_html=True)
                st.write("{}".format(emoji_icon))
                st.write("{}".format(prediction), "{:.0f}%".format(np.max(probability) * 100))
            else:
                st.write("Unknown emotion")

            st.markdown('<p style="color: #c06b3e; ">The Probability Percentage:</p>', unsafe_allow_html=True)
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]
            proba_df_clean['percentage'] = proba_df_clean['probability'] * 100
            st.write(proba_df_clean[['emotions', 'percentage']])

            st.markdown('<p style="color: #c06b3e;">Prediction and probability:</p>', unsafe_allow_html=True)
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
