import streamlit as st
from utils.text2image import generate_sentence_image
from utils.word_motion import compute_word_motion
from utils.sentiment import get_sentiment_scores
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from io import BytesIO

nltk.download("punkt")

st.set_page_config(page_title="MindPaint Literal v3", layout="wide")
st.title("🎨 MindPaint Literal Storytelling")
st.markdown("Each sentence becomes a literal scene, words animate over it, and hover shows sentiment!")

# Sidebar Controls
st.sidebar.header("Animation Controls")
fps = st.sidebar.slider("FPS", 1, 30, 10)
trail_factor = st.sidebar.slider("Trail factor", 0.0, 0.99, 0.9)
glow_intensity = st.sidebar.slider("Glow intensity", 0.0, 1.0, 0.15)

# Text input
user_text = st.text_area("Enter paragraph (multiple sentences):", height=150)
animate_btn = st.button("Generate & Animate")

if animate_btn and user_text.strip():
    sentences = [s.strip() for s in user_text.split(".") if s.strip()]

    for idx, sentence in enumerate(sentences):
        st.text(f"Generating image for sentence {idx+1}: {sentence}")
        img = generate_sentence_image(sentence)
        
        words = word_tokenize(sentence)
        words = [w for w in words if w.isalpha()]
        sentiments = get_sentiment_scores(words)
        
        # ---------- Matplotlib Animation ----------
        fig, ax = plt.subplots(figsize=(6,6))
        ax.axis("off")
        img_plot = ax.imshow(img)

        sc = ax.scatter([], [], c="red", s=80, alpha=0.8)

        def update(frame):
            offsets = compute_word_motion(len(words), frame)
            # Apply trail and glow
            offsets = offsets * 5 + 50
            sc.set_offsets(offsets)
            return [sc, img_plot]

        ani = FuncAnimation(fig, update, frames=60, interval=100, blit=True)

        # ---------- Streamlit Display ----------
        buf = BytesIO()
        ani.save(buf, writer=PillowWriter(fps=fps))
        buf.seek(0)
        st.image(buf, caption=f"Sentence {idx+1} Animation", use_column_width=True)

    st.success("✅ All sentences animated!")
