import streamlit as st
from utils.text2image import generate_sentence_image
from utils.word_motion import compute_semantic_motion
from utils.sentiment import get_sentiment_scores
from utils.tfidf_weights import compute_word_weights

from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
from io import BytesIO

st.set_page_config(page_title="MindPaint Ultimate v6", layout="wide")
st.title("🎨 MindPaint Ultimate v6")
st.markdown("Fully interactive cinematic animation: literal motion + glow + hover sentiment!")

# Sidebar
fps = st.sidebar.slider("FPS", 5, 30, 10)
trail_factor = st.sidebar.slider("Trail factor", 0.0, 0.99, 0.85)
glow_intensity = st.sidebar.slider("Glow intensity", 0.0, 1.0, 0.25)
particle_scale = st.sidebar.slider("Particle scale", 20, 200, 100)

# Text input
user_text = st.text_area("Enter paragraph:", height=200)
animate_btn = st.button("Generate Cinematic Animation")

if animate_btn and user_text.strip():
    sentences = [s.strip() for s in user_text.split(".") if s.strip()]
    word_weights = compute_word_weights(sentences)

    st.text("Generating literal images per sentence...")
    images, sentence_words, sentence_sentiments, sentence_sizes = [], [], [], []

    for s in sentences:
        img = generate_sentence_image(s)
        images.append(np.array(img))
        words = [w for w in word_tokenize(s) if w.isalpha()]
        sentence_words.append(words)
        sentiments = get_sentiment_scores(words)
        sentence_sentiments.append(sentiments)
        sizes = [particle_scale*word_weights.get(w.lower(),1) for w in words]
        sentence_sizes.append(sizes)
    st.success("✅ Images ready!")

    fig, ax = plt.subplots(figsize=(6,6))
    ax.axis("off")
    img_plot = ax.imshow(images[0])
    sc = ax.scatter([], [], c=[], s=[], alpha=0.8)
    annotation = ax.annotate("", xy=(0,0), xytext=(10,10),
                             textcoords="offset points", color="white",
                             bbox=dict(boxstyle="round", fc="black", alpha=0.5))
    annotation.set_visible(False)

    frame_count = 80
    total_frames = frame_count * len(sentences)

    def update(frame):
        sentence_idx = min(frame // frame_count, len(sentences)-1)
        local_frame = frame % frame_count

        # Interpolate image
        img_curr = images[sentence_idx].astype(np.float32)
        if sentence_idx < len(sentences)-1:
            img_next = images[sentence_idx+1].astype(np.float32)
            t = local_frame/frame_count
            img_plot.set_data((1-t)*img_curr + t*img_next)
        else:
            img_plot.set_data(img_curr)

        # Particle animation
        words = sentence_words[sentence_idx]
        sentiments = sentence_sentiments[sentence_idx]
        sizes = sentence_sizes[sentence_idx]
        offsets = compute_semantic_motion(words, sentiments, local_frame)
        offsets = offsets*5 + 50
        sc.set_offsets(offsets)
        colors = [(0.5*(1+s), 0.5*(1-s), 0.3 + glow_intensity*np.sin(local_frame/5)) for s in sentiments]
        sc.set_color(colors)
        sc.set_sizes(sizes)
        return [sc, img_plot]

    ani = FuncAnimation(fig, update, frames=total_frames, interval=100, blit=True)

    st.text("Rendering MP4... this will take a while")
    ani.save("MindPaint_Ultimate_v6.mp4", writer=FFMpegWriter(fps=fps), dpi=150)
    st.success("✅ MindPaint_Ultimate_v6.mp4 saved!")
    st.video("MindPaint_Ultimate_v6.mp4")
