import numpy as np

def compute_word_motion(num_words, frame):
    """
    Returns x,y offsets for each word at current frame for live animation.
    """
    angles = np.linspace(0, 2*np.pi, num_words, endpoint=False)
    radius = 0.05 + 0.05*np.sin(frame/10)
    x_offsets = radius * np.cos(angles + frame/20)
    y_offsets = radius * np.sin(angles + frame/20)
    return np.stack([x_offsets, y_offsets], axis=1)
