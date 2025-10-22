import numpy as np

def compute_semantic_motion(words, sentiments, frame):
    """
    Return offsets and motion vectors based on word meaning + sentiment
    """
    offsets = []
    for i, word in enumerate(words):
        angle = frame/15 + i*0.5
        # Semantic patterns
        if "fox" in word.lower() or "sly" in word.lower():
            r = 0.05 + 0.02*np.sin(frame/10 + i)
            x = r*np.cos(angle)
            y = r*np.sin(angle)*0.5
        elif "wolf" in word.lower() or "cunning" in word.lower():
            r = 0.07 + 0.03*np.sin(frame/12 + i)
            x = r*np.cos(angle)*1.2
            y = r*np.sin(angle)*1.2
        elif "rose" in word.lower() or "flower" in word.lower():
            r = 0.03 + 0.02*np.sin(frame/8 + i)
            x = r*np.cos(angle)
            y = r*np.sin(angle)
        else:
            r = 0.04 + 0.01*np.sin(frame/20 + i)
            x = r*np.cos(angle)
            y = r*np.sin(angle)
        # Apply sentiment to vertical offset
        y += sentiments[i]*0.02
        offsets.append([x, y])
    return np.array(offsets)
