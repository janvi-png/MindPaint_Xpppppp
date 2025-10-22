from diffusers import StableDiffusionPipeline
import torch

# Load model (CUDA if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
).to(device)

def generate_sentence_image(sentence, style="digital art, cinematic"):
    """
    Generates a literal image from the input sentence.
    """
    prompt = f"{sentence}, {style}"
    image = pipe(prompt, guidance_scale=7.5).images[0]
    return image
