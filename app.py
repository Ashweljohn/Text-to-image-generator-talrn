
import streamlit as st
import torch
import uuid
import os
from diffusers import StableDiffusionPipeline
from typing import List

OUTPUT_DIR = "generated"
MODEL_ID = "stabilityai/sd-turbo"  
os.makedirs(OUTPUT_DIR, exist_ok=True)

@st.cache_resource
def load_model(model_id: str = MODEL_ID):
    """
    Loads the Stable Diffusion pipeline. Uses float16 on CUDA and float32 on CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{model_id}'. Make sure it's available and you have internet access.\nOriginal error: {e}"
        )

    pipe = pipe.to(device)
   
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    return pipe, device

pipe, device = load_model()
def generate_images(pipe: StableDiffusionPipeline, prompt: str, style: str,
                    negative_prompt: str, num_images: int) -> List[str]:
    """
    Generate images with minimal steps for CPU-friendly fast output.
    Returns list of filepaths.
    """
    style = (style or "none").lower().strip()
    engineered_prompt = prompt.strip()

    if style == "photorealistic":
        engineered_prompt += ", ultra realistic, photo, 8k, sharp details"
    elif style == "cartoon":
        engineered_prompt += ", cartoon, clean lines, vibrant colors"
    elif style == "artistic":
        engineered_prompt += ", oil painting, textured brush strokes"

    files = []
    generator = None
    if torch.cuda.is_available():
        generator = torch.Generator(device="cuda").manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))

    for _ in range(num_images):
        out = pipe(
            engineered_prompt,
            negative_prompt=negative_prompt or None,
            guidance_scale=1.0,       
            num_inference_steps=1,    
            generator=generator
        )
        img = out.images[0]
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        img.save(filepath)
        files.append(filepath)
    return files

st.set_page_config(page_title="Fast SD-Turbo Image Generator", layout="centered")
st.title("🎨 Fast Stable Diffusion — SD Turbo (CPU friendly)")
st.write(f"Using device: **{device.upper()}**")

with st.form("generate_form"):
    prompt = st.text_area("Enter prompt:", height=120, placeholder="A beautiful portrait of ...")
    style = st.selectbox("Select style:", ["None", "Photorealistic", "Cartoon", "Artistic"])
    negative_prompt = st.text_input("Negative prompt (optional):")
    num_images = st.slider("Number of images", 1, 4, 1)
    submit = st.form_submit_button("Generate")

if submit:
    if not prompt or prompt.strip() == "":
        st.error("Please enter a prompt.")
    else:
        try:
            with st.spinner("Generating images (fast)..."):
                generated_files = generate_images(pipe, prompt, style, negative_prompt, num_images)

            st.success("Images generated!")
            cols = st.columns(min(4, len(generated_files)))
            for i, fp in enumerate(generated_files):
                with open(fp, "rb") as f:
                    img_bytes = f.read()
                cols[i % len(cols)].image(img_bytes, use_column_width=True, caption=os.path.basename(fp))
                cols[i % len(cols)].download_button("Download", data=img_bytes, file_name=os.path.basename(fp), mime="image/png")

        except Exception as e:
            st.error(f"Image generation failed: {e}")

st.markdown("---")
st.info("Model: **sd-turbo** (fast, CPU-friendly). If you have GPU later, this app will use it automatically.")
st.caption("If you want higher-fidelity images and have GPU access, ask me to switch back to a GPU-optimized model and settings.")
