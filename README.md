# 🎨 AI-Powered Text-to-Image Generator (SD Turbo)  
### Internship Task – Talrn.com  
**Built using Stable Diffusion Turbo + Streamlit (Google Colab Compatible)**

---

## 📌 Overview

This project is an **AI-powered text-to-image generator**, developed as part of the Machine Learning Internship Task for **Talrn**.

The system converts any text description into high-quality AI-generated images using **Stable Diffusion Turbo (SD-Turbo)** — an extremely fast diffusion model optimized for **both CPU and GPU**.

This project demonstrates:

- ✔ Text-to-image generation  
- ✔ Adjustable style modes  
- ✔ Negative prompts  
- ✔ Multi-image generation  
- ✔ Web UI using Streamlit  
- ✔ CPU/GPU auto-detection  
- ✔ Prompt engineering for higher quality  
- ✔ Automatic image saving  
- ✔ Ethical AI usage guidelines  

All development and testing were done entirely in **Google Colab**.

---

## 📁 Project Structure
```
Text-to-image-generator-talrn/
│
├── app.py                       # Main Streamlit app
├── requirements.txt             # Python dependencies
├── samples/                     # Sample generated outputs
├── generated/                   # Auto-created output folder
├── notebooks/
│   └── ai_image_generator.ipynb # Full development notebook (Colab)
├── LICENSE
└── README.md
```

---

## 🚀 Features

### 🔹 **Text → Image Generation**
Generate 1–4 images instantly from any text prompt.

### 🔹 **Style Selection**
- Photorealistic  
- Artistic  
- Cartoon  

Each style automatically applies prompt-enhancement modifiers.

### 🔹 **Negative Prompts**
Example:
```
blurry, low quality, distorted hands, extra fingers
```

### 🔹 **Turbo Mode (Fast)**
- SD-Turbo requires **only 1 inference step**  
- Extremely fast on **CPU**  
- Faster on **GPU**

### 🔹 **Streamlit Web UI**
- User-friendly interface  
- Image previews  
- Download buttons  
- Progress indicator  
- Device info display (CPU/GPU)

### 🔹 **Automatic Image Saving**
Images saved with unique filenames inside:
```
generated/
```

---

## 🧠 Model Used: Stable Diffusion Turbo (SD-Turbo)

- Optimized for speed  
- Suitable for real-time prototype apps  
- Supports both CPU and GPU  
- Lightweight compared to SDXL  

HuggingFace Model:  
👉 https://huggingface.co/stabilityai/sd-turbo

---

## ⚙️ Installation (Local Machine)

### **1️⃣ Clone the repository**
```bash
git clone https://github.com/Ashweljohn/Text-to-image-generator-talrn.git
cd Text-to-image-generator-talrn
```

### **2️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Streamlit app**
```bash
streamlit run app.py
```

---

# 🧪 Running the Streamlit App in Google Colab (with ngrok)
This project was fully developed and tested in **Google Colab**, using ngrok to expose the Streamlit UI publicly.

---

## **1️⃣ Install all required packages**
```bash
!pip install streamlit diffusers transformers accelerate safetensors
!pip install torch --index-url https://download.pytorch.org/whl/cpu
!pip install pyngrok
```

---

## **2️⃣ Authenticate ngrok**

Replace `"YOUR_TOKEN_HERE"` with your ngrok auth token.

```python
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN_HERE")
```

---

## **3️⃣ Create a tunnel on port 6006**
```python
public_url = ngrok.connect(6006, proto="http")
public_url
```

A public URL will be generated (e.g.):
```
https://xxxx-xx-xx.ngrok-free.app
```

Open this URL to access the Streamlit UI.

---

## **4️⃣ Start Streamlit**
```bash
!streamlit run app.py --server.port=6006 --server.headless true &
```

Now refresh the ngrok URL — your app will be live.

---

## 📝 Notes
- Works on both CPU and GPU in Colab  
- SD-Turbo is optimized for **fast CPU generation**  
- A new ngrok URL is required for each Colab session  
- Ensure `app.py` exists in the working directory:
```bash
!ls
```

---

## 📸 Sample Generated Images
Sample outputs are included in the `samples/` folder.  
These demonstrate:

- Prompt performance  
- Style variations  
- SD-Turbo generation speed  

---

## 🎛 Prompt Engineering Tips

### ✔ Improve detail:
```
ultra-detailed, 8k, realistic textures, dramatic lighting, sharp focus
```

### ✔ Enhance style:
```
oil painting, digital art, anime style, 3D render
```

### ✔ Use negative prompts:
```
blurry, grainy, low resolution, distorted hands, extra limbs
```

---

## 🔐 Ethical AI Usage

This project follows responsible AI guidelines:

- 🚫 No explicit, harmful, or violent content  
- 🚫 No recreating realistic images of real people  
- ✔ Encourage watermarking  
- ✔ Transparency: All images are AI-generated  
- 🚫 No political or unethical material  

---

## 🧩 Limitations

- SD-Turbo prioritizes **speed over photorealism**  
- Output limited to **512×512 resolution**  
- CPU generation is slower  
- No fine-tuning included  
- Complex prompts may produce minor artifacts  

---

## 🚀 Future Improvements

- Add SDXL support  
- Add image-to-image (img2img) mode  
- Implement output gallery  
- Add watermarking  
- Support DreamBooth / LoRA training  
- Support higher resolution outputs  
- Provide Docker deployment  

---

## 💼 Author

**Ashwel John**  
Machine Learning Intern Applicant – Talrn  
📧 Email: **ashweljohn46@gmail.com**  
🔗 GitHub Profile: https://github.com/Ashweljohn  
🔗 Project Repository: https://github.com/Ashweljohn/Text-to-image-generator-talrn

---

