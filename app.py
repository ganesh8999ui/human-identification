import streamlit as st
import cv2
import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
import gdown
import os
import tempfile

# Google Drive File IDs
MODEL_ID = "1EYExPGwRhsnzfgoKaqBcQ3V9__J6gaBH"
CLASS_NAMES_ID = "1qPUqFN4xxf6fkWA72RUKYeE2LUcodce6"
GANESH_VP_ID = "1sOuwqjUQeYYhcTSJyRKNGf0_prH3uvXG"
SUDARSHAN_VP_ID = "1hBq1fy4HvCjLY2be2m7_81COWN_YdPVO"
IMAGE_SIZE = (200, 200)

@st.cache_resource
def load_models():
    st.info("Models load hot āhēt, thoda wait kar...")

    # Download files from Google Drive
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", "model.keras", quiet=False)
    gdown.download(f"https://drive.google.com/uc?id={CLASS_NAMES_ID}", "class_names.npy", quiet=False)
    gdown.download(f"https://drive.google.com/uc?id={GANESH_VP_ID}", "ganesh_fp.npy", quiet=False)
    gdown.download(f"https://drive.google.com/uc?id={SUDARSHAN_VP_ID}", "sudarshan_fp.npy", quiet=False)

    model = load_model("model.keras")
    class_names = np.load("class_names.npy", allow_pickle=True)
    voice_fps = {
        "Ganesh": np.load("ganesh_fp.npy"),
        "Sudarshan": np.load("sudarshan_fp.npy")
    }
    return model, class_names, voice_fps

def get_voice_fingerprint(audio_path, n_chunks=20):
    sr, y = wavfile.read(audio_path)
    y = y.astype(float)
    if len(y.shape) > 1:
        y = y[:, 0]
    chunk_size = len(y) // n_chunks
    if chunk_size == 0:
        return None
    fingerprint = []
    for i in range(n_chunks):
        chunk = y[i*chunk_size:(i+1)*chunk_size]
        fingerprint.append(np.std(chunk))
    return np.array(fingerprint)

def identify_face(image, model, class_names):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return None, 0, img
    x, y, w, h = faces[0]
    face_img = cv2.resize(img[y:y+h, x:x+w], IMAGE_SIZE)
    face_array = np.expand_dims(face_img.astype("float32") / 255.0, axis=0)
    preds = model.predict(face_array, verbose=0)
    name = class_names[np.argmax(preds[0])]
    confidence = preds[0][np.argmax(preds[0])] * 100
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, f"{name} ({confidence:.1f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return name, confidence, img

st.title("Human Identification System")
st.subheader("Face + Voice Recognition")

model, class_names, voice_fps = load_models()
st.success("Models ready!")

tab1, tab2 = st.tabs(["Face Recognition", "Voice Recognition"])

with tab1:
    st.header("Face Recognition")
    uploaded_photo = st.file_uploader("Photo upload kara", type=["jpg", "jpeg", "png"])
    if uploaded_photo:
        image = Image.open(uploaded_photo)
        name, confidence, result_img = identify_face(image, model, class_names)
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption="Result", use_container_width=True)
        if name:
            if confidence > 80:
                st.success(f"Identified: {name} ({confidence:.1f}%)")
            else:
                st.warning(f"Possibly: {name} ({confidence:.1f}%)")
        else:
            st.error("Chehra sapdalaa nahi!")

with tab2:
    st.header("Voice Recognition")
    uploaded_voice = st.file_uploader("Voice file upload kara (.wav)", type=["wav"])
    if uploaded_voice:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_voice.read())
            tmp_path = tmp.name
        fp = get_voice_fingerprint(tmp_path)
        os.unlink(tmp_path)
        if fp is not None:
            results = {}
            for person_name, ref_fp in voice_fps.items():
                sim = (1 - cosine(fp, ref_fp)) * 100
                results[person_name] = sim
            best_name = max(results, key=results.get)
            best_sim = results[best_name]
            st.write("### Results:")
            for person_name, sim in results.items():
                st.progress(int(max(0, min(100, sim))), text=f"{person_name}: {sim:.1f}%")
            if best_sim > 70:
                st.success(f"Voice identified: {best_name} ({best_sim:.1f}%)")
            else:
                st.warning("Voice identify karta ala nahi")
