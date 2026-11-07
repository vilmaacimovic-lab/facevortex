import streamlit as st
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
from PIL import Image
import io

os.environ["LANG"] = "en_US.UTF-8"

st.set_page_config(
    page_title="FaceVortex | AI Studio",
    page_icon="🌀",
    layout="centered"
)

st.title("🌀 FaceVortex Studio")
st.markdown("<h3 style='color: #ff4b4b;'>Aplikacija za zamenu lica</h3>", unsafe_allow_html=True)

# Učitavanje modela
@st.cache_resource
def load_model():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app

with st.spinner("Pokretanje AI modela..."):
    face_app = load_model()

model_path = "inswapper_128.onnx"

st.markdown("### Izaberi sliku za promenu")
target_file = st.file_uploader("1. Ubaci glavnu sliku (gde menjaš lice)", type=["jpg", "jpeg", "png"])
source_file = st.file_uploader("2. Ubaci novo lice koje stavljaš", type=["jpg", "jpeg", "png"])

if target_file is not None and source_file is not None:
    img_target = cv2.imdecode(np.frombuffer(target_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img_source = cv2.imdecode(np.frombuffer(source_file.read(), np.uint8), cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB), caption="Glavna slika", use_container_width=True)
    st.image(cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB), caption="Novo lice", use_container_width=True)

    if st.button("🚀 POKRENI ZAMENU LICA"):
        if not os.path.exists(model_path):
            st.warning("Napomena: Potrebno je ubaciti inswapper_128.onnx model u repozitorijum.")
        else:
            with st.spinner("AI menja lice..."):
                swapper = insightface.model_zoo.get_model(model_path, download=False, download_zip=False)
                t_faces = face_app.get(img_target)
                s_faces = face_app.get(img_source)

                if len(t_faces) == 0 or len(s_faces) == 0:
                    st.error("AI nije detektovao lice na jednoj od slika.")
                else:
                    res_img = img_target.copy()
                    for t_face in t_faces:
                        res_img = swapper.get(res_img, t_face, s_faces[0], paste_back=True)

                    st.success("Uspešno završeno!")
                    final_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                    st.image(final_rgb, caption="Rezultat", use_container_width=True)
