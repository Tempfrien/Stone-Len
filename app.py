import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from style_config import apply_custom_style

# --- ZONE 1: ตั้งค่าระบบ (Setup & Session) ---
st.set_page_config(page_title="STONE LEN - Rock Classification", layout="wide")
apply_custom_style()

if 'page' not in st.session_state:
    st.session_state.page = 'Main'

def change_page(name):
    st.session_state.page = name

# --- ZONE 2: Logic AI (โหลดโมเดล) ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5", compile=False)

def load_labels():
    with open("labels.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

model = load_model()
labels = load_labels()

# --- ZONE 3: องค์ประกอบลอยตัว (Fixed Image) ---
st.markdown("""
    <div class="fixed-image">
        <img src="https://lh3.googleusercontent.com/u/0/d/1j2yrrBp-xXv1vfk4fdrIxZxVmyX4Bszu">
    </div>
    """, unsafe_allow_html=True)

# --- ZONE 4: หน้าหลัก (AI Classification) ---
if st.session_state.page == 'Main':
    st.markdown('<h1 class="main-title">STONE LEN</h1>', unsafe_allow_html=True)
    
    if st.button("📖 เรียนรู้ลักษณะหิน"):
        change_page('Knowledge')
        st.rerun()

    st.markdown("""
    <p style="color: white; font-size: 20px; text-shadow: 1px 1px 5px rgba(0,0,0,0.8);
              position: relative; top: -10px; left: 10px;">
        ROCK CLASSIFICATION WEBSITE : เว็บไซต์จำแนกประเภทหิน เพื่อการศึกษาทางธรณีวิทยา<br>
        <b>วิธีใช้งาน</b><br>
        - ถ่ายรูปภาพหินที่ต้องการทราบชนิด หรือ<br>
        - อัปโหลดรูปภาพลงในช่องอัปโหลด<br>
        <b>ข้อแนะนำ</b><br>
          หากความแม่นยำต่ำกว่า60%<br>
          ควรเปลี่ยนพื้นหลังเป็นสีขาวจะเพิ่มความแม่นยำในการตรวจสอบ
    </p>
    """, unsafe_allow_html=True)

    # แก้ไขระบบ Tabs ให้ถูกต้อง (ตัด CSS ออกจากฟังก์ชัน Python)
    tab1, tab2 = st.tabs(["📸 ถ่ายภาพสด", "📁 อัปโหลดไฟล์"])
    source_img = None

    with tab1:
        # แก้ไข camera_input ให้ถูกต้อง
        cam_file = st.camera_input("ถ่ายภาพหินจากกล้อง")
        if cam_file:
            source_img = cam_file

    with tab2:
        # แก้ไข file_uploader ให้ถูกต้อง
        uploaded_file = st.file_uploader("เลือกรูปภาพจากเครื่อง", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            source_img = uploaded_file

    if source_img is not None:
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        image = Image.open(source_img).convert("RGB")
        
        with col1:
            st.image(image, caption="รูปที่ใช้ประมวลผล", use_container_width=True)
        
        # ประมวลผล AI
        size = (224, 224)
        image_processed = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image_processed)
        normalized_img = (img_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_img
        
        prediction = model.predict(data)
        index = np.argmax(prediction)
        
        with col2:
            st.markdown(f"""
                <div class="result-box">
                    <h2 style='text-align:center;'>🔍 ผลการวิเคราะห์</h2>
                    <hr>
                    <p style='font-size:20px;'>หินชนิดนี้คือ: <b style='color:#dcb799;'>{labels[index]}</b></p>
                    <p style='font-size:18px;'>ความแม่นยำ: <b>{prediction[0][index] * 100:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)

# --- ZONE 5: หน้าความรู้ (Rock Info) ---
elif st.session_state.page == 'Knowledge':
    st.markdown('<h1 class="main-title">ROCK INFO</h1>', unsafe_allow_html=True)
    
    if st.button("🔙 กลับไปหน้าวิเคราะห์"):
        change_page('Main')
        st.rerun()

    st.markdown("""
        <div class="result-box">
            <h2 style="color:#2d3e33;">ลักษณะภายนอกของหิน</h2>
            <hr>
            <h3>1. หินอัคนี (Igneous Rock)</h3>
                    <p>-เนื้อผลึกหยาบสอดประสานกันแน่น มองเห็นผลึกแร่แต่ละชนิดได้ด้วยตาเปล่า เช่น แกรนิต, ไดโอไรต์</p>
                    <p>- เนื้อละเอียดหรือเป็นแก้วเพราะเย็นตัวเร็วมาก ตัวอย่างเช่น หินออบซิเดียนที่เรียบมันเหมือนแก้ว</p>
                    <p>-เหินพัมมิส หรือ หินสคอเรีย จะมีฟองก๊าซที่ติดอยู่ขณะลาวาเย็นตัว ทำให้หินมีน้ำหนักเบาและอาจลอยน้ำได้</p>
            <br>
            <h3>2. หินตะกอน (Sedimentary Rock)</h3>
            <p>-การเรียงตัวเป็นชั้นชัดเจน เนื้อหินมักประกอบด้วยเม็ดตะกอนขนาดต่าง ๆ ผิวสัมผัสมีความหยาบหรือละเอียด</p>
            <p>-อาจพบซากดึกดำบรรพ์เศษหินหรือร่องรอยการไหลของน้ำ/ลมสีสันหลากหลายมักเป็นสีน้ำตาล แดง เทา หรือขาว ตามแร่ธาตุที่เชื่อมประสา</p>
            <br>
            <h3>3. หินแปร (Metamorphic Rock)</h3>
            <p>-หินแปรส่วนใหญ่จะมีลักษณะที่แร่ธาตุเรียงตัวขนานกันหรือเป็นแผ่นบางๆ ตามทิศทางของแรงดัน ทำให้หินมีแนวแตกเรียบ หรือแตกออกเป็นแผ่นได้ง่าย</p>
            <p>-เป็นหินแปรที่เนื้อแน่น แร่ธาตุไม่เรียงตัวเป็นชั้น มักเกิดจากการแปรสภาพแบบสัมผัส (ความร้อนสูง)</p>
            <br>
            <br>
            <b>แหล่งที่มา :</b> Mindat.org
        </div>
    """, unsafe_allow_html=True)

# --- ZONE 6: แถบรายชื่อผู้พัฒนา ---
st.markdown("""
    <div class="footer-bar">
        Creators : Chadaporn Boonnii, Nopphanat Junnunl, Saranya Changkeb, Phatcharakamon Sodsri
    </div>
    """, unsafe_allow_html=True)
