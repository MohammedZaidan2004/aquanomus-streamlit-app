import tensorflow as tf
import numpy as np
import streamlit as st
import json
import httpx
import os
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

def get_recommendation(prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek/deepseek-r1-0528:free",  # atau model yang sesuai
        "messages": [
            {"role": "system", "content": "You are an expert in aquascape plant health."},
            {"role": "user", "content": prompt}
        ]
    }

    response = httpx.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=15)
    return response.json()['choices'][0]['message']['content']


# Load model sekali di awal
model = tf.keras.models.load_model("Model/ZDN_Model.h5", compile=False)

# Load class names
with open("Model/ZDN_Model_class_names.json", "r") as f:
    class_names = json.load(f)

#Tensorflow Model Prediction
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(150,150))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return int(predictions[0][0] > 0.5) #return index of max element


#Sidebar
st.sidebar.title("Menu")
app_mode = st.sidebar.selectbox("Select Page",["Home","Tentang Model","Deteksi Penyakit"])

#Home/Halaman Utama
if(app_mode=="Home"):
    st.header("Selamat Datang di Aplikasi Web Deteksi Penyakit Tanaman by Aquanomous! 🌿🔍")
    image_path = "Image/Streamlit Homepage.jpg"
    st.image(image_path,use_container_width=True)
    st.markdown("""
    ### Tentang Kami

    <p style="text-align: justify;">
    <b>Aquanomous</b> merupakan aplikasi berbasis website yang menawarkan fitur deteksi penyakit pada tanaman hias aquascape. Tanaman merupakan komponen utama dan paling krusial dalam aquascape, oleh karena itu <b>Aquanomus</b> bertujuan untuk membantu para hobbyist dan pemilik aquascape dalam memonitoring kesehatan tanamannya di dalam akuarium disertai dengan memberikan solusi dan langkah perawatan untuk tanaman yang terdeteksi terdapat gejala penyakit. Dalam memberikan solusi dan langkah perawatan yang akurat, kami menggunakan API <b>DeepSeek</b>.
    </p>
    """,unsafe_allow_html=True)

    st.markdown("""          
    ### Cara Kerja Aplikasi
    1. Foto terlebih dahulu daun tanaman yang ingin dianalisis dan simpan foto di file lokal perangkat.
    2. Pada aplikasi web, pergi ke menu **Dashboard**, lalu pilih sidebar **Deteksi Penyakit**.
    3. Pada bagian bawah halaman **Deteksi Penyakit** tekan ikon **Upload Gambar** untuk mengupload gambar/foto daun tanaman anda yang ingin dianalisis.
    4. **Hasil** deteksi akan langsung keluar setelah gambar terupload. Jika terdeteksi penyakit pada daun tanaman, maka aplikasi akan memberikan solusi dan langkah penanganan yang dapat dilakukan.
    
    **Catatan**: Menu lain seperti **Tentang Model** pada sidebar **Dashboard** juga dapat diakses untuk memberikan informasi terkait model yang digunakan.

    """,unsafe_allow_html=True)

#Tentang Model
if(app_mode=="Tentang Model"):
    st.markdown("""
                
    ### Model
    <p style="text-align: justify;">
    Untuk mendeteksi penyakit pada tanaman, kami membuat model <b>Convolutional Neural Network (CNN)</b> menggunakan library TensorFlow. Model ini terdiri atas lapisan <b>Convolutional Layer</b> yang digunakan untuk mengekstrak pola dari gambar yang dilatih. Selain itu, model juga dilengkapi dengan lapisan <b>BatchNormalization</b> dan <b>MaxPooling2D</b> agar dapat meningkatan performansi model saat training. Model kemudian dilatih menggunakan dataset kami dengan penyesuaian pada layer Dense akhir agar dapat sesuai dengan target output. Dengan pendekatan ini, model diharapkan mampu menangkap fitur dan pola penting dari gambar yang dilatih sehingga menghasilkan prediksi yang akurat.       
    </p>
                
    <b>Arsitektur Model</b>
    """, unsafe_allow_html=True)

    st.image("Image/Model Architecture.png")

    st.markdown("""
    ### Dataset
    Dalam melatih model, kami menggunakan dataset yang terdiri atas gambar-gambar daun tanaman sehat dan tanaman yang terkena penyakit. Dataset ini kami ambil sendiri dari tanaman hias aquascape kami. Dari dataset tersebut, kami membaginya menjadi dua kategori, yaitu tanaman sehat dan tanaman sakit. Berikut konfigurasi dataset yang kami gunakan:               
    
    1. Train (6082 gambar).
    2. Validation (2691 gambar).
    3. Test (22 gambar).
                
    **Catatan**: Dataset yang kami gunakan bukanlah gambar tanaman anubias, namun diharapkan dapat mengekstrasi pola daun yang serupa dengan daun yang dimiliki tanaman anubias.

    ### Hasil Training dan Metrics
    Kami melatih model kami selama 100 epoch. Berikut adalah hasil pelatihan model kami:
                
    **Grafik Hasil Training**
    """, unsafe_allow_html=True)

    st.image("Image/Graph Plot.png", use_container_width=True)

    st.markdown("""
    **Confusion Matrix Model**
    """)

    st.image("Image/Confusion Matrix.png", use_container_width=True)


#Prediction Page
if(app_mode=="Deteksi Penyakit"):
    st.header("Deteksi Penyakit")
    st.markdown("""
    <p style="text-align: justify;">
    <b>Catatan</b>: Agar bisa mendapatkan hasil yang optimal, harap untuk mengupload gambar <b>daun tanaman</b>, <b>BUKAN</b> bagian tanaman lainnya atau seluruh tanaman! Diharapkan juga untuk mengupload gambar yang jelas, output bisa berbeda jika gambar yang diupload buram atau tidak jelas!
    </p>
""", unsafe_allow_html=True)
    test_image = st.file_uploader("Upload Gambar:")
    if(st.button("Tunjukkan Gambar")):
        st.image(test_image,width=4,use_container_width=True)
    #Predict button
    if(st.button("Deteksi")):
        st.write("Prediksi:")
        result_index = model_prediction(test_image)
        #Reading Labels
        #class_name = ['Disease', 'Healthy']
        if class_names[result_index] == 'Healthy':
            st.success("Tanaman Anda Terlihat Sehat!")
        else:
            st.error("Terdeteksi Gejala Penyakit Pada Daun Anda!")
            st.info("Menghasilkan rekomendasi perawatan...")

            # Gunakan prompt ke Generative AI
            prompt = "Tanaman aquascape anubias barteri saya terdeteksi memiliki gejala penyakit pada daunnya, seperti berwarna kuning dan terdapat pola bintik kekuningan. Tolong jelaskan dan jabarkan apa sekiranya penyebab dari hal tersebut dan langkah apa yang sebaiknya saya lakukan?"
            ai_response = get_recommendation(prompt)

            st.markdown("### 💡 Saran dari AI:")
            st.write(ai_response)
    
