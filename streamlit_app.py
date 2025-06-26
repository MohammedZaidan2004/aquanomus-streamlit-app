import tensorflow as tf
import numpy as np
import streamlit as st
import json


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
    st.header("Selamat Datang di Aplikasi Web Deteksi Penyakit Tanaman by Aquanomus! 🌿🔍")
    image_path = "Image/Streamlit Homepage.jpg"
    st.image(image_path,use_container_width=True)
    st.markdown("""
### Tentang Kami

<p style="text-align: justify;">
<b>Aquanomous</b> membantu para hobbyist dan pemilik aquascape dalam memonitoring kesehatan tanaman di akuarium. Tanaman merupakan komponen terpenting dalam menciptakan lingkungan aquascape, oleh karena itu Aquanomus menawarkan fitur pemantauan parameter akuarium via aplikasi IoT <b>Blynk</b> dan deteksi penyakit pada daun tanaman melalui aplikasi web ini agar memastikan lingkungan aquascape tetap terjaga dan tanaman dapat tumbuh dengan sehat.
</p>

<b>Aplikasi Blynk</b>
""", unsafe_allow_html=True)

    st.image("Image/blynk.png", use_container_width=True)

    st.markdown("""          
    ### Cara Kerja Aplikasi
    1. Foto terlebih dahulu daun tanaman yang ingin dianalisis dan simpan foto di file lokal perangkat.
    2. Akses aplikasi website deteksi penyakit Aquanomus via link di aplikasi Blynk.
    3. Pada aplikasi web, pergi ke menu **Dashboard**, lalu pilih sidebar **Deteksi Penyakit**.
    4. Pada bagian bawah halaman **Deteksi Penyakit** tekan ikon **Upload Gambar** untuk mengupload gambar/foto daun tanaman anda yang ingin dianalisis.
    5. **Hasil** deteksi akan langsung keluar setelah gambar terupload.
    
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
    Dalam melatih model, kami menggunakan dataset yang terdiri atas gambar-gambar daun tanaman anubias sehat dan yang terdapat gejala penyakit. Dataset ini kami buat sendiri menggunakan tanaman anubias yang kami miliki. Dari dataset tersebut, kami membaginya menjadi dua kategori, yaitu tanaman sehat dan tanaman sakit. Berikut konfigurasi dataset yang kami gunakan:               
    
    1. Train (6082 gambar).
    2. Validation (2691 gambar).
    3. Test (22 gambar).
                
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
        if class_names [result_index] == 'Healthy':
            st.success("Tanaman Anda Terlihat Sehat!")
        else:
            st.error("Terdeteksi Gejala Penyakit Pada Daun Anda! Anda Bisa Mencoba Melakukan Langkah Berikut: \n 1. Penyesuaian Ulang Parameter \n 2. Potong Daun Yang Terdeteksi Sakit \n 3. Periksa Rizhoma Tanaman")
    
