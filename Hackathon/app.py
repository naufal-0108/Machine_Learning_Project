import os, logging
import streamlit as st
import base64
import time
import pymongo as pm
import streamlit_scrollable_textbox as stx
from together import Together
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[
                        logging.StreamHandler()
                    ])

st.set_page_config(page_title="Roadmap Maker App", layout="centered")


@st.cache_resource
def load_mongodb():
    db_username = st.secrets["mongo"]["username"]
    db_password = st.secrets["mongo"]["password"]
    uri = f"mongodb+srv://{db_username}:{db_password}@st-1.ousywpa.mongodb.net/?retryWrites=true&w=majority&appName=st-1"
    client = MongoClient(uri, server_api=ServerApi('1'))

    try:
        client.admin.command('ping')
        logging.info("Pinged your deployment. You successfully connected to MongoDB!")
    
    except Exception as e:
        logging.error(e)

    db = client['ed-ind']
    cursor = db.summary
    return cursor

@st.cache_resource
def agent_init(model: str, temperature: float, max_tokens: int):

    api_key = st.secrets["together"]["api_key"]
    try:
        client = Together(api_key=api_key)
        logging.info("Connected to Together API")

    except Exception as e:
        logging.error("Failed to connect to Together API")
        logging.error(e)
        st.error("Failed to connect to Together API. Please check your API key.")
    
    llm_params = {"model": model, "temperature": temperature, "max_tokens": max_tokens}

    return client, llm_params

def agent_1(_client, kelas: str, mapel: str, topik: str, topik_details:str, style: str, style_details:str, waktu: str, pertemuan: str, llm_params: dict, messages: list):

    teacher_agent_1_prompt = f"""
    You are a highly experienced senior teacher who specializes in designing effective and engaging teaching strategies for Indonesian students.
    Your job is to assist teachers in Indonesia by crafting a **comprehensive teaching roadmap** tailored to their class context and students’s learning styles.

    The roadmap must help the teacher deliver the topic efficiently and enable the students to grasp the material thoroughly.
    You will receive these inputs:

    - **Kelas** (Education level of the students): 
        {kelas}

    - **Mata Pelajaran** (Subject being taught):
        {mapel}

    - **Topik Pembelajaran** (Specific topic within the subject):
        {topik}

    - **Topik Details** (Details of the topic (optional)):
        {topik_details}

    - **Gaya Belajar** (Learning Style):
        {style}

    - **Gaya Belajar Details** (Details of learning style (optional)):
        {style_details}

    - **Waktu Belajar** (Total available teaching time in one day):
        {waktu}

    - **Pertemuan** (Number of days):
        {pertemuan}

    Based on these inputs, generate a roadmap in **Bahasa Indonesia** with the following structure:

    ### Output Format (Write in Bahasa Indonesia):

    ```
    **Kriteria Pengajaran**
    <one_space>
    - Kelas           : ...
    - Mata Pelajaran  : ...
    - Topik Pelajaran : ...
    - Gaya Pengajaran : ...
    - Waktu Belajar   : ...
    - Pertemuan       : ...
    <one_space>
    **Objektif Capaian**
    <one_space>
    (Tuliskan minimal 5 tujuan pembelajaran yang jelas, spesifik, dan terukur.)
    <one_space>
    **Road Map Pengajaran**
    <one_space>
    (Rincikan langkah-langkah pengajaran yang disesuaikan dengan konteks dari inputs yang diberikan, terdiri dari beberapa pertemuan jika perlu. Setiap pertemuan minimal 5 poin langkah atau aktivitas.)
    <one_space>
    **Aktivitas Refleksi**
    <one_space>
    (Berikan ide aktivitas untuk membantu siswa merefleksikan apa yang mereka pelajari, misalnya: jurnal belajar, diskusi kelompok, exit ticket, dll.)
    <one_space>
    **Penilaian Formatif**
    <one_space>
    (Sertakan metode evaluasi selama proses belajar, seperti kuis singkat, observasi, pertanyaan terbuka, lembar kerja, dll.)
    <one_space>
    **Ide Kuis / Tes Singkat**
    <one_space>
    (Tampilkan 3–5 contoh pertanyaan kuis yang bisa digunakan guru untuk mengukur pemahaman siswa.)
    <one_space>
    **Saran Media & Alat Bantu**
    <one_space>
    (Rekomendasikan media pembelajaran atau alat bantu yang sesuai dengan gaya belajar siswa yang telah diberikan pada inputs.)
    <one_space>
    **Catatan Tambahan untuk Guru**
    <one_space>
    (Berikan tips tambahan atau hal-hal yang perlu diantisipasi saat mengajar topik ini.)
    ```

    Gunakan bahasa Indonesia yang jelas, alami, dan mudah dipahami oleh guru. Tulis seolah-olah Anda sedang membimbing guru baru untuk mengajar dengan percaya diri dan efektif.
    """
    messages += [{"role": "user", "content": teacher_agent_1_prompt}]
    

    response = _client.chat.completions.create(model=llm_params["model"], max_tokens=llm_params["max_tokens"],
                                                temperature=llm_params["temperature"], messages=messages)
    return response.choices[0].message.content


def agent_2(_client, previous_roadmap:str, feedback: str, llm_params: dict, messages: list):

    teacher_agent_2_prompt = f"""
Anda adalah seorang guru senior yang sangat berpengalaman, dengan keahlian khusus dalam mengevaluasi dan menyempurnakan rancangan pembelajaran (roadmap) untuk memaksimalkan efektivitas dan keterlibatan siswa di Indonesia.
Tugas Anda adalah untuk merevisi dan meningkatkan **Rancangan Pembelajaran Sebelumnya** berdasarkan **Umpan Balik** spesifik yang diberikan oleh guru pengguna.

**Input yang Akan Anda Terima:**

1.  **Rancangan Pembelajaran Sebelumnya (previous_roadmap):**
    Ini adalah rancangan pembelajaran awal yang telah dibuat dan memerlukan penyesuaian.
    ```
    {previous_roadmap}
    ```

2.  **Umpan Balik (feedback):**
    Ini adalah masukan, kritik, atau saran konkret dari guru pengguna terkait `previous_roadmap`.
    ```
    {feedback}
    ```

**Instruksi Utama untuk Penyempurnaan Rancangan:**

1.  **Analisis Mendalam:** Pelajari `previous_roadmap` dan `feedback` secara saksama. Identifikasi dengan tepat bagian-bagian dari `previous_roadmap` yang disorot atau perlu disesuaikan berdasarkan setiap poin dalam `feedback`.

2.  **Modifikasi Terarah dan Presisi:**
    * Lakukan perubahan pada `previous_roadmap` secara spesifik untuk merespons `feedback`. Jenis modifikasi dapat mencakup (namun tidak terbatas pada):
        * **Penambahan:** Memasukkan detail, contoh, aktivitas, atau sumber daya baru yang relevan seperti yang disarankan atau tersirat dalam `feedback`.
        * **Penghapusan:** Menghilangkan elemen yang dianggap tidak perlu, redundan, kurang efektif, atau dikritik dalam `feedback`.
        * **Modifikasi/Perbaikan Kata-kata:** Menyempurnakan pilihan kata, struktur kalimat, atau penjelasan untuk meningkatkan kejelasan, keterbacaan, atau kesesuaian nada, tanpa mengubah substansi inti yang tidak dikritik.
        * **Penyesuaian Konten:** Mengubah aspek tertentu dari langkah pembelajaran, metode penilaian, aktivitas refleksi, atau saran media agar lebih selaras dengan `feedback`.

3.  **Prinsip Utama Modifikasi:**
    * **Pertahankan Integritas Inti:** Modifikasi yang Anda lakukan **tidak boleh merombak total** struktur utama atau konten esensial dari `previous_roadmap` yang tidak secara eksplisit atau implisit menjadi target `feedback`.
    * **Fokus pada Umpan Balik:** Prioritaskan perubahan yang secara langsung menjawab poin-poin dalam `feedback`.
    * **Koherensi:** Pastikan bahwa setelah modifikasi, keseluruhan rancangan pembelajaran tetap logis, koheren, dan semua bagiannya saling mendukung.

4.  **Format Output (dalam Bahasa Indonesia):**
    * Hasil akhir harus berupa hanya konten rancangan roadmap yang telah direvisi secara lengkap (**tanpa adanya penambahan kalimat awal seperti "Rancangan Pembelajaran yang Telah Direvisi" dan diakhir!**).
    * **Gunakan format dan struktur yang sama persis** seperti yang ada pada `previous_roadmap`. Jika `previous_roadmap` memiliki bagian seperti "Kriteria Pengajaran", "Objektif Capaian", "Road Map Pengajaran", dll., maka output Anda juga harus mempertahankan bagian-bagian tersebut dengan konten yang sudah diperbarui sesuai hasil revisi.

Tujuan Anda adalah menghasilkan versi rancangan pembelajaran yang disempurnakan, lebih efektif, dan lebih memenuhi ekspektasi guru pengguna berdasarkan umpan baliknya, sambil tetap menghargai dan mempertahankan kerangka kerja serta konten positif dari rancangan sebelumnya.
Sajikan seluruh output dalam **Bahasa Indonesia** yang profesional dan jelas.
"""
    
    messages += [{"role": "user", "content": teacher_agent_2_prompt}]
    
    response = _client.chat.completions.create(model=llm_params["model"], max_tokens=llm_params["max_tokens"],
                                                temperature=llm_params["temperature"], messages=messages)
    
    return response.choices[0].message.content


@st.cache_data
def fetch_data(_cursor: pm.collection.Collection, kelas: str, mapel: str, topik: str):
    data = _cursor.find_one({"kelas": kelas, "mapel": mapel, "bab": topik}, {"_id":0, "ringkasan": 1})
    return data


def kelas_callback():
    st.session_state.kelas = st.session_state.kelas_select
    st.session_state.kelas_update = st.session_state.kelas_select
    st.session_state.topik = None
    st.session_state.style = None
    st.session_state.waktu = None
    st.session_state.pertemuan = None

def mapel_callback():
    st.session_state.mapel = st.session_state.mapel_select
    st.session_state.mapel_update = st.session_state.mapel_select
    st.session_state.topik = None
    st.session_state.style = None
    st.session_state.waktu = None
    st.session_state.pertemuan = None

def topik_toggle_callback():
    st.session_state.topik_toggle_update_state = st.session_state.topik_toggle
    st.session_state.topik = None
    st.session_state.style = None
    st.session_state.waktu = None
    st.session_state.pertemuan = None

def topik_callback():
    if st.session_state.topik_toggle:
        st.session_state.topik = st.session_state.topik_textarea
        st.session_state.topik_update = st.session_state.topik_textarea
    else:
        st.session_state.topik = st.session_state.topik_select
        st.session_state.topik_update = st.session_state.topik_select

def style_callback():
    st.session_state.style = st.session_state.style_select
    st.session_state.style_update = st.session_state.style_select

def waktu_callback():
    st.session_state.waktu = st.session_state.waktu_select
    st.session_state.waktu_update = st.session_state.waktu_select

def pertemuan_callback():
    st.session_state.pertemuan = st.session_state.pertemuan_select
    st.session_state.pertemuan_update = st.session_state.pertemuan_select


def kelas_update_callback():
    st.session_state.kelas_update = st.session_state.kelas_select_update
    st.session_state.topik_update = None

def mapel_update_callback():
    st.session_state.mapel_update = st.session_state.mapel_select_update
    st.session_state.topik_update = None

def topik_toggle_update_callback():
    st.session_state.topik_toggle_update_state = st.session_state.topik_toggle_update
    st.session_state.topik_update = None

def topik_update_callback():
    if st.session_state.topik_toggle_update_state:
        st.session_state.topik_update = st.session_state.topik_textarea_update
    else:
        st.session_state.topik_update = st.session_state.topik_select_update

def style_update_callback():
    st.session_state.style_update = st.session_state.style_select_update

def waktu_update_callback():
    st.session_state.waktu_update = st.session_state.waktu_select_update

def pertemuan_update_callback():
    st.session_state.pertemuan_update = st.session_state.pertemuan_select_update

def generate_roadmap_callback():
    st.session_state.gen = True
    st.session_state.gen_roadmap = True
    st.session_state.state_gen = "first"

def lock_callback():
    cond = all([st.session_state.kelas_update, st.session_state.mapel_update, st.session_state.topik_update,
                st.session_state.style_update, st.session_state.waktu_update, st.session_state.pertemuan_update])
    
    if not cond:
        st.session_state.lock = False
        st.info('Harap lengkapi semua variabel!', icon="ℹ️")
    
    else:
        st.session_state.lock = True
        st.session_state.generate_roadmap = True

def reset_callback():
    # Clear specific session state keys
    keys_to_clear = st.session_state.keys()
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state.generate_roadmap = False
    st.session_state.topik_choosen = False
    st.session_state.lock = False
    st.session_state.gen = False

@st.fragment
def roadmap_fragment():

    if "topic_details" not in st.session_state:
        st.session_state.topik_details = None

    if "roadmap_text" not in st.session_state:
        st.session_state.roadmap_text = None

    if "rev" not in st.session_state:
        st.session_state.rev = False
    
    if "save" not in st.session_state:
        st.session_state.save = False

    if "author" not in st.session_state:
        st.session_state.author = None

    if "title" not in st.session_state:
        st.session_state.title = None

    def rev_callback():
        st.session_state.gen_roadmap = False
        st.session_state.save = False
        st.session_state.rev = True

    def save_callback():
        st.session_state.save = True
        st.session_state.gen_roadmap = False
        st.session_state.rev = False
    
    def rev_confirm_callback():
        st.session_state.gen_roadmap = True

    def author_callback():
        st.session_state.author = st.session_state.author_text_input
    
    def title_callback():
        st.session_state.title = st.session_state.title_text_input
    
    @st.dialog("Konfirmasi download")
    def download_file():
        st.text_input(label="Author", value=None if st.session_state.author is None else st.session_state.author,
                        key="author_text_input",
                        placeholder="Masukkan nama Anda",
                        on_change=author_callback)
        
        st.text_input(label="Judul", value=None if st.session_state.title is None else st.session_state.title,
                        key="title_text_input",
                        placeholder="Masukkan judul file",
                        on_change=title_callback)
        
        if st.session_state.author and st.session_state.title:

            file_name = f"{st.session_state.author}_{st.session_state.title}.txt"

            if st.download_button(label="download",
                                mime='application/octet-stream',
                                file_name=file_name,
                                data=st.session_state.roadmap_text):
                
                st.session_state.save = False
                st.session_state.gen_roadmap = False
                st.session_state.rev = False
                st.rerun()
            
        else:
            st.download_button(label="download", data=st.session_state.roadmap_text,
                               disabled=True)

    with st.container(border=True, key="roadmap_container"):
        if st.session_state.state_gen == "first":

            if st.session_state.gen_roadmap:
                    
                    if st.session_state.topik_toggle_update_state:
                        kelas = st.session_state.kelas_update
                        mapel = st.session_state.mapel_update
                        topik = st.session_state.topik_update
                        style = st.session_state.style_update
                        style_details = dict_style.get(st.session_state.style_update, None)["Deskripsi"]
                        waktu = st.session_state.waktu_update
                        pertemuan = st.session_state.pertemuan_update
                        
                        start = 0

                        progress_bar = st.progress(start, text="Generating Roadmap...")

                        time.sleep(1)
                        progress_text = "Creating Prompt..."
                        start += 25
                        progress_bar.progress(start, text=progress_text)

                        time.sleep(1)
                        start += 50
                        progress_text = "Creating Roadmap..."
                        progress_bar.progress(start, text=progress_text)
                        assert all([kelas, mapel, topik, style, style_details, waktu, pertemuan]), "All variables must be filled"
                        roadmap = agent_1(llm_client, kelas, mapel, topik, style, style_details, waktu, pertemuan, llm_params, topik_details="None", messages=[])
                        st.session_state.state_gen = "second"
                        st.session_state.roadmap_text = roadmap

                        progress_text = "Finished!"
                        progress_bar.progress(100, text=progress_text)
                        progress_bar.empty()
                        st.divider()
                        st.markdown('<div class="roadmap-text">Roadmap Pembelajaran</div>', unsafe_allow_html=True)
                        st.divider()
                        st.write(st.session_state.roadmap_text)

                        cols = st.columns(2)

                        cols[0].button("Save", use_container_width=True, key="save_button", on_click=save_callback)
                        cols[1].button("Regenerate", use_container_width=True, key="rev_button", on_click=rev_callback)


                    else:

                        kelas = st.session_state.kelas_update
                        mapel = st.session_state.mapel_update
                        topik = st.session_state.topik_update
                        style = st.session_state.style_update
                        style_details = dict_style.get(st.session_state.style_update, None)["Deskripsi"]
                        waktu = st.session_state.waktu_update
                        pertemuan = st.session_state.pertemuan_update

                        start = 0
                        progress_bar = st.progress(start, text="Generating Roadmap...")

                        time.sleep(1)
                        progress_text = "Fetching Topic Details..."
                        start += 15
                        data = fetch_data(cursor, st.session_state.kelas, st.session_state.mapel, st.session_state.topik)
                        st.session_state.topik_details = data['ringkasan']
                        progress_bar.progress(start, text=progress_text)

                        time.sleep(1)
                        start += 15
                        progress_text = "Creating Prompt..."
                        progress_bar.progress(start, text=progress_text)

                        time.sleep(1)
                        progress_text = "Creating Roadmap..."
                        start += 45
                        progress_bar.progress(start, text=progress_text)
                        assert all([kelas, mapel, topik, style, style_details, waktu, pertemuan, st.session_state.topik_details]), "All variables must be filled"
                        roadmap = agent_1(llm_client, kelas, mapel, topik, st.session_state.topik_details, style, style_details, waktu, pertemuan, llm_params, messages=[])
                        st.session_state.state_gen = "second"
                        st.session_state.roadmap_text = roadmap

                        progress_text = "Finished!"
                        progress_bar.progress(100, text=progress_text)
                        progress_bar.empty()

                        st.divider()
                        st.markdown('<div class="roadmap-text">Roadmap Pembelajaran</div>', unsafe_allow_html=True)
                        st.divider()

                        st.write(st.session_state.roadmap_text)

                        cols = st.columns(2)

                        cols[0].button("Save", use_container_width=True, key="save_button", on_click=save_callback)
                        cols[1].button("Regenerate", use_container_width=True, key="rev_button", on_click=rev_callback)
 
        elif st.session_state.state_gen == "second":
            if st.session_state.gen_roadmap:
                if st.session_state.rev_toggle:

                    start = 0

                    progress_bar = st.progress(start, text="Generating Roadmap...")

                    time.sleep(1)
                    progress_text = "Creating Prompt..."
                    start += 25
                    progress_bar.progress(start, text=progress_text)

                    time.sleep(1)
                    start += 50
                    progress_text = "Creating Roadmap..."
                    progress_bar.progress(start, text=progress_text)
                    roadmap = agent_2(llm_client, previous_roadmap=st.session_state.roadmap_text, feedback=st.session_state.rev_comment, llm_params=llm_params, messages=[])
                    st.session_state.roadmap_text = roadmap

                    progress_text = "Finished!"
                    progress_bar.progress(100, text=progress_text)
                    progress_bar.empty()
                    
                    with st.container(border=True, key="roadmap_header_container"):
                        st.markdown('<div class="roadmap-text">Roadmap Pembelajaran</div>', unsafe_allow_html=True)
                    
                    st.write(st.session_state.roadmap_text)

                    cols = st.columns(2)

                    cols[0].button("Save", use_container_width=True, key="save_button", on_click=save_callback)
                    cols[1].button("Regenerate", use_container_width=True, key="rev_button", on_click=rev_callback)

                else:

                    if st.session_state.topik_toggle_update_state:

                        kelas = st.session_state.kelas_update
                        mapel = st.session_state.mapel_update
                        topik = st.session_state.topik_update
                        style = st.session_state.style_update
                        style_details = dict_style.get(st.session_state.style_update, None)["Deskripsi"]
                        waktu = st.session_state.waktu_update
                        pertemuan = st.session_state.pertemuan_update
                        
                        start = 0

                        progress_bar = st.progress(start, text="Generating Roadmap...")

                        time.sleep(1)
                        progress_text = "Creating Prompt..."
                        start += 25
                        progress_bar.progress(start, text=progress_text)

                        time.sleep(1)
                        start += 50
                        progress_text = "Creating Roadmap..."
                        progress_bar.progress(start, text=progress_text)
                        assert all([kelas, mapel, topik, style, style_details, waktu, pertemuan]), "All variables must be filled"
                        roadmap = agent_1(llm_client, kelas, mapel, topik, style, style_details, waktu, pertemuan, llm_params, topik_details="None", messages=[])
                        st.session_state.roadmap_text = roadmap

                        progress_text = "Finished!"
                        progress_bar.progress(100, text=progress_text)
                        progress_bar.empty()
                        st.divider()
                        st.markdown('<div class="roadmap-text">Roadmap Pembelajaran</div>', unsafe_allow_html=True)
                        st.divider()
                        st.write(st.session_state.roadmap_text)

                        cols = st.columns(2)

                        cols[0].button("Save", use_container_width=True, key="save_button", on_click=save_callback)
                        cols[1].button("Regenerate", use_container_width=True, key="rev_button", on_click=rev_callback)

                    else:

                        kelas = st.session_state.kelas_update
                        mapel = st.session_state.mapel_update
                        topik = st.session_state.topik_update
                        style = st.session_state.style_update
                        style_details = dict_style.get(st.session_state.style_update, None)["Deskripsi"]
                        waktu = st.session_state.waktu_update
                        pertemuan = st.session_state.pertemuan_update

                        start = 0
                        progress_bar = st.progress(start, text="Generating Roadmap...")

                        time.sleep(1)
                        progress_text = "Fetching Topic Details..."
                        start += 15
                        data = fetch_data(cursor, st.session_state.kelas, st.session_state.mapel, st.session_state.topik)
                        st.session_state.topik_details = data['ringkasan']
                        progress_bar.progress(start, text=progress_text)

                        time.sleep(1)
                        start += 15
                        progress_text = "Creating Prompt..."
                        progress_bar.progress(start, text=progress_text)

                        time.sleep(1)
                        progress_text = "Creating Roadmap..."
                        start += 45
                        progress_bar.progress(start, text=progress_text)

                        assert all([kelas, mapel, topik, style, style_details, waktu, pertemuan, st.session_state.topik_details]), "All variables must be filled"
                        roadmap = agent_1(llm_client, kelas, mapel, topik, st.session_state.topik_details, style, style_details, waktu, pertemuan, llm_params, messages=[])
                        st.session_state.roadmap_text = roadmap

                        progress_text = "Finished!"
                        progress_bar.progress(100, text=progress_text)
                        progress_bar.empty()

                        st.divider()
                        st.markdown('<div class="roadmap-text">Roadmap Pembelajaran</div>', unsafe_allow_html=True)
                        st.divider()

                        st.write(st.session_state.roadmap_text)

                        cols = st.columns(2)

                        cols[0].button("Save",use_container_width=True, key="save_button", on_click=save_callback)
                        cols[1].button("Regenerate", use_container_width=True, key="rev_button", on_click=rev_callback)

            elif st.session_state.rev:
                
                st.divider()
                st.markdown('<div class="roadmap-text">Roadmap Pembelajaran</div>', unsafe_allow_html=True)
                st.divider()
                st.write(st.session_state.roadmap_text)

                rev_toggle = st.toggle(
                    "Komentar Tambahan?",
                    key="rev_toggle",
                )

                if rev_toggle:

                    st.text_area("Masukkan komentar yang anda sampaikan:",
                                height=200,
                                placeholder="Contoh: Modifikasikan roadmap menjadi lebih detail!",
                                key="rev_comment")
                    
                    if not st.session_state.rev_comment:
                        st.button("Confirm", use_container_width=True, key="rev_confirm", on_click=rev_confirm_callback, disabled=True)
                            

                    else:
                        st.button("Confirm", use_container_width=True, key="rev_confirm", on_click=rev_confirm_callback)

                else:
                    st.button("Confirm", use_container_width=True, key="rev_confirm", on_click=rev_confirm_callback)

            elif st.session_state.save:
                download_file()
                        
            else:
                
                st.divider()
                st.markdown('<div class="roadmap-text">Roadmap Pembelajaran</div>', unsafe_allow_html=True)
                st.divider()
                st.write(st.session_state.roadmap_text)

                cols = st.columns(2)

                cols[0].button("Save", use_container_width=True, key="save_button", on_click=save_callback)
                cols[1].button("Regenerate", use_container_width=True, key="rev_button", on_click=rev_callback)

cursor = load_mongodb()
llm_client, llm_params = agent_init(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", temperature=1., max_tokens=3256)

dict_options = {
    "7": {
        "IPA": ["Hakikat Ilmu Sains dan Metode Ilmiah", "Zat dan Perubahannya", "Suhu, Kalor, dan Pemuaian", "Gerak dan Gaya",
        "Klasifikasi Makhluk Hidup","Ekologi dan Keanekaragaman Hayati Indonesia","Bumi dan Tata Surya"],
        "IPS": ["Kehidupan Sosial dan Kondisi Lingkungan Sekitar", "Keberagaman Lingkungan Sekitar", "Potensi Ekonomi Lingkungan", "Pemberdayaan Masyarakat"],
        "Matematika": ["Bilangan Bulat", "Aljabar Dasar", "Persamaan Linear", "Perbandingan Senilai dan Berbalik Nilai", "Bangun Datar", "Bangun Ruang", "Penggunaan Data"]
    },
    "8": {
        "IPA": ["Pengenalan Sel", "Struktur dan Fungsi Makhluk Hidup", "Usaha, Energi dan Pesawat Sederhana", "Getaran, Gelombang dan Cahaya", "Unsur, Senyawa dan Campuran", "Struktur Bumi dan Perkembangannya"],
        "IPS": ['Kondisi Geografis dan Pelestarian Sumber Daya Alam', "Kemajemukan Masyarakat Indonesia", "Nasionalisme dan Jati Diri Bangsa", "Pembangunan Perekonomian Indonesia"],
        "Matematika": ["Penyederhanaan Bentuk Aljabar", "Sistem Persamaan Linear Dua Variabel", "Fungsi Linear", "Sifat-Sifat Bangun Geometri", "Segitiga dan Segi Empat", "Peluang Dasar"]
    },
    "9": {
        "IPA": ["Pertumbuhan dan Perkembangan", "Sistem Koordinasi, Reproduksi dan Homeostatis Manusia", "Tekanan", "Listrik, Magnet dan Sumber Energi Alternatif", "Reaksi-Reaksi Kimia dan Dinamikanya",
       "Pewarisan Sifat dan Bioteknologi", "Isu-Isu Lingkungan"],
        "IPS": ["Manusia dan Perubahan", "Perkembangan Ekonomi Digital", "Tantangan Pembangunan Indonesia", "Tujuan Pembangunan Berkelanjutan (SDG)"],
        "Matematika": ["Sistem Persamaan Linear Dua Variabel", "Bangun Ruang", "Transformasi Geometri", "Peluang dan Pemilihan Sampel"]
    }
}




kelas_str = """
Tingkat atau jenjang kelas tempat siswa berada, seperti Kelas 6 SD, Kelas 9 SMP, atau Kelas 12 SMA. Informasi ini penting agar materi ajar sesuai dengan tingkat pemahaman dan perkembangan kognitif siswa di kelas tersebut.
"""

mapel_str = """
Bidang studi atau pelajaran yang akan diajarkan, misalnya Matematika, Bahasa Indonesia, IPA, IPS, atau Bahasa Inggris. Setiap mata pelajaran memiliki kurikulum dan kompetensi dasar yang berbeda, sehingga roadmap akan disesuaikan berdasarkan mata pelajaran yang dipilih.
"""

topik_str = """
Subtopik atau materi spesifik dari mata pelajaran yang ingin diajarkan, misalnya “Pecahan” dalam Matematika atau “Sistem Pernapasan” dalam IPA. Pemilihan topik membantu AI menyusun langkah-langkah pembelajaran yang fokus dan mendalam sesuai kebutuhan siswa.
"""

style_str = """
Preferensi atau cara terbaik siswa menyerap informasi. Dengan mengenali gaya belajar siswa, AI akan membantu menyusun aktivitas dan metode ajar yang lebih efektif dan menyenangkan. Berikut adalah contoh dari gaya belajar yang ada:
***
**Direct Instruction (Instruksi Langsung)**\n
Berpusat pada guru; pelajaran disampaikan secara terstruktur dan sistematis, langkah demi langkah, dengan tujuan pembelajaran yang jelas.
Implementasi:
- Penjelasan materi secara tertulis dan sistematis
- Pemberian contoh sebelum siswa berlatih mandiri
- Penggunaan lembar kerja, latihan soal, dan kuis
- Guru memimpin dan mengarahkan seluruh proses belajar
***
**Inquiry-Based Learning (Pembelajaran Berbasis Penemuan)**\n
Berpusat pada siswa; siswa diajak untuk mengeksplorasi, bertanya, dan membangun pemahaman melalui proses investigasi.
Implementasi:
- Menggunakan pertanyaan terbuka yang memicu rasa ingin tahu
- Melibatkan aktivitas eksperimen atau eksplorasi
- Menggunakan metode Socratic Questioning (bertanya untuk memancing pemikiran kritis)
- Sedikit atau bahkan tanpa penjelasan langsung dari guru di awal
***
**Project-Based Learning (Pembelajaran Berbasis Proyek)**\n
Pembelajaran dilakukan melalui proyek nyata dan bermakna yang diselesaikan dalam jangka waktu tertentu, mendorong kolaborasi dan kreativitas.
Implementasi:
- Menyusun skenario masalah nyata yang relevan
- Menentukan peran dan hasil kerja kelompok atau individu
- Membuat timeline dan target capaian
- Menyediakan momen refleksi untuk mengevaluasi hasil dan proses belajar
***
**Flipped Classroom (Kelas Terbalik)**\n
Pembelajaran dilakukan secara mandiri di rumah (misalnya dengan menonton video), sedangkan waktu di kelas digunakan untuk diskusi, latihan, atau penerapan konsep.
Implementasi:
- Menugaskan video atau materi belajar untuk dipelajari di rumah
- Waktu kelas digunakan untuk memecahkan masalah atau latihan
- Siswa bisa mengajar atau menjelaskan kepada teman (peer instruction)
- Kegiatan kelas berfokus pada diskusi dan pertanyaan tinjauan
***
**Experiential Learning (Pembelajaran Berbasis Pengalaman)**\n
Belajar melalui pengalaman langsung, baik fisik maupun emosional, dengan penekanan pada aktivitas praktis dan refleksi.
Implementasi:
- Melibatkan simulasi, eksperimen, permainan peran, atau kunjungan lapangan
- Diskusi reflektif setelah aktivitas untuk memperdalam pemahaman
- Mendorong proses coba-coba sebagai bagian dari pembelajaran
- Aktivitas dirancang agar merangsang banyak indera siswa
"""

waktu_str = """
Durasi atau waktu yang tersedia dalam setiap sesi pengajaran, misalnya 30 menit, 60 menit, atau 90 menit. Waktu ini akan menentukan seberapa banyak materi yang dapat dibahas dan bagaimana pembagian aktivitas selama pertemuan berlangsung.
"""

pertemuan_str = """
Total sesi pengajaran yang tersedia untuk menyelesaikan topik atau tujuan pembelajaran, misalnya 4 kali pertemuan atau 8 sesi. Jumlah pertemuan ini akan digunakan untuk membagi materi secara bertahap agar lebih terstruktur dan mudah dipahami.
"""

dict_style = {
  "Direct Instruction Learning": {
    "Deskripsi": "Pendekatan pembelajaran yang berpusat pada guru, di mana materi pelajaran disampaikan secara sangat terstruktur dan bertahap dengan tujuan pembelajaran yang jelas. Guru mengendalikan kecepatan dan alur informasi. Implementasinya mencakup penjelasan yang eksplisit dan terperinci, seringkali secara lisan atau tertulis yang jelas; guru memberikan contoh dan memodelkan konsep sebelum siswa berlatih; penggunaan lembar kerja, kuis, dan latihan untuk memperkuat pembelajaran; serta guru memimpin seluruh proses belajar, membimbing siswa melalui materi."
  },
  "Inquiry-Based Learning": {
    "Deskripsi": "Pendekatan yang berpusat pada siswa, di mana peserta didik secara aktif mengeksplorasi pertanyaan, masalah, atau skenario untuk membangun pemahaman mereka sendiri, dengan penekanan pada proses penemuan dan berpikir kritis. Implementasinya melibatkan pengajuan pertanyaan terbuka yang merangsang keingintahuan dan investigasi; melibatkan siswa dalam eksperimen langsung, penelitian, atau kegiatan penemuan; menggunakan teknik pertanyaan Sokratik (mengajukan pertanyaan pemandu untuk merangsang pemikiran yang lebih dalam); serta penjelasan langsung yang minimal dari guru, karena siswa dibimbing untuk menemukan jawaban sendiri."
  },
  "Project-Based Learning": {
    "Deskripsi": "Pedagogi yang berpusat pada siswa di mana pembelajaran terjadi melalui keterlibatan dalam proyek-proyek dunia nyata yang kompleks dan berlangsung dalam jangka waktu tertentu, yang berpuncak pada produk, presentasi, atau kinerja nyata. Implementasinya meliputi perumusan skenario masalah yang autentik atau pertanyaan pendorong; penetapan peran dan hasil kerja kolaboratif untuk tugas kelompok; penyusunan garis waktu yang jelas dengan tonggak pencapaian dan target hasil; serta penggabungan pertanyaan dan kegiatan refleksi sepanjang proyek."
  },
  "Flipped Classroom Learning": {
    "Deskripsi": "Strategi instruksional di mana elemen kuliah tradisional dan pekerjaan rumah dibalik. Siswa pertama-tama mengakses konten instruksional (misalnya, video, bacaan) di luar kelas, dan waktu di kelas didedikasikan untuk pembelajaran aktif, diskusi, dan penerapan pengetahuan. Implementasinya melibatkan pemberian tugas pra-kelas seperti menonton video instruksional atau menyelesaikan daftar bacaan yang telah dikurasi; pemanfaatan waktu kelas untuk pemecahan masalah kolaboratif dan latihan penerapan; fasilitasi instruksi oleh teman sebaya dan tugas pembelajaran kolaboratif; serta penggunaan pertanyaan tinjauan aktif dan diskusi untuk memeriksa pemahaman dan memperdalam pembelajaran."
  },
  "Experiential Learning": {
    "Deskripsi": "Proses belajar melalui pengalaman langsung dan refleksi, dengan fokus pada partisipasi aktif dalam kegiatan konkret dan belajar dari konsekuensi tindakan tersebut. Implementasinya mencakup penggunaan simulasi, eksperimen, permainan peran, atau kunjungan lapangan; pelaksanaan sesi refleksi terstruktur atau diskusi setelah kegiatan; penekanan pada pendekatan coba-coba di mana kesalahan adalah peluang belajar; serta perancangan kegiatan yang melibatkan berbagai indra untuk pengalaman sensorik yang kaya."
  }
}

list_waktu = ["30 menit","45 menit","1 jam", "1,5 jam", "2 jam", "2,5 jam"]
list_pertemuan = ["1 kali", "2 kali", "3 kali", "4 kali"]


st.markdown("""
    <style>
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgb(0, 0, 0, 0);
    }
    .header-text {
        text-align: center;
        font-size: 2em;
        font-weight: bold;
    }
    .header-sidebar-text{
        text-align: left;
        font-size: 1.5em;
        font-weight: bold;  
            
    }
    .info-sidebar-text{
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;  
            
    }
    .subtext {
        font-size: 1.0em;
        text-align: justify;
    }
            
    .roadmap-text{
        text-align: center;
        font-size: 2em;
        font-weight: bold;
    }
    .subtext-form {
        font-size: 2.0 em;
        text-align: center;
        

    }
    </style>
""", unsafe_allow_html=True)


## STARTING POINT

if "kelas_update" not in st.session_state:
    st.session_state.kelas = None

if "mapel_update" not in st.session_state:
    st.session_state.mapel = None

if "topik_update" not in st.session_state:
    st.session_state.topik_update = None

if "style_update" not in st.session_state:
    st.session_state.style_update = None

if "waktu_update" not in st.session_state:
    st.session_state.waktu_update = None

if "pertemuan_update" not in st.session_state:
    st.session_state.pertemuan_update = None

if "kelas" not in st.session_state:
    st.session_state.kelas = None

if "mapel" not in st.session_state:
    st.session_state.mapel = None

if "topik" not in st.session_state:
    st.session_state.topik = None

if "style" not in st.session_state:
    st.session_state.style = None

if "waktu" not in st.session_state:
    st.session_state.waktu = None

if "pertemuan" not in st.session_state:
    st.session_state.pertemuan = None

if "generate_roadmap" not in st.session_state:
    st.session_state.generate_roadmap = False

if "topik_toggle_update_state" not in st.session_state:
    st.session_state.topik_toggle_update_state = False

if "lock" not in st.session_state:
    st.session_state.lock = False

if "gen" not in st.session_state:
    st.session_state.gen = False

if "state_gen" not in st.session_state:
    st.session_state.state_gen = "not_yet"

if "gen_roadmap" not in st.session_state:
    st.session_state.gen_roadmap = False

with st.sidebar:
    with st.container():
        st.markdown('<div class="header-sidebar-text">Selamat Datang di Aplikasi Roadmap Pembelajaran (EduPlanner)!</div>', unsafe_allow_html=True)
    
    st.divider()
    with st.container():
        st.markdown('<div class="subtext">Aplikasi ini bertujuan untuk membantu Anda dalam merencanakan dan mengelola rancangan pembelajaran untuk perserta didik.</div>', unsafe_allow_html=True)
    st.divider()

    with st.container():
        with st.expander("Kelas", icon=":material/school:"):
            st.write(kelas_str)

        with st.expander("Mata Pelajaran", icon=":material/book_2:"):
            st.write(mapel_str)

        with st.expander("Topik", icon=":material/public:"):
            st.write(topik_str)

        with st.expander("Gaya Belajar", icon=":material/co_present:"):
            st.write(style_str)

        with st.expander("Waktu Belajar", icon=":material/schedule:"):
            st.write(waktu_str)

        with st.expander("Pertemuan", icon=":material/calendar_month:"):
            st.write(pertemuan_str)

    


with st.container(border=False,key="header_container"):
    st.markdown('<div class="header-text">EduPlanner</div>', unsafe_allow_html=True)

with st.container(border=True, key="form_container"):
    cond = [st.session_state.kelas, st.session_state.mapel, st.session_state.topik, st.session_state.style, st.session_state.waktu, st.session_state.pertemuan]

    if not all(cond):
        kelas = st.selectbox("Pilih kelas:",
                    index=None if st.session_state.kelas is None else list(dict_options.keys()).index(st.session_state.kelas),
                    placeholder="Pilih Kelas...",
                    options=["7", "8", "9"],
                    key="kelas_select",
                    help="Infomasi tedapat di sidebar.",
                    on_change=kelas_callback)
        
        if st.session_state.kelas:
            
            mapel = st.selectbox("Pilih mata pelajaran:",
                                index=None if st.session_state.mapel is None else list(dict_options[st.session_state.kelas]).index(st.session_state.mapel),
                                placeholder="Pilih mata pelajaran...",
                                options=list(dict_options[st.session_state.kelas].keys()),
                                key="mapel_select",
                                help="Informasi terdapat di sidebar.",
                                on_change=mapel_callback)
            
            if st.session_state.mapel:

                topik_toggle = st.toggle(
                    "Kustomisasi topik pembelajaran sendiri?",
                    key="topik_toggle",
                    on_change=topik_toggle_callback
                )

                if topik_toggle:
                    topik = st.text_area(label="Masukkan topik pembelajaran yang Anda inginkan:",
                                        value=None if st.session_state.topik is None else st.session_state.topik,
                                        height=200,
                                        placeholder="Contoh: Zat dan Perubahannya",
                                        key="topik_textarea",
                                        help="Informasi terdapat di sidebar.",
                                        on_change=topik_callback)
                else:

                    topik = st.selectbox("Pilihan topik pembelajaran:",
                            index=None if st.session_state.topik is None else dict_options[st.session_state.kelas][st.session_state.mapel].index(st.session_state.topik),
                            placeholder="Pilih topik pembelajaran...",
                            options=list(dict_options[st.session_state.kelas][st.session_state.mapel]),
                            key="topik_select",
                            help="Informasi terdapat di sidebar.",
                            on_change=topik_callback)

                if st.session_state.topik:
                    style = st.selectbox("Pilih gaya belajar:",
                                index=None if st.session_state.style is None else list(dict_style.keys()).index(st.session_state.style),
                                placeholder="Pilih gaya belajar...",
                                options=list(dict_style.keys()),
                                key="style_select",
                                help="Informasi terdapat disidebar.",
                                on_change=style_callback)
                    
                    if st.session_state.style:
                        waktu = st.selectbox("Total waktu belajar per pertemuan:",
                                    index=None if st.session_state.waktu is None else list_waktu.index(st.session_state.waktu),
                                    placeholder="Pilih waktu belajar...",
                                    options=["30 menit","45 menit","1 jam", "1,5 jam", "2 jam", "2,5 jam"],
                                    key="waktu_select",
                                    help="Informasi terdapat di sidebar.",
                                    on_change=waktu_callback)

                        if st.session_state.waktu:
                            pertemuan = st.selectbox("Total pertemuan:",
                            index=None if st.session_state.pertemuan is None else list_pertemuan.index(st.session_state.pertemuan),
                            placeholder="Pilih jumlah pertemuan...",
                            options=["1 kali", "2 kali", "3 kali", "4 kali"],
                            key="pertemuan_select",
                            on_change=pertemuan_callback)
                        
                        else:
                                                
                            pertemuan = st.selectbox("Total pertemuan:",
                                        options=["Pilih waktu belajar terlebih dahulu..."],
                                        disabled=True,
                                        help="Informasi terdapat di sidebar.")

                    else:
                        waktu = st.selectbox("Total waktu belajar per pertemuan:",
                                    options=["Pilih gaya belajar terlebih Dahulu..."],
                                    disabled=True,
                                    help="Informasi terdapat di sidebar.")
                        
                        pertemuan = st.selectbox("Total pertemuan:",
                                    options=["Pilih waktu belajar terlebih dahulu..."],
                                    disabled=True,
                                    help="Informasi terdapat di sidebar.")

                else:
                    style = st.selectbox("Pilih gaya belajar:",
                                options=["Pilih topik terlebih Dahulu..."],
                                disabled=True,
                                help="Informasi terdapat di sidebar.")

                    waktu = st.selectbox("Total waktu belajar per pertemuan:",
                                options=["Pilih gaya belajar terlebih Dahulu..."],
                                disabled=True,
                                help="Informasi terdapat di sidebar.")
                    
                    pertemuan = st.selectbox("Total pertemuan:",
                                options=["Pilih waktu belajar terlebih dahulu..."],
                                disabled=True,
                                help="Informasi terdapat di sidebar.")
            
            else:

                topik_choosen = st.toggle(
                    "Kustomisasi topik pembelajaran sendiri?",
                    key="topik_choosen",
                    disabled=True,
                )

                topik = st.selectbox("Pilih topik pembelajaran:",
                        options=["Pilih mata pelajaran terlebih dahulu..."],
                        disabled=True,
                        help="Informasi terdapat di sidebar.")
                
                style = st.selectbox("Pilih gaya belajar:",
                            options=["Pilih topik terlebih Dahulu..."],
                            disabled=True,
                            help="Informasi terdapat di sidebar.")

                waktu = st.selectbox("Total waktu belajar per pertemuan:",
                            options=["Pilih gaya belajar terlebih Dahulu..."],
                            disabled=True,
                            help="Informasi terdapat di sidebar.")
                
                pertemuan = st.selectbox("Total pertemuan:",
                            options=["Pilih waktu belajar terlebih dahulu..."],
                            disabled=True,
                            help="Informasi terdapat di sidebar.")

        else:
            
            mapel = st.selectbox("Pilih mata pelajaran:",
                                options=["Pilih kelas terlebih dahulu..."],
                                disabled=True,
                                help="Informasi terdapat di sidebar.")
            
            topik_choosen = st.toggle(
                "Kustomisasi topik pembelajaran sendiri?",
                key="topik_choosen",
                disabled=True,
            )

            topik = st.selectbox("Pilih topik pembelajaran:",
                    options=["Pilih mata pelajaran terlebih dahulu..."],
                    disabled=True,
                    help="Informasi terdapat di sidebar.")
            
            style = st.selectbox("Pilih gaya belajar:",
                        options=["Pilih topik terlebih Dahulu..."],
                        disabled=True,
                        help="Informasi terdapat di sidebar.")

            waktu = st.selectbox("Total waktu belajar per pertemuan:",
                        options=["Pilih gaya belajar terlebih Dahulu..."],
                        disabled=True,
                        help="Informasi terdapat di sidebar.")
            
            pertemuan = st.selectbox("Total pertemuan:",
                        options=["Pilih waktu belajar terlebih dahulu..."],
                        disabled=True,
                        help="Informasi terdapat di sidebar.")

        cols = st.columns(3)
        lock = cols[1].button("Lock", key="lock_button", disabled=True, use_container_width=True, on_click=lock_callback)
        gen = cols[0].button("Generate Roadmap", disabled=True, use_container_width=True, on_click=generate_roadmap_callback)
        res = cols[2].button("Reset", on_click=reset_callback, use_container_width=True)
        st.markdown('<div class="subtext-form">Lengkapi form terlebih dahulu untuk mendapatkan roadmap pembelajaran</div>', unsafe_allow_html=True)


    else:
        if st.session_state.lock:

            kelas_updt = st.selectbox("Pilih Kelas:",
                                index=st.session_state.kelas if st.session_state.kelas_update is None else list(dict_options.keys()).index(st.session_state.kelas_update),
                                placeholder="Pilih kelas...",
                                key="kelas_select_update",
                                options=["7", "8", "9"],
                                on_change=kelas_update_callback,
                                disabled=True)
            
            
            mapel_updt = st.selectbox("Pilih mata pelajaran:",
                                placeholder="Pilih mata pelajaran...",
                                index=st.session_state.mapel if st.session_state.mapel_update is None else list(dict_options[st.session_state.kelas_update]).index(st.session_state.mapel_update),
                                options=list(dict_options[st.session_state.kelas].keys()),
                                key="mapel_select_update",
                                help="Informasi terdapat di sidebar.",
                                on_change=mapel_update_callback,
                                disabled=True)

            topik_choosen = st.toggle(value=st.session_state.topik_toggle_update_state,
                label="Kustomisasi topik pembelajaran sendiri?",
                key="topik_toggle_update",
                on_change=topik_toggle_update_callback,
                disabled=True
            )

            if st.session_state.topik_toggle_update_state:
                topik_updt = st.text_area(label="Masukkan topik pembelajaran yang Anda inginkan:",
                                          value=None if st.session_state.topik_update is None else st.session_state.topik_update,
                                          height=200,
                                          placeholder="Contoh: Zat dan Perubahannya",
                                          key="topik_textarea_update",
                                          help="Informasi terdapat di sidebar.",
                                          on_change=topik_update_callback,
                                          disabled=True)
                
            else:
                topik_updt = st.selectbox(label="Pilihan topik pembelajaran:",
                            index=None if st.session_state.topik_update is None else dict_options[st.session_state.kelas_update][st.session_state.mapel_update].index(st.session_state.topik_update),
                            placeholder="Pilih topik pembelajaran...",
                            options=list(dict_options[st.session_state.kelas_update][st.session_state.mapel_update]),
                            key="topik_select_update",
                            help="Informasi terdapat di sidebar.",
                            on_change=topik_update_callback,
                            disabled=True)
            

            style = st.selectbox("Pilih gaya belajar:",
                                index=st.session_state.style if st.session_state.style_update is None else list(dict_style.keys()).index(st.session_state.style_update),
                                placeholder="Pilih gaya belajar...",
                                options=list(dict_style.keys()),
                                key="style_select_update",
                                help="Informasi terdapat disidebar.",
                                on_change=style_update_callback,
                                disabled=True)
            
            waktu = st.selectbox("Total waktu belajar per pertemuan:",
                        index=st.session_state.waktu if st.session_state.waktu_update is None else list_waktu.index(st.session_state.waktu_update),
                        placeholder="Pilih waktu belajar...",
                        options=["30 menit","45 menit","1 jam", "1,5 jam", "2 jam", "2,5 jam"],
                        key="waktu_select_update",
                        help="Informasi terdapat di sidebar.",
                        on_change=waktu_update_callback,
                        disabled=True)
            
            pertemuan = st.selectbox("Total pertemuan:",
                        index=st.session_state.pertemuan if st.session_state.pertemuan_update is None else list_pertemuan.index(st.session_state.pertemuan_update),
                        placeholder="Pilih jumlah pertemuan...",
                        options=["1 kali", "2 kali", "3 kali", "4 kali"],
                        key="pertemuan_select_update",
                        help="Informasi terdapat di sidebar.",
                        on_change=pertemuan_update_callback,
                        disabled=True)


        else:
            kelas_updt = st.selectbox("Pilih Kelas:",
                                index=st.session_state.kelas if st.session_state.kelas_update is None else list(dict_options.keys()).index(st.session_state.kelas_update),
                                placeholder="Pilih kelas...",
                                key="kelas_select_update",
                                options=["7", "8", "9"],
                                on_change=kelas_update_callback)
            
            
            mapel_updt = st.selectbox("Pilih mata pelajaran:",
                                placeholder="Pilih mata pelajaran...",
                                index=st.session_state.mapel if st.session_state.mapel_update is None else list(dict_options[st.session_state.kelas_update]).index(st.session_state.mapel_update),
                                options=list(dict_options[st.session_state.kelas].keys()),
                                key="mapel_select_update",
                                help="Informasi terdapat di sidebar.",
                                on_change=mapel_update_callback)
            
            topik_choosen = st.toggle(value=st.session_state.topik_toggle_update_state,
                label="Kustomisasi topik pembelajaran sendiri?",
                key="topik_toggle_update",
                on_change=topik_toggle_update_callback
            )

            if st.session_state.topik_toggle_update_state:
                topik_updt = st.text_area(label="Masukkan topik pembelajaran yang Anda inginkan:",
                                          value=None if st.session_state.topik_update is None else st.session_state.topik_update,
                                          height=200,
                                          placeholder="Contoh: Zat dan Perubahannya",
                                          key="topik_textarea_update",
                                          help="Informasi terdapat di sidebar.",
                                          on_change=topik_update_callback)
                
            else:
                topik_updt = st.selectbox(label="Pilihan topik pembelajaran:",
                            index=None if st.session_state.topik_update is None else dict_options[st.session_state.kelas_update][st.session_state.mapel_update].index(st.session_state.topik_update),
                            placeholder="Pilih topik pembelajaran...",
                            options=list(dict_options[st.session_state.kelas_update][st.session_state.mapel_update]),
                            key="topik_select_update",
                            help="Informasi terdapat di sidebar.",
                            on_change=topik_update_callback)
            

            style = st.selectbox("Pilih gaya belajar:",
                                index=st.session_state.style if st.session_state.style_update is None else list(dict_style.keys()).index(st.session_state.style_update),
                                placeholder="Pilih gaya belajar...",
                                options=list(dict_style.keys()),
                                key="style_select_update",
                                help="Informasi terdapat disidebar.",
                                on_change=style_update_callback)
            
            waktu = st.selectbox("Total waktu belajar per pertemuan:",
                        index=st.session_state.waktu if st.session_state.waktu_update is None else list_waktu.index(st.session_state.waktu_update),
                        placeholder="Pilih waktu belajar...",
                        options=["30 menit","45 menit","1 jam", "1,5 jam", "2 jam", "2,5 jam"],
                        key="waktu_select_update",
                        help="Informasi terdapat di sidebar.",
                        on_change=waktu_update_callback)
            
            pertemuan = st.selectbox("Total pertemuan:",
                        index=st.session_state.pertemuan if st.session_state.pertemuan_update is None else list_pertemuan.index(st.session_state.pertemuan_update),
                        placeholder="Pilih jumlah pertemuan...",
                        options=["1 kali", "2 kali", "3 kali", "4 kali"],
                        key="pertemuan_select_update",
                        help="Informasi terdapat di sidebar.",
                        on_change=pertemuan_update_callback)
            

        cols = st.columns(3)

        if st.session_state.gen:
            lock = cols[1].button("Lock", key="lock_button", disabled=True, use_container_width=True, on_click=lock_callback)
            gen = cols[0].button("Generate Roadmap", disabled=True, use_container_width=True, on_click=generate_roadmap_callback)
            res = cols[2].button("Reset", on_click=reset_callback, use_container_width=True)
            with st.expander("Informasi yang telah dikunci"):
                st.write(f"**Kelas**: {st.session_state.kelas_update}")
                st.write(f"**Mata Pelajaran**: {st.session_state.mapel_update}")
                st.write(f"**Topik**: {st.session_state.topik_update}")
                st.write(f"**Gaya Belajar**: {st.session_state.style_update}")
                st.write(f"**Total Waktu Belajar**: {st.session_state.waktu_update}")
                st.write(f"**Total Pertemuan**: {st.session_state.pertemuan_update}")

        elif st.session_state.generate_roadmap:
            lock = cols[1].button("Lock", key="lock_button", disabled=True, use_container_width=True, on_click=lock_callback)
            gen = cols[0].button("Generate Roadmap", use_container_width=True, on_click=generate_roadmap_callback)
            res = cols[2].button("Reset", on_click=reset_callback, use_container_width=True)

            with st.expander("Informasi yang telah dikunci"):
                st.write(f"**Kelas**: {st.session_state.kelas_update}")
                st.write(f"**Mata Pelajaran**: {st.session_state.mapel_update}")
                st.write(f"**Topik**: {st.session_state.topik_update}")
                st.write(f"**Gaya Belajar**: {st.session_state.style_update}")
                st.write(f"**Total Waktu Belajar**: {st.session_state.waktu_update}")
                st.write(f"**Total Pertemuan**: {st.session_state.pertemuan_update}")

        else:
            lock = cols[1].button("Lock", key="lock_button", use_container_width=True, on_click=lock_callback)
            gen = cols[0].button("Generate Roadmap", disabled=True, use_container_width=True, on_click=generate_roadmap_callback)
            res = cols[2].button("Reset", on_click=reset_callback, use_container_width=True)
            st.markdown('<div class="subtext-form">Lengkapi form terlebih dahulu untuk mendapatkan roadmap pembelajaran</div>', unsafe_allow_html=True)

roadmap_fragment()