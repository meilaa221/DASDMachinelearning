# 13. Agile, DevOps/MLOps, dan Continuous Improvement

## A. Agile Development

Proyek ini dikerjakan oleh **kelompok beranggotakan 4 mahasiswa**, dengan metode **Kanban** di
board **ClickUp** (kolom *Backlog → In Progress → Review → Done*) — dipilih dibanding Scrum penuh
karena durasi pengerjaan tugas relatif singkat (minggu ke-10 s.d. 16) dan alur kerja proyek sains
data (EDA → preprocessing → modeling → dashboard → analisis keamanan/etika → laporan) lebih cocok
mengalir sebagai tugas-tugas yang bergerak terus-menerus antar anggota, dibanding dipecah ke dalam
beberapa sprint tetap dengan seremoni penuh (sprint planning, daily standup, sprint review terpisah)
yang overhead-nya kurang sepadan untuk tim sekecil ini.

Setiap kartu di ClickUp diberi **assignee** (anggota penanggung jawab) sehingga satu papan yang
sama dipakai bersama, bukan empat papan terpisah — ini yang membuatnya tetap disebut Kanban
*kelompok*, bukan Kanban personal per anggota.

Contoh pembagian kartu Kanban untuk proyek ini (ganti kolom "PIC" dengan nama anggota kelompok
yang sebenarnya bertanggung jawab):

| Kolom | Contoh Kartu | PIC (contoh peran) |
|---|---|---|
| Backlog | "Cari & validasi dataset", "Riset metrik evaluasi tambahan" | Anggota 1 (Data) |
| In Progress | "Bangun pipeline preprocessing (`src/preprocessing.py`)" | Anggota 2 (ML Engineer) |
| In Progress | "Desain halaman Visualisasi & Insight dashboard" | Anggota 3 (Frontend/Dashboard) |
| Review | "Cek ulang metrik evaluasi sebelum dipakai di dashboard" | Anggota 2 & Anggota 4 |
| Review | "Review analisis keamanan & etika data sebelum masuk laporan" | Anggota 4 (Security & Docs) |
| Done | "Dataset documentation", "Notebook EDA", "FastAPI dashboard app" | Seluruh anggota |

**Menjawab pertanyaan — Bagaimana memastikan kolaborasi efektif dan mengatasi hambatan?**

- **Pembagian peran berdasarkan tahapan, bukan berdasarkan halaman semata**: tiap anggota
  memegang satu area tanggung jawab utama (mis. data & preprocessing, model & evaluasi, dashboard
  FastAPI, dokumentasi/keamanan/laporan), tapi review kartu dilakukan silang antar anggota supaya
  tidak ada satu bagian pun yang hanya dipahami satu orang.
- **Board ClickUp sebagai pengganti rapat harian**: setiap anggota memperbarui status kartu
  miliknya sebelum sesi kerja bersama berikutnya, sehingga siapa pun bisa melihat progres tim
  tanpa perlu rapat sinkron setiap hari — mengurangi hambatan koordinasi jadwal kuliah yang
  berbeda-beda antar anggota.
- **Iterasi kecil dan sering**: setiap tahapan (satu fungsi preprocessing, satu model, satu
  halaman dashboard) diselesaikan dan diuji oleh PIC-nya sebelum kartu dipindah ke *Review*,
  alih-alih menunggu seluruh bagian selesai baru diintegrasikan — mengurangi risiko konflik/bug
  besar saat digabung menjelang deadline.
- **Mengatasi hambatan teknis atau ketergantungan antar anggota**: ketika satu kartu terhambat
  (mis. hasil model belum stabil sehingga anggota dashboard belum bisa lanjut), kartu dipindah ke
  kolom *Blocked* dengan catatan penyebab dan siapa yang perlu dihubungi, sehingga anggota lain
  bisa mengerjakan kartu independen lain (bukan ikut menganggur) sambil menunggu penyelesaian.
- **Definition of Done yang jelas per kartu**: mis. kartu "Notebook EDA" baru dianggap *Done*
  jika sudah direview minimal satu anggota lain dan dieksekusi ulang dari awal tanpa error (bukan
  sekadar selesai menurut satu orang yang mengerjakannya).

## B. DevOps/MLOps Sederhana

Alur kerja MLOps yang diterapkan mengikuti: **Data Collection → Data Preprocessing → Model
Training → Model Evaluation → Application Development → Testing → Deployment → Monitoring →
Improvement**.

| Komponen MLOps | Implementasi Nyata di Proyek Ini |
|---|---|
| **Version control** | Kode disimpan di repository Git/GitHub (`src/`, `app/`, `notebooks/`, `docs/`), memisahkan kode dari data (`data/`) dan artefak model (`models/`) |
| **Environment** | `requirements.txt` mendaftarkan seluruh dependency dengan versi minimum (pandas, scikit-learn, xgboost, lightgbm, fastapi, uvicorn, dll.) agar environment dapat direproduksi di mesin lain; disiapkan lewat virtual environment (`.venv`) khusus proyek |
| **Model storage** | Model hasil training disimpan sebagai `models/best_model.pkl` beserta metadata (`best_model_name.json`, `model_comparison_metrics.csv`) — tidak perlu training ulang setiap deployment |
| **Testing** | Notebook dijalankan ulang penuh (`nbconvert --execute`) untuk memastikan tidak ada error dari awal sampai akhir; aplikasi FastAPI diuji otomatis per route memakai `fastapi.testclient.TestClient`, termasuk uji submit form prediksi manual dan uji upload CSV batch (kasus valid, kosong, kolom hilang, nilai tidak valid) |
| **Deployment** | Aplikasi dijalankan lokal dengan `uvicorn app.main:app`; dapat pula di-deploy ke layanan cloud yang mendukung ASGI Python (mis. Render, Railway, atau VM/Container Service) dengan menghubungkan repository GitHub |
| **Monitoring** | Lihat sub-bagian *Monitoring Performa Model* di bawah |
| **Improvement** | Lihat sub-bagian *Rencana Pengembangan Berikutnya* di bawah |

### Monitoring Performa Model

Karena ini proyek akademik tanpa traffic produksi nyata, monitoring dijelaskan secara konseptual:
- **Potensi penurunan performa (model drift)**: jika tren harga properti berubah signifikan
  (mis. inflasi, kebijakan properti baru, pergeseran preferensi lokasi pasca-pandemi), pola yang
  dipelajari model dari data historis bisa menjadi usang — prediksi berisiko bias meski secara
  teknis model tidak "rusak".
  - **Indikator yang perlu dipantau**: RMSE/MAE pada data transaksi baru dibandingkan RMSE/MAE
    saat training; jika selisihnya melebar signifikan dari waktu ke waktu, itu tanda drift.
  - **Rencana penanganan**: mencatat metrik evaluasi setiap kali model dilatih ulang
    (`models/all_metrics.json`) agar bisa dibandingkan antar versi model dari waktu ke waktu.

### Rencana Pengembangan Berikutnya

1. Menambah sumber data baru (listing terbaru) secara berkala dan menjadwalkan retraining
   otomatis (mis. bulanan) alih-alih manual.
2. Menambahkan CI sederhana (GitHub Actions) yang otomatis menjalankan test `AppTest` setiap kali
   ada perubahan kode, sebelum deployment.
3. Melakukan hyperparameter tuning lebih sistematis (`GridSearchCV`/`Optuna`) pada XGBoost/LightGBM
   untuk mengejar peningkatan akurasi lebih lanjut.
4. Menambah fitur eksternal (jarak ke fasilitas publik, tren harga per periode) untuk memperkaya
   sinyal prediksi di luar luas tanah/bangunan yang saat ini mendominasi.

## Menjawab Pertanyaan Reflektif

**1. Bagaimana MLOps diterapkan agar model dapat di-deploy dan diperbarui secara otomatis?**
Model dipisahkan total dari kode aplikasi lewat serialisasi `.pkl` — aplikasi FastAPI hanya
memuat file model yang sudah jadi (`joblib.load`, dengan cache `functools.lru_cache` di
`app/services.py`), tidak pernah melatih ulang saat runtime. Ini
berarti pembaruan model cukup dilakukan dengan menjalankan ulang `src/train_model.py` lalu
mengganti file `models/best_model.pkl` — tanpa mengubah satu baris pun kode aplikasi, sehingga
proses "deploy model baru" dan "deploy aplikasi" bisa berjalan independen. Otomatisasi penuh
(mis. training terjadwal via cron/GitHub Actions yang otomatis push model baru) menjadi langkah
lanjutan yang direncanakan namun belum diimplementasikan pada versi akademik ini.

**2. Tantangan dalam mengelola pipeline continuous deployment, dan bagaimana mengatasinya?**
Tantangan utama sebagai tim dengan pembagian peran (anggota yang mengerjakan model berbeda dari
anggota yang mengerjakan dashboard) adalah **konsistensi preprocessing** antara tahap
training dan tahap inference (prediksi di dashboard) — jika logika encoding/scaling ditulis
ulang secara terpisah di notebook dan di aplikasi, keduanya mudah tidak sinkron seiring waktu.
Solusinya: seluruh logika preprocessing dipusatkan dalam satu modul (`src/preprocessing.py`)
yang **diimpor bersama** oleh `train_model.py` (saat training) dan `app/services.py` (saat
prediksi di FastAPI), sehingga hanya ada satu sumber kebenaran (*single source of truth*) untuk
transformasi data.

**3. Bagaimana memastikan model yang di-deploy terus diperbarui dengan data baru agar hasilnya
tetap akurat?**
Karena arsitektur training dan serving sudah dipisah (lihat poin 1), model dapat diperbarui
kapan pun data baru tersedia cukup dengan menjalankan kembali `src/train_model.py` — proses ini
otomatis membandingkan ulang keempat algoritma (Linear Regression, Random Forest, XGBoost,
LightGBM) dan menyimpan model dengan R² tertinggi sebagai model produksi, sehingga pembaruan
data juga otomatis diikuti pemilihan ulang model terbaik, bukan sekadar melatih ulang satu model
yang sama secara membabi buta.
