# 12. Optimasi dan Skalabilitas Aplikasi

**Pertanyaan:** Apa strategi yang dipilih untuk memastikan aplikasi dapat menangani volume data
yang besar dan tetap responsif?

Strategi yang diterapkan pada aplikasi Rumahku bukan sekadar konsep — sudah tercermin langsung
di kode `app/main.py`, `app/services.py`, dan `src/train_model.py`.

| Aspek | Optimasi yang Diterapkan | Bukti di Implementasi |
|---|---|---|
| **Data** | Fitur `id_properti` (tidak relevan untuk prediksi) dibuang sejak awal; kolom numerik distandardisasi agar konvergensi model lebih cepat; missing value diimputasi lewat pipeline, bukan proses manual berulang | `src/preprocessing.py` — `ALL_FEATURES` tidak menyertakan `id_properti`; `ColumnTransformer` |
| **Model** | Hyperparameter dipilih agar seimbang antara akurasi dan waktu training (`n_estimators`, `max_depth`, `subsample`, `colsample_bytree`); 4 model dibandingkan lalu hanya **model terbaik** yang disimpan dan dipakai di produksi, bukan menjalankan ke-4 model setiap kali ada request | `src/train_model.py` — hanya `best_pipeline` yang di-`joblib.dump` |
| **Aplikasi** | Data dan model **di-cache di memori** memakai `functools.lru_cache` sehingga file CSV dan model `.pkl` hanya dibaca sekali dari disk selama proses server hidup, bukan pada setiap request | `app/services.py` — dekorator `@functools.lru_cache(maxsize=1)` pada `load_data()`, `load_model()`, `load_metrics()`, dll. |
| **Dashboard** | Grafik menampilkan **data agregat** (rata-rata harga per kota, distribusi, top-5 feature importance) alih-alih me-render seluruh 5.000 baris mentah ke pengguna; tabel besar dibatasi lewat parameter jumlah baris | `app/main.py` route `/` (`groupby('kota')`), `/insight` (`importance_df.head(5)`), `/dataset` (`?n=` di query string) |
| **Infrastruktur** | Model disimpan sebagai artefak terpisah (`.pkl`, `.csv`, `.json`) di folder `models/`, terlepas dari kode aplikasi — memudahkan pemindahan ke container/cloud storage tanpa mengubah kode; `requirements.txt` memastikan environment reproducible di server manapun | `models/*.pkl`, `models/*.json`, `requirements.txt` |
| **Keamanan** | Aplikasi tidak mengekspos endpoint tulis ke dataset asli (hanya baca), hasil upload CSV pengguna diproses in-memory dan tidak disimpan permanen di server, dan tidak ada kredensial hardcoded di kode | Semua akses data di `app/services.py` bersifat read-only (`pd.read_csv`, `joblib.load`); upload di `/prediksi/batch` dibaca lewat `io.BytesIO`, tidak ditulis ke disk |
| **Model deployment** | Model disimpan dalam file `.pkl` (`models/best_model.pkl`) sehingga **tidak perlu training ulang** setiap kali aplikasi dijalankan — training hanya dijalankan manual lewat `python src/train_model.py` saat data diperbarui | `models/best_model.pkl`, dipanggil lewat `joblib.load()` di `load_model()` |

## Strategi Skalabilitas ke Depan (jika volume data tumbuh jauh lebih besar)

Jika volume data bertumbuh dari ribuan ke jutaan baris, langkah lanjutan yang direkomendasikan:
1. **Data**: pindahkan penyimpanan dari CSV ke database relasional (PostgreSQL) dengan index pada
   kolom yang sering difilter (`kota`, `tipe_properti`), dan gunakan sampling/agregasi
   sebelum data dikirim ke dashboard.
2. **Model**: pertimbangkan training incremental atau retraining terjadwal (bukan real-time),
   serta gunakan format model yang lebih ringan (mis. `ONNX`) untuk inferensi lebih cepat.
3. **Aplikasi**: tambahkan lapisan caching eksternal (Redis) jika jumlah pengguna simultan besar,
   dan pertimbangkan memisahkan endpoint prediksi (API JSON) dari endpoint halaman (HTML) agar
   beban komputasi model tidak membebani proses rendering UI.
4. **Infrastruktur**: deploy aplikasi FastAPI di container (Docker) di belakang server ASGI
   produksi (mis. beberapa worker Uvicorn/Gunicorn) di atas layanan cloud (VM/Container Service)
   agar mudah discale secara horizontal saat traffic meningkat.
