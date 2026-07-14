# B. Dataset — Rumahku House Prices

> **Catatan:** Dataset ini disebutkan berasal dari Kaggle, namun link persisnya belum tersedia.
> **TODO mahasiswa:** lengkapi URL Kaggle yang sebenarnya, dan tetap konsultasikan dataset ini ke
> dosen pengampu sebelum digunakan sebagai dataset final (sesuai ketentuan tugas besar).

## Ringkasan Komponen

| Komponen | Penjelasan |
|---|---|
| **Nama dataset** | Rumahku House Prices Dataset (`rumahku_house_prices.csv`) |
| **Sumber dataset** | Kaggle *(link belum dilengkapi — TODO)* |
| **Domain dataset** | Properti / Real Estate — estimasi harga jual rumah/villa/townhouse |
| **Jumlah data** | 5.000 baris (record properti) |
| **Jumlah fitur** | 13 kolom total: 1 identifier (`id_properti`), 11 fitur prediktor, 1 variabel target |
| **Variabel target** | `harga_jual` (numerik, dalam juta Rupiah) → kasus **supervised learning, regresi** |
| **Jenis data** | Campuran: numerik kontinu, numerik diskrit, kategorikal nominal/ordinal, dan biner |

## Struktur Kolom

| Kolom | Tipe | Keterangan |
|---|---|---|
| `id_properti` | Teks (identifier) | Kode unik properti, tidak digunakan sebagai fitur prediksi |
| `luas_tanah` | Numerik kontinu (m²) | Luas tanah |
| `luas_bangunan` | Numerik kontinu (m²) | Luas bangunan |
| `kamar_tidur` | Numerik diskrit | Jumlah kamar tidur |
| `kamar_mandi` | Numerik diskrit | Jumlah kamar mandi |
| `jumlah_lantai` | Numerik diskrit | Jumlah lantai bangunan |
| `usia_bangunan` | Numerik kontinu (tahun) | Usia bangunan |
| `jarak_pusat_kota` | Numerik kontinu (km) | Jarak ke pusat kota |
| `kondisi_bangunan` | Kategorikal ordinal | Buruk / Cukup / Baik |
| `tipe_properti` | Kategorikal nominal | Rumah / Villa / Townhouse |
| `garasi` | Biner | 1 = punya garasi, 0 = tidak |
| `kota` | Kategorikal nominal | Bandung, Surabaya, Jakarta, Semarang, Medan |
| `harga_jual` | Numerik kontinu (juta Rupiah) | **Target** — harga jual properti |

## Alasan Pemilihan Dataset

1. **Kesesuaian dengan kasus regresi**: variabel target `harga_jual` bersifat numerik kontinu, cocok untuk membangun dan membandingkan model regresi (Linear Regression, Random Forest, XGBoost, LightGBM).
2. **Keragaman tipe fitur**: dataset memuat numerik, kategorikal nominal, kategorikal ordinal, dan biner sekaligus — representatif untuk mendemonstrasikan seluruh tahapan preprocessing (encoding, scaling, penanganan missing value).
3. **Ukuran data proporsional**: 5.000 baris cukup besar untuk melatih model yang stabil, namun tetap ringan untuk diproses di laptop/Colab dan ditampilkan real-time di dashboard web (FastAPI).
4. **Kasus bisnis jelas**: relevan dengan aplikasi estimasi harga properti (proptech), sehingga insight dan rekomendasi model mudah diterjemahkan menjadi keputusan bisnis nyata.
5. **Kualitas data realistis untuk latihan**: mengandung missing value dan potensi outlier secara alami, sehingga tahapan pembersihan data (bagian C) benar-benar diperlukan, bukan sekadar formalitas.

## Potensi Risiko Data

| Risiko | Detail Temuan | Dampak |
|---|---|---|
| **Missing value** | `luas_tanah` (50), `luas_bangunan` (50), `usia_bangunan` (75), `jarak_pusat_kota` (75), `kondisi_bangunan` (50), `kota` (25), `harga_jual` (40) baris kosong dari total 5.000 | Jika tidak ditangani, mengurangi jumlah data valid atau menyebabkan error saat training model |
| **Outlier pada target** | 136 baris `harga_jual` berada di luar batas IQR (whisker atas ≈ 4.850 juta), harga maksimum mencapai 19.784 juta | Bisa menarik garis regresi linear secara tidak proporsional jika tidak dievaluasi dengan cermat |
| **Ketidakseimbangan kategori** | `tipe_properti`: Rumah mendominasi (3.031) dibanding Villa (1.080) dan Townhouse (889); `kondisi_bangunan`: mayoritas "Baik" (2.464) dan "Cukup" (1.711) dibanding "Buruk" (775) | Model berpotensi kurang akurat memprediksi harga pada kategori minoritas (mis. Townhouse kondisi buruk) |
| **Data tidak lengkap per baris** | Beberapa baris kehilangan lebih dari satu nilai fitur sekaligus | Perlu strategi imputasi yang konsisten, bukan sekadar drop baris agar tidak kehilangan terlalu banyak data |
| **Representativeness** | Sumber asli dataset (Kaggle) belum terverifikasi linknya; kemungkinan data bersifat sintetis/simulasi | Insight harga mungkin tidak sepenuhnya mencerminkan kondisi pasar properti riil — perlu disebutkan sebagai keterbatasan saat menyampaikan hasil |
| **Data sensitif** | Tidak ditemukan data pribadi (nama pemilik, alamat lengkap, kontak) — hanya `id_properti` sebagai kode acak | Risiko privasi rendah, aman digunakan untuk keperluan akademik |
