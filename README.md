# Deep Learning Advanced CNNs and Transfer Learning

Proyek ini berisi eksperimen klasifikasi gambar hewan peliharaan menggunakan pendekatan **CNN lanjutan** dan **transfer learning** pada PyTorch.

## Isi Proyek

- `Eksperimen_ChristellaCindyW_2802407742.ipynb` : notebook utama berisi seluruh pipeline eksperimen.
- `config_n_split.json` : konfigurasi eksperimen dan pembagian data (train/val/test).
- `hasil_ringkasan.csv` : ringkasan hasil akhir tiap skenario eksperimen.
- `riwayat_epoch.csv` : riwayat metrik per epoch.
- `kesalahan_test_B5_efficientnet_b0.csv` : detail prediksi salah pada model B5.
- `kurva_loss.png`, `failure_analysis.png`, `c5_feature_map.png` : visualisasi hasil eksperimen.

## Metode Eksperimen

Eksperimen membandingkan beberapa strategi pelatihan:

1. Pelatihan dari nol (`B1_dari_nol`)
2. Feature extraction (`B2_feature_extraction`)
3. Fine-tuning (`B3_finetuning`)
4. Fine-tuning tanpa augmentasi (`B4_B3_tanpa_augmentasi`)
5. EfficientNet-B0 feature extraction (`B5_efficientnet_b0`)

Berdasarkan `hasil_ringkasan.csv`, performa terbaik diperoleh pada eksperimen **B5_efficientnet_b0**.

## Persiapan Environment

1. Buat virtual environment (opsional tapi direkomendasikan).
2. Install dependensi:

```bash
pip install -r requirements.txt
```

## Menjalankan Proyek

- Buka dan jalankan notebook:
  - `Eksperimen_ChristellaCindyW_2802407742.ipynb`
- Jika menggunakan Google Colab, pastikan akses dataset dan Google Drive sudah sesuai.

## Catatan

- Konfigurasi acak menggunakan `seed = 42`.
- Notebook disusun untuk komputasi GPU (contoh: Tesla T4), namun tetap bisa diadaptasi ke CPU dengan waktu proses lebih lama.
