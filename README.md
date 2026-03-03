# 🏆 Hology 8.0 — Crowd Detection & Counting with CSRNet

> **Kompetisi:** Hology 8.0 — Data Mining Track  
> **Task:** Estimasi jumlah orang dalam gambar kerumunan (*crowd counting*)  
> **Pendekatan:** CSRNet + Adaptive Gaussian Density Map Regression

---

## 📋 Deskripsi Proyek

Proyek ini merupakan solusi end-to-end untuk **Crowd Detection & Crowd Counting** menggunakan arsitektur **CSRNet (Congested Scene Recognition Network)**. Model melakukan *density map regression*: alih-alih mendeteksi setiap individu satu per satu, model menghasilkan **peta kepadatan 2D** yang bila dijumlahkan menghasilkan estimasi total jumlah orang di gambar—pendekatan yang jauh lebih robust pada scene sangat padat.

---

## 📁 Struktur Repository

```
Hology-9.0-Crowd-Detection/
├── Notebook/
│   ├── HOLOGY_Solid.ipynb   # Notebook utama (EDA → Training → Inference)
│   └── requirements.txt     # Dependensi Python
├── assets/
│   └── density_map_comparison.png   # Visualisasi perbandingan density map (di-generate dari notebook)
└── README.md
```

---

## 🔍 Exploratory Data Analysis (EDA)

Dataset terdiri dari gambar resolusi tinggi beserta anotasi titik (`(x, y)` per orang). Distribusi `human_count` sangat **right-skewed**: sebagian besar gambar berisi kerumunan rendah (0–250 orang), dengan ekor panjang di atas 800 orang.

| Statistik | Keterangan |
|---|---|
| Format anotasi | JSON per gambar — array `points: [{x, y}]` |
| Split | 90% train / 10% val (`random_state=42`) |
| Rentang crowd | ~0 — 1000+ orang per gambar |

---

## 🗺️ Density Map — Inti Pendekatan

Daripada bounding box, setiap orang direpresentasikan sebagai satu titik impuls yang kemudian *di-smear* dengan kernel Gaussian. Jumlah seluruh piksel density map = jumlah orang sebenarnya.

Kami mengimplementasikan dan membandingkan **dua strategi** pembuatan density map:

### 1. Fixed Gaussian Filter (baseline sederhana)
$$F(x, y) = \sum_{i=1}^{N} \delta(x - x_i,\, y - y_i) * G_{\sigma}(x, y), \quad \sigma = 15$$

Semua titik diberi kernel yang sama — cepat, tetapi area padat cenderung "mencair" karena kernel terlap.

### 2. Adaptive k-NN Gaussian Kernel (metode utama)
$$\sigma_i = \text{clip}\!\left(\beta \cdot \frac{1}{k}\sum_{j=1}^{k}\text{dist}(p_i, p_{i_j}),\; \sigma_{\text{floor}},\; \sigma_{\text{cap}}\right)$$

Setiap titik mendapat $\sigma_i$ yang **menyesuaikan kerapatan sekitarnya** via kNN (k=3, β=0.3). Kernel dikerjakan hanya di area lokal radius $3\sigma_i$ sehingga konservasi massa lebih terjaga dan sinyal lebih terlokalisasi pada area padat.

### Perbandingan Visual

<img width="1490" height="1405" alt="image" src="https://github.com/user-attachments/assets/e440c46b-119e-40b5-bd2e-2981d51e4be9" />


| | Fixed Gaussian | Adaptive k-NN Gaussian |
|---|---|---|
| Sigma | Tetap (σ=15) | Adaptif per titik |
| Area padat | Blob menyatu | Puncak terlokalisasi |
| Konservasi massa | Baik | Lebih stabil |
| Kecepatan generate | Sangat cepat | Cukup cepat (kd-tree) |

---

## 🏗️ Arsitektur Model — CSRNet

```
Input Image [3 × H × W]
      │
      ▼
┌─────────────────────────────┐
│  Frontend: VGG-16-BN        │  ← Pre-trained ImageNet (frozen sebagian)
│  conv1_1 ... conv5_3        │
│  Output stride: 8×          │
└─────────────────────────────┘
      │  [512 × H/8 × W/8]
      ▼
┌─────────────────────────────┐
│  Backend: Dilated Conv ×6   │  ← 6 lapisan dilated conv (dilation=2)
│  512→512→512→256→128→64     │  memperluas receptive field tanpa downsampling
└─────────────────────────────┘
      │  [64 × H/8 × W/8]
      ▼
┌─────────────────────────────┐
│  Output: Conv2d(64→1)       │
│  + ReLU (non-negatif)       │
└─────────────────────────────┘
      │
      ▼
Density Map [1 × H/8 × W/8]
  → sum() = Predicted Count
```

**Mengapa VGG-16-BN?** Backbone klasik untuk CSRNet — representasi fitur spatial sangat baik, tidak terlalu *over-parametrized* untuk task regresi density.

---

## ⚙️ Pipeline Preprocessing & Augmentasi

### Dataset Class
| Class | Tujuan | Input Size |
|---|---|---|
| `CrowdTrainDataset` | Training dengan augmentasi + point-biased crop | patch 512×512 |
| `CrowdValDataset` | Validasi full-res tanpa augmentasi | max side 1536 px |

### Point-Biased Crop (70% probabilitas)
Crop dicentrasi secara acak di salah satu titik anotasi sehingga **patch hampir selalu mengandung orang** — mengurangi patch kosong yang percuma.

### Augmentasi (Albumentations)
- `RandomResizedCrop` (scale 0.5–1.0)
- `HorizontalFlip` (p=0.5)
- `Rotate` (±10°)
- `ColorJitter` + `GaussianBlur` (kondisi lapangan)
- `Normalize` ImageNet mean/std
- **Keypoint-aware**: titik anotasi ikut ter-transform secara konsisten

---

## 🏋️ Strategi Training

### Phase 1 — Baseline (25 Epoch)
| Parameter | Nilai |
|---|---|
| Optimizer | AdamW, lr=1e-4, wd=1e-4 |
| Scheduler | Cosine decay per epoch |
| Batch size | 16 |
| Loss | Charbonnier (density) + L1 count + relative count |
| AMP | ✅ Mixed Precision |

### Phase 2 — Fine-Tuning dari Baseline (25 Epoch + Early Stopping)
| Parameter | Nilai |
|---|---|
| LR Frontend (VGG) | 1e-6 (discriminative — lebih kecil) |
| LR Backend | 2e-5 |
| Scheduler | Warmup Cosine (warmup 1000 steps) |
| Batch size | 8 |
| EMA decay | 0.999 |
| Early stopping | patience = 7 epoch |
| Gradient clipping | max norm = 1.0 |

### Fungsi Loss
$$\mathcal{L} = \underbrace{\frac{1}{N}\sum\sqrt{(\hat{d}-d)^2+\varepsilon^2}}_{\text{Charbonnier (density)}} + 0.1\cdot\underbrace{|\hat{c}-c|}_{\text{L1 count}} + 0.05\cdot\underbrace{\frac{|\hat{c}-c|}{c+1}}_{\text{relative count}}$$

Charbonnier dipilih atas MSE karena lebih robust terhadap outlier density.

---

## 🔮 Inference & Test-Time Augmentation (TTA)

Saat prediksi, tiap gambar di-infer pada **3 skala** (0.8×, 1.0×, 1.25×) dan dengan/tanpa horizontal flip → total **6 forward pass**, hasilnya di-average untuk prediksi final yang lebih stabil.

```python
pred_count = mean([count(scale=0.8), count(scale=0.8, flip),
                   count(scale=1.0), count(scale=1.0, flip),
                   count(scale=1.25), count(scale=1.25, flip)])
```

---

## 🚀 Cara Menjalankan

### 1. Install Dependensi
```bash
pip install -r Notebook/requirements.txt
```

### 2. Jalankan di Google Colab
Buka `Notebook/HOLOGY_Solid.ipynb` di Google Colab, lalu jalankan sel secara berurutan:
1. **Mounting Drive** — mount Google Drive & set working directory
2. **Download Data** — download dataset via Kaggle API
3. **EDA** — eksplorasi distribusi crowd count
4. **Preprocessing & Modelling** — build density maps, dataset, model, training
5. **Predict** — generate `submission.csv`

> ⚠️ Pastikan Kaggle credentials tersimpan di Colab Secrets dengan key `KAGGLE_JSON`.

---

## 📦 Dependensi Utama

| Library | Fungsi |
|---|---|
| `torch` / `torchvision` | Model, training, VGG backbone |
| `albumentations` | Augmentasi keypoint-aware |
| `scipy` | Gaussian filter, kd-tree kNN |
| `opencv-python` | Load & manipulasi gambar |
| `scikit-learn` | Train/val split |
| `tqdm` | Progress bar training |

---

## 📊 Metrik Evaluasi

**Mean Absolute Error (MAE)** — rata-rata selisih absolut antara jumlah prediksi dan ground truth:
$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i - y_i|$$


