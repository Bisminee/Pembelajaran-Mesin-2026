# Penjelasan Kode Praktikum Pembelajaran Mesin

Dokumen ini menjelaskan **setiap sel kode** pada seluruh notebook praktikum, pertemuan 1 sampai pertemuan 3, baris demi baris beserta alasan di baliknya. Di akhir dokumen ada ringkasan konsep dan daftar perbaikan kode yang disarankan.

## Peta Materi

| Lokasi | Notebook | Topik |
|---|---|---|
| Pertemuan1 | `Praktikum1.ipynb` | Notebook kosong (tanpa sel) |
| Pertemuan1 | `Praktikum_1.ipynb` | Persiapan lingkungan (install `mne`) |
| Pertemuan1 | `TG1_244107020216_BISMA ADHIAKSA.ipynb` | Instalasi library + jawaban esai etika & lingkungan AI |
| Pertemuan2 | `Praktikum1.ipynb` | Exploratory Data Analysis (EDA) dataset Titanic |
| Pertemuan2 | `Praktikum2.ipynb` | Data imputation (mean, konstanta, modus) |
| Pertemuan2 | `Praktikum3.ipynb` | Encoding & standardisasi |
| Pertemuan2 | `Praktikum4.ipynb` | Pengolahan citra dasar dengan OpenCV & Matplotlib |
| Pertemuan2 | `Tugas Praktikum/breast_cancer_analysis.ipynb` | Tugas: seleksi variabel, encoding, standardisasi data kanker payudara |
| Pertemuan3 | `Praktikum1.ipynb` | Ekstraksi fitur (feature engineering) Titanic |
| Pertemuan3 | `Praktikum2.ipynb` | Pra-pemrosesan: encoding manual, deck Cabin, standardisasi |
| Pertemuan3 | `Praktikum3.ipynb` | Split data (train/val/test) & K-Fold Cross Validation |
| Pertemuan3 | `Praktikum4.ipynb` | Ekstraksi fitur citra: histogram channel RGB |

---

# PERTEMUAN 1

## 1. `Pertemuan1/Praktikum1.ipynb`

Notebook ini **kosong**: berisi `"cells": []` pada file JSON-nya. Hanya ada metadata kernel (Python 3.11.9). Jadi tidak ada kode yang bisa dijelaskan — kemungkinan file dibuat untuk latihan awal namun tidak diisi.

## 2. `Pertemuan1/Praktikum_1.ipynb`

**Sel 1 (markdown)** — badge "Open in Colab":

```html
<a href="https://colab.research.google.com/github/.../Praktikum_1.ipynb" target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
```

- HTML `<a>` = hyperlink, `<img>` = gambar badge.
- Fungsinya membuat tombol di notebook GitHub agar file bisa langsung dibuka dan dijalankan di Google Colab (`colab.research.google.com/github/...`).

**Sel 2 (kode)**:

```python
%pip install mne
```

- `%pip` adalah *magic command* IPython/Jupyter yang menjalankan `pip install` **di dalam kernel notebook yang sedang aktif** (bukan di terminal terpisah). Awalan `%` membuat perintah ini dijalankan oleh IPython, bukan Python.
- `mne` = library untuk analisis sinyal **MNE (Magnetoencephalography/Elektroensefalografi)**, yaitu pengolahan data EEG/MEG. Ini langkah persiapan agar library tersedia sebelum praktik.

**Sel 3 (kode)** — kosong (tidak ada isi).

## 3. `Pertemuan1/TG1_244107020216_BISMA ADHIAKSA.ipynb`

**Sel 1 (kode)** — instalasi library pendukung:

```python
%pip install pyprep
%pip install scipy
%pip install wandb
%pip install scikit-learn
%pip install pyecg --no-deps
```

- `%pip install ...` → memasang setiap library ke lingkungan notebook (satu perintah per baris agar mudah melihat mana yang gagal).
- `pyprep` → pra-pemrosesan sinyal **EEG otomatis** (deteksi channel buruk, interpolasi, referencing).
- `scipy` → algoritma matematika & komputasi ilmiah (FFT, filter, statistik, optimisasi).
- `wandb` → *Weights & Biases*: mencatat, memantau, dan mengelola eksperimen Machine Learning/Deep Learning.
- `scikit-learn` → library utama ML: preprocessing, scaling, algoritma ML, reduksi dimensi, evaluasi & validasi model.
- `pyecg --no-deps` → analisis sinyal **ECG (elektrokardiogram/jantung)**. Flag `--no-deps` artinya menginstal pyecg **tanpa dependensinya**, supaya pip tidak menimpa/menurunkan versi library yang sudah ada (mencegah konflik versi, mis. numpy/scipy).

**Sel 2 (markdown)** — penjelasan library (lihat di atas) dan catatan bahwa `pyecg` awalnya error karena versi terlalu lama sehingga dipasang `--no-deps` + `scikit-learn`.

**Sel 3 (markdown)** — jawaban esai:

1. **Contoh pelanggaran etika & hukum AI (manipulasi riset)**:
   - Etika: melanggar integritas peneliti sesuai Permendikbudristek No. 39 Tahun 2021.
   - Hukum: melanggar UU ITE Pasal 35 (manipulasi informasi elektronik/dokumen elektronik).
2. **Dampak energi & lingkungan AI**:
   - Pusat data AI diperkirakan "mengonsumsi" air setara kebutuhan harian 5 juta orang pada 2030 (±450 juta galon, tahun 2022 baru 292 juta galon) untuk pendinginan.
   - Konsumsi listrik AI diprediksi 945 TWh (2030) dari 415 TWh (2024); sekitar 20% permintaan listrik baru, sementara pasokan masih ±40% bahan bakar fosil.
   - Solusi yang bisa disebutkan: efisiensi model (smaller models/quantization), pemakaian energi terbarukan, optimasi pendinginan data center, dan pengaturan/regulasi.

---

# PERTEMUAN 2

Tema: **pra-pemrosesan data** — EDA, imputasi, encoding, scaling, dan pengolahan citra.

## 1. `Pertemuan2/Praktikum1.ipynb` — Exploratory Data Analysis (EDA)

**Sel 2**: import library.

```python
import pandas as pd          # manipulasi tabel data (DataFrame)
import numpy as np           # operasi numerik/array
import matplotlib.pyplot as plt  # visualisasi dasar (histogram, scatter, boxplot)
```

**Sel 4**: memuat data.

```python
df = pd.read_csv('Titanic-Dataset.csv')  # baca CSV menjadi DataFrame
df.head(100)                             # tampilkan 100 baris pertama
```

- `pd.read_csv` membaca file CSV di folder yang sama dengan notebook.
- `df.head(n)` menampilkan `n` baris teratas; dipakai `100` agar bisa melihat karakter data lebih banyak sekaligus.

**Sel 6**: inspeksi dimensi & tipe data.

```python
print('(baris, kolom) =', df.shape, '\n')   # pengecekan dimensi
df.info()                                   # pengecekan tipe data dan variabel
```

- `df.shape` → tuple `(jumlah_baris, jumlah_kolom)`; Titanic: `(891, 12)`.
- `df.info()` → ringkasan tiap kolom: nama, jumlah nilai tidak-null, tipe data (`int64`, `float64`, `object`), dan penggunaan memori. Kolom `object` = teks/kategorikal.

**Sel 7**: cek missing value.

```python
df.isnull().sum()   # kode mengecek jumlah data hilang pada tiap kolom
```

- `df.isnull()` menghasilkan tabel boolean (True = hilang/NaN), lalu `.sum()` menjumlahkan True per kolom. Hasil penting Titanic: `Age` 177 hilang, `Cabin` 687 hilang, `Embarked` 2 hilang.

**Sel 8**: statistik deskriptif.

```python
df.describe()   # kode untuk menampilkan statistik ringkas dari data
```

- Untuk kolom numerik menampilkan: `count` (jumlah non-null), `mean`, `std` (simpangan baku), `min`, kuartil 25/50/75%, dan `max`. Berguna mendeteksi rentang tidak wajar (mis. `Fare` maksimum 512 jauh di atas rata-rata 32 → indikasi outlier).

**Sel 11**: histogram distribusi numerik.

```python
num_cols = ['Age','Fare']
for col in num_cols:
    plt.figure()                    # buat kanvas plot baru tiap loop
    plt.hist(df[col], bins=30)      # histogram dengan 30 bin
    plt.title(f'Distribusi {col}')
    plt.xlabel(col); plt.ylabel('Frekuensi')
    plt.show()
```

- `plt.figure()` membuat figure baru agar histogram kedua tidak menimpa yang pertama.
- `plt.hist(..., bins=30)` membagi rentang nilai menjadi 30 interval dan menghitung frekuensinya → melihat bentuk distribusi (normal, miring/skewed).
- `f'Distribusi {col}'` = f-string, menyisipkan nilai variabel ke dalam teks.

**Sel 13**: boxplot outlier.

```python
plt.figure()
plt.boxplot(df['Fare'].dropna(), vert=True)  # dropna: buang NaN sebelum plot
plt.title('Boxplot Fare')
plt.ylabel('Fare')
plt.show()
```

- `dropna()` membuang nilai kosong agar tidak mengganggu plot.
- Boxplot menampilkan median (garis tengah kotak), Q1–Q3 (kotak), dan titik di luar ±1.5×IQR sebagai **outlier**. `vert=True` membuat orientasi vertikal.

**Sel 15**: komposisi variabel kategorikal.

```python
cat_cols = ['Survived','Pclass','Sex','Embarked']

fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # grid 2x2
axes = axes.flatten()                            # jadikan array 1 dimensi

for i, col in enumerate(cat_cols):
    df[col].value_counts(dropna=False).plot(kind='bar', ax=axes[i])
    axes[i].set_title(f'Distribusi {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Jumlah')

plt.tight_layout()  # atur jarak antar subplot agar tidak tumpang tindih
plt.show()
```

- `plt.subplots(2,2)` membuat 4 area plot sekaligus; `axes.flatten()` memudahkan mengakses lewat indeks `axes[i]`.
- `value_counts()` menghitung frekuensi tiap nilai unik; `dropna=False` ikut menghitung NaN (jika ada).
- `.plot(kind='bar', ax=axes[i])` menggambar diagram batang **pada subplot tertentu** (parameter `ax`).
- `enumerate` memberikan indeks `i` dan nama kolom `col` sekaligus.
- `tight_layout()` merapikan tata letak.

**Sel 17**: heatmap korelasi.

```python
import seaborn as sns

num_only = df.select_dtypes(include=[np.number])  # hanya kolom numerik
corr = num_only.corr(numeric_only=True)           # matriks korelasi

plt.figure(figsize=(8, 6))
sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt=".2f")
plt.title('Heatmap Korelasi (numerik)')
plt.tight_layout()
plt.show()

corr['Survived'].sort_values(ascending=False)
```

- `select_dtypes(include=[np.number])` memfilter kolom bertipe angka (korelasi hanya bermakna untuk numerik).
- `.corr()` menghitung koefisien korelasi (default Pearson, rentang −1..1) antar pasangan kolom.
- `sns.heatmap`: `cmap='coolwarm'` (biru=negatif, merah=positif), `vmin/vmax` mengunci skala warna, `annot=True` menulis angka di sel, `fmt=".2f"` dua desimal.
- Baris terakhir mengurutkan korelasi tiap fitur terhadap `Survived` → mencari fitur yang paling berpengaruh (mis. `Sex`, `Pclass`, `Fare`).

**Sel 19**: scatter plot.

```python
plt.figure()
surv_map = {0:'No', 1:'Yes'}   # variabel pemetaan label (didefinisikan tapi tidak dipakai)
plt.scatter(df['Age'], df['Fare'], alpha=0.7)  # alpha = transparansi titik
plt.xlabel('Age'); plt.ylabel('Fare')
plt.title('Scatter Age vs Fare')
plt.show()
```

- Scatter plot melihat hubungan dua variabel numerik. `alpha=0.7` membuat titik sedikit transparan agar kepadatan data (overplotting) terlihat.
- Catatan kecil: `surv_map` tidak dipakai di plot ini.

## 2. `Pertemuan2/Praktikum2.ipynb` — Data Imputation

**Sel 3**: import `pandas`, `numpy`.

**Sel 5**: muat data dan tampilkan 5 baris pertama.

```python
df = pd.read_csv('Titanic-Dataset.csv')
df.head()
```

**Sel 7–10**: inspeksi standar (`shape`, `info`, `isnull().sum()`, `describe()`).

**Sel 12**: imputasi (pengisian) missing value.

```python
# Age - mean
df['Age'].fillna(value=df['Age'].mean(), inplace=True)

# Cabin - "DECK"
df['Cabin'].fillna(value="DECK", inplace=True)

# Embarked - modus
df['Embarked'].fillna(value=df['Embarked'].mode, inplace=True)
```

- **Age → mean** (`df['Age'].mean()` ≈ 29.7): dipilih karena Age numerik dan relatif simetris; mean mudah dan menjaga rata-rata tetap. (Median lebih tahan outlier/skew — lihat Pertemuan 3.)
- **Cabin → "DECK"**: kolom kategorikal dengan 687/891 kosong; angka kosong diisi label baru `"DECK"` sebagai kategori "tidak diketahui", bukan dihapus.
- **Embarked → modus**: niatnya mengisi 2 nilai kosong dengan nilai tersering (`S`). **Namun ada bug**: `df['Embarked'].mode` (tanpa tanda kurung) mengoper *method*-nya, bukan nilainya. Yang benar: `df['Embarked'].mode()[0]`.
- `inplace=True` membuat perubahan langsung ke DataFrame. Pada pandas modern, gaya `df[col].fillna(..., inplace=True)` menimbulkan `FutureWarning: chained assignment` karena objek antara dianggap salinan. Gaya yang disarankan:
  ```python
  df['Age'] = df['Age'].fillna(df['Age'].mean())
  # atau
  df.fillna({'Age': df['Age'].mean(), 'Cabin': 'DECK', 'Embarked': df['Embarked'].mode()[0]}, inplace=True)
  ```

**Sel 14–15**: validasi hasil.

```python
df.info()
df.isnull().sum()
df.head(10)
```

- `df.isnull().sum()` semua bernilai 0 → tidak ada lagi data hilang. Perhatikan juga baris `5` pada `head(10)` yang sebelumnya NaN Age, kini terisi 29.699118 (mean), dan Cabin terisi `DECK`.

## 3. `Pertemuan2/Praktikum3.ipynb` — Encoding & Standardisasi

**Sel 1**: import.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
```

- `LabelEncoder` = mengubah label kategorikal (teks) menjadi integer 0..n−1.
- `StandardScaler` = standardisasi z-score: `z = (x − μ) / σ`.

**Sel 3**: muat data hasil imputasi dari notebook sebelumnya.

```python
dpath = 'Titanic-Dataset-fixed.csv'
df = pd.read_csv(dpath)
df.head()
```

**Sel 5**: memilih kolom yang dipakai.

```python
df = df[['Survived', 'Pclass', 'Age', 'Sex', 'Cabin']]
df.head()
```

- `df[['...','...']]` (kurung siku ganda) = seleksi **beberapa kolom** sekaligus menghasilkan DataFrame baru; beda dengan `df['kolom']` yang menghasilkan Series.

**Sel 7**: Label Encoding.

```python
le = LabelEncoder()                            # objek encoder
df['Sex'] = le.fit_transform(df['Sex'])        # encoding Sex
df['Cabin'] = le.fit_transform(df['Cabin'])    # encoding Cabin
```

- `fit` = mempelajari daftar kategori unik; `transform` = mengubah tiap kategori menjadi angka; `fit_transform` = keduanya sekaligus.
- `Sex`: secara alfabetis `female=0`, `male=1`.
- `Cabin`: setiap nilai unik seperti `C85`, `C123` menjadi angka urut alfabetis. **Catatan kritis**: Cabin bukan data ordinal, sehingga angka hasil LabelEncoder **tidak punya makna urutan** (C123 tidak "lebih besar" dari C85). Untuk model, ini kurang tepat; alternatif: one-hot encoding atau ambil dek (huruf pertama) seperti di Pertemuan 3.

**Sel 11**: standardisasi Age.

```python
std = StandardScaler()
df['Age'] = std.fit_transform(df[['Age']])
```

- `df[['Age']]` (2D) karena `StandardScaler` membutuhkan input 2D (baris × kolom).
- Hasil: rata-rata Age menjadi 0 dan simpangan baku 1, sehingga skala Age setara kolom lain — penting untuk model berbasis jarak (KNN, SVM) atau gradient descent.
- Catatan ML: idealnya scaler di-`fit` hanya pada data **train**, lalu `transform` data validasi/test, agar tidak terjadi **data leakage** (kebocoran informasi mean/std dari data test).

**Sel 9 & 12**: verifikasi dengan `df.head()`.

## 4. `Pertemuan2/Praktikum4.ipynb` — Pengolahan Citra dengan OpenCV

**Sel 1**:

```python
!pip install -q opencv-python matplotlib
```

- `!pip` (tanda seru) menjalankan perintah shell/pip dari dalam notebook; `-q` = quiet, output lebih ringkas.

**Sel 2**:

```python
import cv2                     # OpenCV: baca/olah citra
import matplotlib.pyplot as plt  # menampilkan gambar
```

**Sel 4**: baca & tampilkan gambar.

```python
img = cv2.imread("Lenna.png")                       # baca gambar (format BGR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # konversi BGR -> RGB
plt.imshow(img_rgb)
plt.title("Gambar Asli")
plt.axis("off")                                     # sembunyikan sumbu x/y
plt.show()
```

- `cv2.imread` membaca gambar sebagai array NumPy dengan urutan channel **BGR** (Blue-Green-Red), berbeda dari standar umumnya RGB.
- Matplotlib menganggap urutan channel adalah RGB, sehingga tanpa konversi warna merah dan biru akan tertukar. `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` memperbaikinya.
- `plt.axis("off")` menyembunyikan koordinat piksel agar tampilan seperti foto.

**Sel 6**: resize.

```python
img_resized = cv2.resize(img_rgb, (128, 128))
plt.imshow(img_resized)
plt.title("Gambar setelah Resize (128x128)")
plt.axis("off")
plt.show()
```

- `cv2.resize` mengubah ukuran menjadi 128×128 piksel. Ukuran seragam penting agar citra bisa dijadikan input model (mis. CNN) atau dibandingkan satu sama lain.

**Sel 8**: grayscale.

```python
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray, cmap="gray")
plt.title("Gambar Grayscale")
plt.axis("off")
plt.show()
```

- Konversi ke satu channel keabuan dengan rumus luminance `Y = 0.299R + 0.587G + 0.114B` (versi OpenCV menyesuaikan urutan channel input).
- `cmap="gray"` memberi tahu Matplotlib agar array 2D ditampilkan hitam-putih (tanpa ini, warna dipetakan ke colormap ungu-kuning).
- Detail teknis: karena `img_resized` sudah RGB tetapi dikonversi dengan `COLOR_BGR2GRAY`, bobot channel R dan B tertukar. Perbedaannya kecil secara visual; konversi yang tepat adalah `cv2.COLOR_RGB2GRAY` untuk input RGB.

**Sel 9**: kosong.

## 5. `Pertemuan2/Tugas Praktikum/breast_cancer_analysis.ipynb`

**Sel 5**: muat data.

```python
df = pd.read_csv('wbc.csv')
df.info()
df.head()
```

- Dataset *Wisconsin Breast Cancer*: 569 baris, target `diagnosis` berisi `M` (malignant/ganas) dan `B` (benign/jinak), plus fitur numerik seperti `radius_mean`, `texture_mean`, dst.
- Kolom `Unnamed: 32` muncul karena file CSV punya koma berlebih di akhir header; isinya kosong semua.
- Kolom `id` hanyalah identitas, tidak berguna untuk prediksi.

**Sel 7**: memisahkan variabel yang dapat/tidak dapat digunakan.

```python
unuse = ['id', 'Unnamed: 32']          # kolom yang dibuang
target_col = 'diagnosis'               # kolom target
usable = [c for c in df.columns if c not in unuse + [target_col]]  # fitur
df = df.drop(columns=unuse)            # buang kolom tak terpakai
```

- `unuse + [target_col]` menggabungkan dua list menjadi `['id','Unnamed: 32','diagnosis']`.
- *List comprehension* `[c for c in df.columns if c not in ...]` menghasilkan daftar nama kolom fitur numerik.
- `df.drop(columns=...)` menghapus kolom `id` dan `Unnamed: 32` dari tabel.

**Sel 9**: encoding target.

```python
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])
```

- `B` dan `M` diubah menjadi angka: `B=0`, `M=1` (urut alfabetis). Model ML butuh target numerik.

**Sel 12**: standardisasi semua fitur numerik.

```python
scaler = StandardScaler()
df[usable] = scaler.fit_transform(df[usable])
```

- `df[usable]` = sub-tabel semua kolom fitur (2D), distandardisasi sekaligus (tiap kolom dihitung mean/std-nya sendiri).
- Tujuan: fitur seperti `area_mean` (ratusan/ribuan) tidak mendominasi fitur seperti `smoothness_mean` (0–1).

**Sel 14**: verifikasi.

```python
df[usable].describe().loc[['mean', 'std']]
```

- `.describe()` menghasilkan tabel statistik; `.loc[['mean','std']]` mengambil hanya baris mean dan std. Hasilnya mean ≈ 0 dan std ≈ 1 → standardisasi berhasil.

---

# PERTEMUAN 3

Tema: **ekstraksi fitur** (feature engineering) dan **validasi model**.

## 1. `Pertemuan3/Praktikum1.ipynb` — Ekstraksi Fitur Titanic

**Sel 1**: muat data.

```python
data = 'data/Titanic-Dataset.csv'
df = pd.read_csv(data)
```

**Sel 2–4**: inspeksi (`head`, `info`, `isnull().sum`).

**Sel 5**: imputasi missing value (sama seperti Pertemuan 2).

```python
df['Age'].fillna(value=df['Age'].mean(), inplace=True)     # Age -> mean
df['Cabin'].fillna(value="DECK", inplace=True)             # Cabin -> "DECK"
df['Embarked'].fillna(value=df['Embarked'].mode, inplace=True)  # Embarked -> modus (bug: harusnya .mode()[0])
```

**Sel 6**: import visualisasi (`matplotlib`, `seaborn`).

**Sel 7**: inti praktikum — membuat fitur baru.

```python
# 1. FamilySize: SibSp + Parch + 1 (diri sendiri)
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# 2. Title: gelar dari kolom Name
df["Title"] = df["Name"].str.extract(r',\s*([^\.]+)\.', expand=False)
title_mapping = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Rare", "Countess": "Rare", "Capt": "Rare", "Col": "Rare",
    "Don": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare",
    "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare"
}
df["Title"] = df["Title"].replace(title_mapping)

# 3. AgeBin: kelompokkan usia
df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100],
                      labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"])

# 4. CabinDeck: huruf pertama Cabin
df["CabinDeck"] = df["Cabin"].astype(str).str[0]

# 5. Fare per person: harga tiket / jumlah keluarga
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

print("✅ Hasil Ekstraksi fitur")
df[["Name", "Title", "Age", "AgeBin", "FamilySize", "Cabin", "CabinDeck", "FarePerPerson"]].head()
```

- **FamilySize** = `SibSp` (saudara/pasangan) + `Parch` (orang tua/anak) + 1 (penumpang itu sendiri). Berpotensi lebih prediktif daripada dua kolom terpisah.
- **Title** = gelar sosial dari nama (`Braund, Mr. Owen Harris` → `Mr`). Regex ``r',\s*([^\.]+)\.'``: cari koma, spasi, lalu **capture** (`(...)`) karakter apa pun selain titik (`[^\.]+`) sampai ketemu titik.
- `title_mapping` menyeragamkan gelar langka: `Mlle`/`Ms` → `Miss`, `Mme` → `Mrs`, sisanya (bangsawan/militer/dokter) → `Rare`. Tujuannya memperkecil jumlah kategori agar model tidak kewalahan.
- `.replace(mapping)` mengganti nilai sesuai kamus; nilai yang tidak ada di kamus dibiarkan.
- **AgeBin** = binning umur dengan `pd.cut`: interval default *right-closed* `(0,12]`, `(12,18]`, `(18,35]`, `(35,60]`, `(60,100]`, label sesuai. Binning membantu menangkap hubungan non-linear umur–keselamatan.
- **CabinDeck** = huruf pertama Cabin (`"C85"` → `"C"`). Deck kapal (A–G) punya makna posisi/lokasi kabin. `astype(str)` supaya aman dari NaN.
- **FarePerPerson** = `Fare / FamilySize`, koreksi karena harga tiket tercatat per kelompok keluarga, bukan per orang.
- `f'...'` dan ekspresi terakhir menampilkan tabel agar hasil bisa diverifikasi.

**Sel 8**: visualisasi fitur baru.

```python
plt.figure(figsize=(12, 6))
sns.barplot(x="FamilySize", y="Survived", data=df)
plt.title("Survival Rate berdasarkan Family Size")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x="Title", y="Survived", data=df,
            order=df.groupby("Title")["Survived"].mean().sort_values().index)
plt.title("Survival Rate berdasarkan Title")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x="AgeBin", y="Survived", data=df,
            order=["Child", "Teen", "YoungAdult", "Adult", "Senior"])
plt.title("Survival Rate berdasarkan Kelompok Usia (AgeBin)")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x="CabinDeck", y="Survived", data=df,
            order=df.groupby("CabinDeck")["Survived"].mean().sort_values().index)
plt.title("Survival Rate berdasarkan Cabin Deck")
plt.show()
```

- `sns.barplot` menghitung **rata-rata `Survived` per kategori**. Karena Survived = 0/1, rata-rata = **tingkat keselamatan (proporsi)**, dengan garis error 95% confidence interval.
- `order=...groupby(...).mean().sort_values().index` mengurutkan batang dari tingkat survival terendah ke tertinggi.
- `plt.xticks(rotation=45)` memutar label agar tidak bertumpuk.
- Untuk `AgeBin`, urutan dikunci manual supaya logis (Child → Senior), bukan urut alfabetis.

## 2. `Pertemuan3/Praktikum2.ipynb` — Pra-pemrosesan (versi manual)

**Sel 1**: import `pandas`, `LabelEncoder`, `StandardScaler`.

**Sel 2**: muat `data/Titanic-Dataset-fixed.csv`; `head()`.

**Sel 3**: encoding `Sex` dengan mapping manual.

```python
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
print(df[['Sex']].head(), "\n")
```

- `.map(dict)` mengganti nilai sesuai kamus. Dibanding `LabelEncoder`, cara ini **eksplisit** — kita sendiri yang menentukan `male=1`, `female=0` (LabelEncoder bergantung urutan alfabetis).
- `df[['Sex']]` menampilkan sebagai tabel (bukan Series).

**Sel 4**: bersihkan & ekstrak `Cabin`.

```python
df['Cabin'] = df['Cabin'].fillna('Unknown')
print(df[['Cabin']].head(5), "\n")

df['Cabin'] = df['Cabin'].apply(lambda x: x[0] if x != 'Unknown' else 'U')
print(df[['Cabin']].head(5), "\n")
```

- `fillna('Unknown')` menyiapkan kategori eksplisit untuk kabin tak diketahui.
- `.apply(lambda ...)` menjalankan fungsi ke setiap elemen kolom: ambil huruf pertama `x[0]` bila bukan `'Unknown'`; jika Unknown gunakan `'U'`. Hasil = deck kabin A–G/U.
- Dengan ini kolom Cabin berubah dari puluhan nilai unik menjadi ±8 kategori → lebih ringkas dan bermakna.

**Sel 5**: imputasi Age dengan median.

```python
df['Age'] = df[['Age']].fillna(df['Age'].median())
print(df[['Age']].head(10), "\n")
```

- Kali ini imputasi memakai **median**, bukan mean. Median lebih **robust terhadap outlier/skewness** sehingga sering lebih aman untuk Age (ada penumpang bayi dan usia sangat tua).
- `df[['Age']].fillna(...)` menghasilkan DataFrame baru, lalu di-assign balik ke `df['Age']`; tidak memakai `inplace=True` sehingga bebas FutureWarning.

**Sel 6**: standardisasi Age.

```python
scaler = StandardScaler()
df['Age'] = scaler.fit_transform(df[['Age']])
```

- Sama seperti sebelumnya: ubah Age ke skala z-score (mean 0, std 1).

**Sel 7–8**: verifikasi & pemilihan kolom akhir.

```python
print("=== Hasil Preprocessing (5 baris pertama) ===")
df.head()

df = df[['Survived', 'Pclass', 'Age', 'Sex', 'Cabin']]
df.head()
```

- Di akhir, dataset disiapkan hanya dengan 4 fitur + 1 target, siap dipakai modeling.

## 3. `Pertemuan3/Praktikum3.ipynb` — Split Data & K-Fold Cross Validation

### Bagian A: Train / Validation / Test Split (rasio 8:1:1)

**Sel 1**: muat `data/Titanic-Dataset-selected.csv` (dataset hasil preprocessing).

**Sel 2**: splitting.

```python
from sklearn.model_selection import train_test_split

df_train, df_unseen = train_test_split(df, test_size=0.2, random_state=0)
df_val, df_test = train_test_split(df_unseen, test_size=0.5, random_state=0)

print(f'Jumlah data asli: {df.shape[0]}')
print(f'Jumlah data train: {df_train.shape[0]}')
print(f'Jumlah data val: {df_val.shape[0]}')
print(f'Jumlah data test: {df_test.shape[0]}')

print('=========')
print(f'Jumlah label data asli:\n{df.Survived.value_counts()}')
...
```

- Split pertama: 80% train, 20% "unseen" (belum dilihat model). Split kedua: 20% tadi dibagi dua → 10% validasi + 10% test. Total **8:1:1**.
- Parameter `random_state=0` mengunci *seed* acak agar hasil split **reproducible** (diulang menghasilkan pembagian sama).
- Validasi dipakai untuk memilih/menyetel model; test hanya dipakai **sekali** di akhir untuk mengukur performa jujur.
- Output `value_counts()` menunjukkan proporsi kelas `Survived` di tiap subset — pada split acak biasa, proporsinya bisa sedikit berbeda dari data asli (sekitar 38% selamat).

**Sel 3**: muat ulang data ke `df2`.

**Sel 4**: splitting **dengan stratifikasi**.

```python
df2_train, df2_unseen = train_test_split(df2, test_size=0.2, random_state=0, stratify=df['Survived'])
df2_val, df2_test = train_test_split(df2_unseen, test_size=0.5, random_state=0, stratify=df_unseen['Survived'])

print(f'Jumlah label data asli:\n{df2.Survived.value_counts()}')
...
```

- `stratify=y` memaksa rasio kelas pada setiap subset **sama dengan data aslinya**. Penting untuk dataset tidak seimbang agar kelas minoritas tetap terwakili di train/val/test.
- **Catatan/perbaikan**: seharusnya memakai variabel yang benar — `stratify=df2['Survived']` dan `stratify=df2_unseen['Survived']`, bukan `df`/`df_unseen` dari sel sebelumnya. Kode lama tetap jalan karena isi `df` dan `df2` sama (dibaca dari file yang sama), tapi berpotensi membingungkan/bug jika file berbeda.

### Bagian B: K-Fold Cross Validation

**Sel 5**: muat ulang data ke `df3`.

**Sel 6**: K-Fold murni (tanpa hold-out test).

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=4)
print(f'Jumlah fold: {kf.get_n_splits()}')
print(f'Obyek KFold: {kf}')

kf_split = kf.split(df3)
print(f'Jumlah data df: {df.shape[0]}')

for train_index, test_index in kf_split:
    print(f'Index train: {train_index} | Index test: {test_index}')
```

- `KFold(n_splits=4)` membagi data menjadi 4 bagian (fold) berukuran sama. Setiap iterasi: 3 fold jadi train, 1 fold jadi test, bergantian sampai semua fold pernah menjadi test.
- `kf.split(df3)` menghasilkan pasangan **indeks** (bukan datanya) → `for train_index, test_index in ...` mencetak indeks untuk tiap fold.
- Kelebihan: setiap sampel dipakai untuk test tepat sekali dan untuk train k−1 kali → evaluasi lebih stabil daripada satu kali split, cocok untuk data kecil.

**Sel 7**: muat ulang data ke `df4`.

**Sel 8**: K-Fold tetap menyisakan data test.

```python
from sklearn.model_selection import train_test_split, KFold

df4_train, df4_test = train_test_split(df4, test_size=0.2, random_state=0)

kf2 = KFold(n_splits=4)
print(f'Jumlah fold: {kf2.get_n_splits()}')
print(f'Obyek KFold: {kf2}')

kf2_split = kf2.split(df_train)
print(f'Jumlah data df_train: {df4_train.shape[0]}')

for train_index, test_index in kf2_split:
    print(f'Index train: {train_index} | Index test: {test_index}')
```

- Strategi: hold-out 20% data test **dikunci sejak awal** untuk pengujian akhir; 80% sisanya dipakai dengan K-Fold (train + validasi bergantian). Ini memberi tuning model via cross-validation sekaligus kejujuran evaluasi akhir.
- **Catatan/perbaikan**: `kf2.split(df_train)` memakai variabel *stale* `df_train` dari Bagian A, padahal yang dimaksud `df4_train`. Ukurannya kebetulan sama sehingga berjalan tanpa error, tetapi sebaiknya ditulis `kf2.split(df4_train)`.

## 4. `Pertemuan3/Praktikum4.ipynb` — Histogram Citra (Pillow)

**Sel 1**: instalasi.

```python
!pip install Pillow
```

**Sel 2**: buka & tampilkan gambar.

```python
from PIL import Image

img = Image.open('data/Lenna.png')
img.show()      # membuka viewer gambar bawaan OS
display(img)    # metode alternatif: render gambar di dalam notebook
```

- `Image.open` membaca file gambar menjadi objek PIL (`PngImageFile`).
- `img.show()` membuka aplikasi penampil gambar eksternal.
- `display(img)` adalah fungsi IPython untuk menampilkan objek (termasuk gambar) langsung di output notebook.

**Sel 3**: ekstraksi channel & histogram.

```python
r, g, b = img.split()            # pisahkan channel Red, Green, Blue

print(len(r.histogram()))        # 256 -> jumlah bin intensitas
print(r.histogram())             # daftar 256 angka frekuensi piksel
```

- `img.split()` memecah gambar berwarna menjadi tiga gambar **grayscale** terpisah: `r`, `g`, `b` (masing-masing objek PIL).
- `r.histogram()` mengembalikan list berisi **256 angka**: elemen ke-i = jumlah piksel pada channel merah dengan intensitas `i` (0 = hitam, 255 = paling terang). `len(...)` = 256 karena kedalaman warna 8-bit.
- **Manfaat untuk ML**: histogram adalah *feature descriptor* citra yang ringkas — pola warna/terang gambar diringkas menjadi 3×256 angka dan bisa dibandingkan antar gambar (mis. klasifikasi, image retrieval) tanpa menyimpan seluruh piksel.

---

# Ringkasan Alur Praktikum

```
Muat data (pd.read_csv / cv2.imread / Image.open)
        │
        ▼
Inspeksi (shape, info, isnull, describe, visualisasi)
        │
        ▼
Bersihkan & imputasi (mean / median / modus / konstanta)
        │
        ▼
Ekstraksi fitur (FamilySize, Title, AgeBin, CabinDeck, histogram citra)
        │
        ▼
Encoding (LabelEncoder / map)  ──►  Standardisasi (StandardScaler)
        │
        ▼
Split data (train/val/test + stratify)
        │
        ▼
Validasi (K-Fold Cross Validation)
```

## Konsep Kunci

- **EDA**: memahami data sebelum memodelkan — distribusi, outlier (boxplot), komposisi kategori, korelasi (heatmap), hubungan antar variabel (scatter).
- **Missing value**: mean untuk numerik simetris, median untuk numerik dengan outlier, modus untuk kategorikal, konstanta/label baru ("DECK"/"Unknown"/"U") bila kehilangan data bermakna.
- **Encoding**: kategorikal → angka. `LabelEncoder`/`map` untuk data ordinal atau biner; one-hot lebih tepat untuk nominal (jangan beri urutan palsu, mis. Cabin).
- **Standardisasi**: z-score (mean 0, std 1) agar skala fitur seragam; fit di train saja untuk menghindari data leakage.
- **Train/val/test & stratify**: 8:1:1 dengan `random_state` agar reprodusibel; `stratify` menjaga proporsi kelas.
- **K-Fold CV**: evaluasi lebih stabil; kombinasikan dengan hold-out test untuk pengujian akhir.
- **Feature engineering**: fitur turunan yang lebih bermakna (FamilySize, Title, AgeBin, CabinDeck, FarePerPerson; resize/grayscale/histogram untuk citra).

## Catatan Perbaikan Kode (Bug & Best Practice)

| Lokasi | Masalah | Perbaikan |
|---|---|---|
| Pertemuan2/Praktikum2 sel 12; Pertemuan3/Praktikum1 sel 5 | `df['Embarked'].mode` tanpa `()` → mengisi dengan *method*, bukan nilai modus | `df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)` |
| Semua `df[col].fillna(..., inplace=True)` | FutureWarning *chained assignment* di pandas baru | `df[col] = df[col].fillna(...)` |
| Pertemuan2/Praktikum3 sel 7 | `LabelEncoder` pada `Cabin` (nominal) memberi urutan palsu | Gunakan one-hot encoding atau ekstrak deck (huruf pertama) |
| Pertemuan2/Praktikum3 sel 11; Tugas sel 12 | `fit_transform` pada seluruh dataset (data leakage) | `fit` di train, `transform` di val/test |
| Pertemuan2/Praktikum4 sel 8 | Konversi gambar RGB dengan `COLOR_BGR2GRAY` | Gunakan `cv2.COLOR_RGB2GRAY` |
| Pertemuan3/Praktikum3 sel 4 | `stratify=df['Survived']` dan `df_unseen[...]` (variabel lama) | `stratify=df2['Survived']` dan `df2_unseen['Survived']` |
| Pertemuan3/Praktikum3 sel 8 | `kf2.split(df_train)` (variabel lama) | `kf2.split(df4_train)` |
| Pertemuan2/Praktikum1 sel 19 | `surv_map` didefinisikan tapi tidak dipakai | Hapus atau pakai untuk pewarnaan titik berdasarkan survival |
| Pertemuan1/Praktikum1.ipynb | Notebook kosong tanpa sel | Isi atau hapus agar tidak membingungkan |
