# Laporan Proyek Machine Learning - Haldies Gerhardien Pasya

## Domain Proyek

​Dalam era digital saat ini, analisis risiko kredit menjadi aspek krusial dalam proses persetujuan pinjaman. Dengan meningkatnya permintaan akan pinjaman, lembaga keuangan menghadapi tantangan dalam menilai kelayakan kredit pemohon secara akurat dan efisien. Kesalahan dalam penilaian ini dapat menyebabkan kerugian finansial yang signifikan akibat gagal bayar.​
Untuk mengatasi tantangan tersebut, berbagai pendekatan berbasis teknologi telah dikembangkan, salah satunya dengan memanfaatkan metode pembelajaran mesin (machine learning). Penelitian menunjukkan bahwa model pembelajaran mesin, seperti XGBoost dan Random Forest, dapat mencapai akurasi tinggi dalam memprediksi risiko kredit. Misalnya, studi oleh Fekadu et al. (2022) menemukan bahwa XGBoost mencapai skor F1 tertinggi dalam memprediksi pinjaman bermasalah, dengan fitur penting termasuk usia pemohon, tahun pengalaman kerja, dan total pendapatan [1]. ​
arXiv Selain itu, pendekatan hibrida yang menggabungkan pembelajaran mesin dan pembelajaran mendalam telah menunjukkan peningkatan kinerja dalam prediksi persetujuan kartu kredit. Tong et al. (2024) mengembangkan kerangka kerja yang mengintegrasikan berbagai model untuk meningkatkan akurasi dan keandalan prediksi [2].

## Business Understanding

proses klarifikasi masalah dimulai dengan memahami tujuan utama bisnis, yaitu meningkatkan akurasi dan efisiensi dalam proses persetujuan pinjaman. Lembaga keuangan membutuhkan sistem yang mampu memprediksi risiko kredit secara tepat agar dapat meminimalkan potensi gagal bayar. Dengan memanfaatkan data historis pemohon dan informasi terkait pinjaman, kita dapat membangun model prediktif yang membantu dalam pengambilan keputusan. Hal ini penting untuk mendukung proses bisnis yang lebih objektif, cepat, dan berbasis data.

### Problem Statements
Berdasarkan latar belakang diatas, proyek ini berfokus pada beberapa masalah utama yang perlu dipecahkan:

1. Lembaga keuangan sering mengalami kesulitan dalam mengidentifikasi calon peminjam yang berisiko tinggi karena keterbatasan dalam proses penilaian manual dan tidak adanya sistem prediksi yang andal.

2. Data historis kredit seringkali memiliki ketidakseimbangan kelas (class imbalance), di mana jumlah peminjam yang disetujui jauh lebih banyak dibandingkan yang ditolak, yang dapat memengaruhi performa model klasifikasi.

3. Pemilihan fitur yang tidak tepat dan kurangnya optimasi model sering kali menyebabkan hasil prediksi yang tidak akurat dalam proses penilaian risiko pinjaman.

### Goals
Tujuan dibuat proyek ini adalah sebagai barikut:

1. Mengembangkan sistem berbasis machine learning yang mampu memprediksi status persetujuan pinjaman (loan_status) secara otomatis dan akurat dengan menggunakan variabel-variabel relevan dari data pemohon.
   
2. Mengatasi ketidakseimbangan data dengan teknik seperti Synthetic Minority Oversampling Technique for Nominal and Continuous (SMOTENC), sehingga model dapat mengenali pola dari kedua kelas secara seimbang.

3. Meningkatkan akurasi model dengan melakukan pemilihan fitur (feature selection) yang optimal dan menerapkan teknik hyperparameter tuning untuk memaksimalkan performa model.

### Solution statements
- Menerapkan beberapa algoritma klasifikasi seperti Logistic Regression, Random Forest, dan XGBoost untuk membandingkan performa dalam memprediksi loan_status. Evaluasi dilakukan menggunakan metrik seperti akurasi, precision, recall, dan F1-score untuk memastikan keandalan model.

- Mengimplementasikan teknik feature engineering dan feature selection berdasarkan korelasi dan importance score dari masing-masing fitur terhadap target variabel.

- Melakukan hyperparameter tuning pada model terbaik menggunakan metode seperti Grid Search atau Random Search untuk meningkatkan performa prediksi.

- Menangani ketidakseimbangan data dengan SMOTENC agar distribusi kelas target menjadi lebih proporsional, sehingga model tidak bias terhadap kelas mayoritas dan dapat memberikan prediksi yang lebih adil.

## Data Understanding

Dataset yang digunakan dalam proyek ini merupakan versi sintetis dari data risiko kredit yang terinspirasi dari *Credit Risk Dataset* di Kaggle. Data ini diperluas dengan variabel tambahan yang relevan dengan penilaian risiko keuangan dan telah melalui proses augmentasi menggunakan teknik **SMOTENC (Synthetic Minority Oversampling Technique for Nominal and Continuous)**. Dataset ini tersedia untuk diunduh melalui platform [Kaggle Credit Risk Synthetic Dataset](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data). Dataset ini terdiri dari 45.000 entri dan 14 variabel.

### Variabel-variabel pada dataset ini adalah sebagai berikut:

- **person_age** : Merupakan usia dari peminjam.  
- **person_gender** : Jenis kelamin dari peminjam.  
- **person_education** : Tingkat pendidikan terakhir dari peminjam.  
- **person_income** : Pendapatan tahunan dari peminjam.  
- **person_emp_exp** : Lama pengalaman kerja dalam tahun.  
- **person_home_ownership** : Status kepemilikan rumah peminjam, misalnya *rent*, *own*, atau *mortgage*.  
- **loan_amnt** : Jumlah pinjaman yang diminta oleh peminjam.  
- **loan_intent** : Tujuan peminjaman, seperti untuk *debt consolidation*, *education*, *medical*, dll.  
- **loan_int_rate** : Persentase suku bunga yang dikenakan terhadap pinjaman.  
- **loan_percent_income** : Persentase jumlah pinjaman terhadap pendapatan tahunan peminjam.  
- **cb_person_cred_hist_length** : Panjang riwayat kredit peminjam dalam tahun.  
- **credit_score** : Skor kredit peminjam, mencerminkan kelayakan mereka dalam mendapatkan pinjaman.  
- **previous_loan_defaults_on_file** : Indikator apakah peminjam pernah mengalami gagal bayar pada pinjaman sebelumnya. 
- **loan_status** *(target variable)* : Status persetujuan pinjaman. Nilai 1 menandakan pinjaman disetujui, dan 0 menandakan pinjaman ditolak.

---
## 🧾 visual dataset Struktur Dataset

![alt text](https://raw.githubusercontent.com/haldies/machine_terapan_h/main/image/StrukturDataset.png)
---
Tipe Data: Secara keseluruhan, tipe data sudah sesuai.

## 📈 Statistik Deskriptif (Numerik)

![alt text](https://raw.githubusercontent.com/haldies/machine_terapan_h/main/image/Deskriptif.png)
---
Usia (person_age): Nilai maksimal usia adalah 144, yang tidak realistis. Ini menunjukkan adanya outlier yang perlu diperiksa lebih lanjut.

Pendapatan (person_income): Nilai maksimal yang sangat tinggi (7.2 juta), yang mungkin merupakan outlier. Biasanya, pendapatan ekstrem ini perlu dikoreksi.

Pengalaman Kerja (person_emp_exp): Pengalaman kerja maksimal yang sangat tinggi (125 tahun) jelas merupakan outlier yang tidak wajar.

## 🔍 Cek Missing Value

| Parameter                      | Nilai |
| ------------------------------ | ----- |
| person_age                     | 0     |
| person_gender                  | 0     |
| person_education               | 0     |
| person_income                  | 0     |
| person_emp_exp                 | 0     |
| person_home_ownership          | 0     |
| loan_amnt                      | 0     |
| loan_intent                    | 0     |
| loan_int_rate                  | 0     |
| loan_percent_income            | 0     |
| cb_person_cred_hist_length     | 0     |
| credit_score                   | 0     |****
| previous_loan_defaults_on_file | 0     |
| loan_status                    | 0     |


Tidak ada data yang memiliki nilai hilang (missing value).

## 📊 Visualisasi Distribusi Data
![alt text](https://raw.githubusercontent.com/haldies/machine_terapan_h/main/image/Distribusi_Data.png)
.
1. **Distribusi Umur Pemohon (`person_age`)**
**Insight**:  
🔎 Sebagian besar peminjam masih muda, di rentang usia produktif (20–30 tahun). Tapi perlu pembersihan data untuk outlier umur >100 tahun.

---

2. **Distribusi Pengalaman Kerja (`person_emp_exp`)**
**Insight**:  
🔎 Banyak peminjam adalah pekerja awal karir (1–8 tahun pengalaman). Perlu menghapus pengalaman kerja tidak logis (di atas 60 tahun).

---

3. **Distribusi Jumlah Pinjaman (`loan_amnt`)**
**Insight**:  
🔎 Sebagian besar pinjaman berada di kisaran kecil hingga menengah, cocok untuk pinjaman pribadi atau kebutuhan konsumtif.

---
4. **Distribusi Suku Bunga Pinjaman (`loan_int_rate`)**
**Insight**:  
🔎 Rentang bunga cukup wajar untuk kredit konsumer. Mayoritas bunga berada antara **8%–13%**, artinya tidak terlalu mahal.

---

5. **Distribusi Rasio Pinjaman terhadap Pendapatan (`loan_percent_income`)**
**Insight**:  
🔎 Kebanyakan peminjam hanya menggunakan sebagian kecil pendapatannya untuk membayar pinjaman, yang berarti beban cicilan masih **cukup aman**.

---

6. **Distribusi Panjang Riwayat Kredit (`cb_person_cred_hist_length`)**
**Insight**:  
🔎 Banyak peminjam baru memiliki riwayat kredit pendek (sekitar 3–8 tahun), menandakan banyaknya peminjam generasi muda atau baru mulai membangun kredit.

---

7. **Distribusi Skor Kredit (`credit_score`)** 
🔎 Skor rata-rata ada di **kategori wajar** (600–700), tapi ada juga yang rendah (di bawah 600) yang perlu perhatian lebih karena berisiko default.

---

## 📊 Melihat visualisasi Distribusi Data Kategorikal
![alt text](https://raw.githubusercontent.com/haldies/machine_terapan_h/main/image/Data_Kategorikal.png)

- **Distribusi Gender (person_gender)**
🔎 Jumlah peminjam laki-laki sedikit lebih banyak dibanding perempuan. Ini menunjukkan bahwa partisipasi dalam pengajuan pinjaman didominasi laki-laki, tapi perbedaannya tidak terlalu jauh.

- **Distribusi Pendidikan (person_education)**
🔎 Mayoritas peminjam memiliki pendidikan minimal SMA hingga S1. Jumlah yang berpendidikan Master dan Doctorate cukup kecil, yang wajar karena semakin tinggi pendidikan, biasanya profil peminjam lebih sedikit.

- **Distribusi Kepemilikan Rumah (person_home_ownership)**
🔎 Sebagian besar peminjam tinggal di rumah sewa atau sedang dalam cicilan KPR. Ini bisa berarti banyak peminjam berada dalam tahap membangun aset dan belum memiliki rumah pribadi sepenuhnya.

- **Distribusi Default Pinjaman Sebelumnya (previous_loan_defaults_on_file)**
🔎 Jumlah peminjam yang pernah mengalami gagal bayar hampir sama dengan yang tidak. Ini mengindikasikan dataset mengandung risiko yang cukup besar dan perlu perhatian dalam analisis risiko pinjaman.

- **Distribusi Status Pinjaman (loan_status)**
🔎 Distribusi data pada kolom loan_status menunjukkan bahwa data pinjaman sekitar (77,8%) tidak disetujui (loan_status = 0) dan data pinjaman sekitar (22,2%) yang disetujui (loan_status = 1). Hal ini menunjukkan ketidakseimbangan data (data imbalance), yang penting diperhatikan saat melakukan analisis atau pelatihan model prediksi, karena model bisa cenderung memprediksi mayoritas kelas.


## 🔗 Tabel Korelasi Antar Variabel Numerik
![alt text](Korelasi.png)

Berdasarkan matriks korelasi di atas, variabel yang memiliki hubungan paling kuat dengan loan_status (status kelolosan pinjaman) adalah loan_percent_income (0.38) dan loan_int_rate (0.33), yang artinya semakin tinggi persentase pinjaman terhadap pendapatan atau suku bunga, semakin besar kemungkinan pengajuan pinjaman lolos. Sementara variabel seperti person_age, person_income, dan credit_score memiliki korelasi negatif lemah terhadap loan_status, menunjukkan pengaruh yang kecil. Secara umum, tidak ada korelasi yang sangat kuat (mendekati 1 atau -1), namun dua variabel tadi bisa menjadi kandidat penting dalam analisis prediksi status pinjaman.

## 📊 Distribusi Status Pinjaman

![alt text](https://raw.githubusercontent.com/haldies/machine_terapan_h/main/image/loan_status.png)
Distribusi data pada kolom loan_status menunjukkan bahwa data pinjaman sekitar (77,8%) tidak disetujui (loan_status = 0) dan data pinjaman sekitar (22,2%) yang disetujui (loan_status = 1). Hal ini menunjukkan ketidakseimbangan data (data imbalance), yang penting diperhatikan saat melakukan analisis atau pelatihan model prediksi, karena model bisa cenderung memprediksi mayoritas kelas.

## Data Preparation
Pada bagian ini, dilakukan beberapa tahapan preprocessing data untuk memastikan data yang digunakan dalam pemodelan bersih, seimbang, dan siap untuk dilatih. Adapun tahapan-tahapan data preparation yang dilakukan secara berurutan adalah sebagai berikut:

### 1. Penanganan Outlier
Beberapa variabel ditemukan memiliki nilai ekstrem yang tidak realistis. Karena jumlah data outlier ini sangat sedikit (kurang dari 10 data), maka diputuskan untuk menghapus (drop) data tersebut agar tidak memengaruhi hasil analisis. Adapun outlier ditemukan pada fitur berikut:

- person_age: Nilai maksimal mencapai 144 tahun.

- person_emp_exp: Pengalaman kerja hingga 125 tahun.

- person_income: Pendapatan tahunan yang sangat tinggi (hingga 7.2 juta).
  
Penghapusan data ekstrem ini dilakukan agar model tidak bias terhadap data yang tidak representatif.

---

### 2. Label Encoding Data  
Fitur kategorikal perlu dikonversi terlebih dahulu menjadi nilai numerik agar dapat digunakan dalam algoritma machine learning. Proses ini dilakukan menggunakan **Label Encoding**, yaitu teknik yang mengubah setiap kategori unik menjadi representasi angka. Contohnya, fitur `"gender"` dengan nilai `"male"` dan `"female"` akan dikonversi menjadi `0` dan `1`.

Langkah ini **sangat penting dilakukan sebelum proses Data Balancing menggunakan SMOTENC**, karena:

- **SMOTENC tidak dapat memproses data dalam bentuk string atau objek kategori.**
- SMOTENC membutuhkan seluruh data, baik numerik maupun kategorikal, dalam format numerik agar dapat melakukan proses oversampling.
- Setelah Label Encoding dilakukan, fitur kategorikal yang telah dikonversi menjadi angka dapat ditandai dalam parameter `categorical_features` pada SMOTENC, sehingga metode ini tetap memperlakukan fitur tersebut sebagai kategorikal, bukan numerik biasa.

Dengan demikian, **Label Encoding dilakukan sebelum Data Balancing** agar data siap diproses oleh SMOTENC secara optimal dan menjaga struktur asli dari fitur kategorikal dalam proses pembangkitan data sintetis.

---

### 3. Penyeimbangan Data (Data Balancing)
Distribusi variabel target loan_status dalam dataset sangat tidak seimbang, dengan sekitar 77,8% data berada pada kelas ditolak (0) dan hanya 22,2% disetujui (1), sehingga model cenderung bias ke kelas mayoritas. Untuk mengatasi hal ini, digunakan teknik SMOTENC (Synthetic Minority Oversampling Technique for Nominal and Continuous) karena dataset mengandung kombinasi fitur kategorikal dan numerik. Pertama, fitur input (X) dan target (y) dipisahkan, lalu fitur kategorikal ditentukan berdasarkan indeks kolom. Dengan SMOTENC, data kelas minoritas (1) di-oversample hingga mencapai 15.000 data. Selanjutnya, digunakan RandomUnderSampler untuk mengurangi data dari kelas mayoritas (0) menjadi 15.000 juga, sehingga kedua kelas seimbang. Data hasil balancing ini kemudian digabungkan kembali ke dalam DataFrame baru df_resampled yang memiliki distribusi kelas 0 dan 1 masing-masing sebanyak 15.000, dan siap digunakan untuk pelatihan model secara lebih adil.

### 4. Split Data  
Pada tahap ini, dataset dibagi menjadi fitur dan target dengan perintah:

```python
X = df_resampled.drop(columns=['loan_status'])  
y = df_resampled['loan_status']
```

- **Fitur (X)**: Data yang digunakan untuk melatih model.  
- **Target (y)**: Label yang ingin diprediksi, yaitu status pinjaman.

Selanjutnya, dilakukan pembagian data menjadi **data pelatihan** dan **data pengujian** menggunakan fungsi `train_test_split` dengan proporsi **80:20**. Pembagian ini bertujuan agar model dapat belajar dari data yang tersedia dan diuji pada data yang belum pernah dilihat, guna menghindari *overfitting* dan memperoleh evaluasi performa yang objektif. Parameter `random_state=42` digunakan untuk memastikan hasil pembagian data konsisten setiap kali dijalankan.

---

## 🧠 Modeling
Dalam proyek ini, digunakan tiga algoritma machine learning untuk menyelesaikan permasalahan klasifikasi status pinjaman, yaitu **Logistic Regression**, **Random Forest Classifier**, dan **XGBoost Classifier**. Pemilihan ketiga algoritma ini didasarkan pada pertimbangan variasi kompleksitas, kemampuan generalisasi, serta efisiensi komputasi.

---

### 📌 **Alasan Pemilihan Algoritma**

- **Logistic Regression** dipilih sebagai baseline karena merupakan algoritma klasifikasi linear yang sederhana dan cepat. Meski tidak mampu menangkap hubungan non-linear antar fitur, model ini efektif pada dataset yang seimbang dan bersih.
  
- **Random Forest Classifier** merupakan algoritma ensemble berbasis pohon keputusan. Ia mampu menangani relasi non-linear dan cukup robust terhadap overfitting karena menggunakan teknik bagging dan rata-rata dari banyak decision tree.

- **XGBoost** merupakan algoritma boosting yang sangat populer karena performa akurasi yang tinggi, efisiensi dalam komputasi, serta memiliki kemampuan regularisasi yang baik untuk menghindari overfitting. Algoritma ini juga sangat fleksibel dan cocok untuk dataset besar dan kompleks.

---

### ⚙️ **Parameter Awal dan Alasan Pemilihan**

- **Logistic Regression**
  - `max_iter=1000`: Digunakan untuk memastikan proses konvergensi selesai, terutama jika jumlah fitur cukup banyak atau data tidak terstandarisasi dengan sempurna.
  - `random_state=42`: Digunakan untuk memastikan hasil yang konsisten dan reproducible.

- **Random Forest**
  - `random_state=42`: Untuk memastikan hasil eksperimen dapat direproduksi.

- **XGBoost**
  - `use_label_encoder=False`: Dinonaktifkan karena label encoder bawaan XGBoost sudah deprecated.
  - `eval_metric='logloss'`: Dipilih karena permasalahan ini adalah klasifikasi biner, dan logloss merupakan metrik loss yang umum digunakan pada klasifikasi biner untuk mengukur probabilitas prediksi.
  - `random_state=42`: Konsistensi hasil eksperimen.

---

### 🔍 **Hyperparameter Tuning dan Rasionalisasi**

Untuk meningkatkan performa, dilakukan proses **GridSearchCV** terhadap kombinasi parameter yang relevan. Berikut alasan pemilihan parameter untuk masing-masing model:

- **Logistic Regression**
  - `C`: Mengontrol strength dari regularisasi. Nilai kecil berarti regularisasi kuat (model lebih sederhana), nilai besar berarti model lebih kompleks.
  - `solver`: Memilih algoritma optimasi. Misalnya, 'lbfgs' cocok untuk dataset kecil hingga menengah.
  - `penalty`: Untuk memilih jenis regularisasi, seperti `l2` atau `elasticnet`.
  - `max_iter`: Untuk memastikan solver mencapai konvergensi.

- **Random Forest**
  - `n_estimators`: Jumlah pohon dalam hutan. Semakin banyak pohon bisa meningkatkan akurasi, tapi juga meningkatkan waktu komputasi.
  - `max_depth`: Membatasi kedalaman pohon untuk menghindari overfitting.
  - `min_samples_split` & `min_samples_leaf`: Mengontrol pembagian node agar tidak terlalu spesifik terhadap data latih.
  - `max_features`: Menentukan jumlah fitur yang dipertimbangkan saat split untuk mengurangi korelasi antar pohon.

- **XGBoost**
  - `n_estimators`: Jumlah boosting rounds. Lebih banyak iterasi bisa meningkatkan akurasi jika tidak overfit.
  - `max_depth`: Mengontrol kompleksitas model. Depth terlalu tinggi bisa overfit.
  - `learning_rate`: Mengontrol ukuran langkah setiap boosting step. Nilai kecil membuat training lebih lambat tapi bisa mencapai generalisasi lebih baik.
  - `subsample`: Proporsi sampel yang digunakan untuk melatih setiap pohon. Membantu mengurangi overfitting.
  - `colsample_bytree`: Proporsi fitur yang dipilih untuk setiap pohon. Membantu diversifikasi model.

---

### 🏆 **Pemilihan Model Terbaik**

Setelah dilakukan tuning, performa terbaik ditunjukkan oleh **XGBoost** dengan akurasi 93% dan F1-score yang tinggi serta seimbang di kedua kelas. Meskipun Logistic Regression dan Random Forest juga memberikan hasil yang baik, peningkatan performa setelah tuning tidak terlalu signifikan. 

XGBoost unggul karena:

- Memiliki **kemampuan regularisasi** (`lambda`, `alpha`) yang membantu menghindari overfitting.
- Dapat menangani dataset besar dan kompleks dengan baik.
- Memberikan **hasil evaluasi terbaik secara konsisten**, baik sebelum maupun sesudah tuning
---

## 📊 **Evaluation**
### ✅ **1. Metrik Evaluasi yang Digunakan**

Dalam proyek klasifikasi ini, digunakan beberapa metrik evaluasi untuk menilai performa model:

- **Accuracy**: Proporsi prediksi yang benar dari seluruh data.
- **Precision**: Kemampuan model dalam menghindari kesalahan prediksi positif.
- **Recall**: Kemampuan model dalam menangkap semua data positif.
- **F1 Score**: Rata-rata harmonik antara precision dan recall, cocok untuk dataset yang tidak seimbang.

Formulasi masing-masing metrik:
- \(\text{Precision} = \frac{TP}{TP + FP}\)
- \(\text{Recall} = \frac{TP}{TP + FN}\)
- \(\text{F1 Score} = 2 \times \frac{Precision \times Recall}{Precision + Recall}\)

---

### 🤖 **2. Hasil Evaluasi Model**

| **Model**               | **Accuracy** | **Precision (1)** | **Recall (1)** | **F1 Score (1)** |
| ----------------------- | ------------ | ----------------- | -------------- | ---------------- |
| **Logistic Regression** | 0.88         | 0.84              | 0.93           | 0.88             |
| **Random Forest**       | 0.91         | 0.89              | 0.94           | 0.91             |
| **XGBoost**             | 0.93         | 0.92              | 0.94           | 0.93             |

Keterangan:
- **(1)** mengacu pada kelas positif (`loan_status = 1`, pinjaman disetujui).

---

### 🔍 **3. Interpretasi dan Analisis**

- **Logistic Regression**:
  - Memiliki **recall tertinggi (93%)** pada kelas 1, artinya sangat baik dalam menangkap data peminjam yang seharusnya disetujui.
  - Namun, **precision-nya lebih rendah (84%)**, menunjukkan masih ada false positive yang cukup banyak.
  - Cocok jika tujuan utama adalah **mengurangi risiko kehilangan nasabah potensial**, meski dengan konsekuensi beberapa pinjaman disetujui secara salah.

- **Random Forest**:
  - Performa lebih seimbang, dengan **f1-score = 0.91** untuk kelas 1.
  - **Precision dan recall cukup tinggi dan seimbang**, menunjukkan model lebih stabil dalam mengklasifikasi kedua kelas.
  - Cocok untuk penggunaan di dunia nyata yang membutuhkan keseimbangan antara resiko dan peluang.

- **XGBoost**:
  - Memiliki performa terbaik secara keseluruhan, dengan **akurasi tertinggi (93%)** dan **f1-score 0.93** untuk kelas 1.
  - Sangat andal dalam menilai dengan presisi dan menangkap data yang relevan.
  - Cocok untuk solusi akhir yang mengutamakan akurasi tinggi dan performa optimal.

---

### 🧠 **4. Kesimpulan Akhir Evaluasi**

Model **XGBoost** memberikan hasil terbaik di antara ketiganya dan sangat direkomendasikan untuk digunakan dalam kasus klasifikasi pinjaman ini. Namun, dalam praktik, pilihan model tetap perlu mempertimbangkan **kemudahan interpretasi**, **kompleksitas**, serta **kebutuhan bisnis**.

Jika ingin model yang lebih cepat dan sederhana, Logistic Regression bisa dipilih. Jika ingin model yang andal dan stabil, Random Forest adalah opsi yang aman. Namun jika ingin **hasil optimal dengan akurasi dan f1-score terbaik**, maka XGBoost adalah pilihan utama.

---


## Daftar Pustaka
[1]R. Fekadu, A. Getachew, Y. Tadele, N. Ali, and I. Goytom, “Machine Learning Models Evaluation and Feature Importance Analysis on NPL Dataset,” arXiv.org, 2022. https://arxiv.org/abs/2209.09638 (accessed Apr. 14, 2025).
‌


[2]K. Tong, Z. Han, Y. Shen, Y. Long, and Y. Wei, “An Integrated Machine Learning and Deep Learning Framework for Credit Card Approval Prediction,” arXiv.org, 2024. https://arxiv.org/abs/2409.16676
‌
