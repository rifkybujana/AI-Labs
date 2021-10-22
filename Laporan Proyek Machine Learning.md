# Laporan Proyek *Machine Learning* - Rifky Bujana Bisri

## Domain Proyek

Telekomunikasi adalah salah satu bidang yang saat ini menopang kehidupan. Telekomunikasi adalah teknik pengiriman atau penyampaian informasi jarak jauh, dari suatu tempat ke tempat lain. informasi tersebut bisa berupa tulisan, suara, gambar, ataupun objek lainnya. Telekomunikasi menjadi penghubung antar manusia dalam berkomunikasi dari jarak jauh, salah satunya adalah situs *Community Question Answering* (*CQA*).

Situs *Community Question Answering* (*CQA*) menjadi salah satu jalan yang cukup populer untuk menyediakan dan mencari informasi. situs *CQA* menyediakan sebuah antarmuka untuk pengguna menukar dan berbagi pengetahuan. Pengguna menanyakan pertanyaan terhadap suatu topik yang kurang dia pahami, dan mencari seseorang yang ahli untuk memberikan informasi yang diinginkan (Baltadzhieva & Chrupala, 2015). Salah satu situs *CQA* yang cukup populer dan digunakan oleh banyak orang saat ini adalah *Stack Overflow*.

*Stack Overflow* adalah sebuah situ *CQA* dalam lingkup *computer programming*. Pada *Stack Overflow* jawaban dipilih berdasarkan kepuasan penanya. Penanya dapat menandai sebuah pertanyaan untuk menunjukkan kepada sebuah subjek spesifik. 

Namun, dengan meningkatnya popularitas *Stack Overflow*, jumlah pertanyaan yang tidak terjawab juga meningkat. Berdasarkan statistik pada 2012 saja, diperkirakan terdapat 45 pertanyaan setiap bulan yang tidak terjawab (Asaduzzaman, 2013). Pada Maret 2014, terdapat 752.533 dari 6.912.743 pertanyaan yang tidak terjawab (diperkirakan 10,9%) (Baltadzhieva & Chrupala, 2015). Yang menarik, pertanyaan yang tidak dijawab bukan disebabkan karena pengguna tidak melihatnya. Faktanya pertanyaan yang tidak terjawab rata-rata dilihat 139 kali (Asaduzzaman, 2013).

## Business Understanding

Seperti yang sudah dijelaskan sebelumnya, *Stack Overflow* memberikan kebebasan terhadap penggunanya untuk dapat bertanya melalui platformnya. Lalu, pertanyaan tersebut akan ditampilkan pada halaman utama di *Stack Overflow* kepada para pengguna yang memiliki kemampuan relevan dengan pertanyaan tersebut.

Namun terkadang pertanyaan yang diajukan melalui platform tersebut tidak dijawab oleh pengguna lainnya. Dan pertanyaan yang tidak dijawab tersebut bukan disebabkan tidak terlihat oleh pengguna lainnya. Oleh karena itu setiap pertanyaan perlu dianalisa apakah pertanyaan-pertanyaan yang ada di *Stack Overflow* untuk memberikan masukan apakah pertanyaan tersebut perlu ditutup, diedit atau pertanyaan tersebut cukup baik tanpa perlu dilakukan apapun.

### Problem Statements

- Bagaimana membuat sistem yang dapat menilai suatu pertanyaan pada *Stack Overflow* dengan baik?
- Apakah *machine learning* dapat menjadi salah satu solusi dalam menilai pertanyaan pada *Stack Overflow*?
- Model apa yang cocok dalam melakukan tugas ini?

### Goals

Membangun sebuah model *machine learning* yang dapat memprediksi kualitas dari suatu pertanyaan pada *Stack Overflow* dengan baik.

### Solution Statements

Solusi yang saya ajukan dalam menjawab permasalahan ini adalah dengan menggunakan model *multinomial logistic regression* atau *multionomial naive bayse* untuk melakukan prediksi awal terhadap pertanyaan pada *Stack Overflow*. Sehingga pertanyaan-pertanyaan pada *Stack Overflow* dapat memiliki kualitas yang cukup baik agar memudahkan pengguna lain dalam menjawab pertanyaan tersebut.

*Multinomial logistic regression* atau disebut juga model *logit politomus* adalah model regresi yang digunakan untuk menyelesaikan kasus regresi dengan variabel dependen berupa data kualitatif berbentuk *multinomial* (lebih dari dua kategori) dengan satu atau lebih variabel independen.

Metode *multinomial naive bayes* merupakan variasi lain dari *naive bayes*. Metode ini mengasumsikan bahwa semua atribut saling bergantung satu sama lain mengingat konteks kelas, dan mengabaikan semua dependensi antar atribut 

```
𝑃(𝐶) = 𝑐𝑜𝑢𝑛𝑡 (𝑐)+𝐾 / 𝑁+𝐾 . |𝑐𝑙𝑎𝑠𝑠𝑒𝑠|

dimana:
P : probabilitas dari variable c
Count (c) : jumlah kemunculan dari sampel c
K : nilai parameter
N : jumlah total kejadian dari sampel c
|𝑐𝑙𝑎𝑠𝑠𝑒𝑠| : jumlah kelas pada sampel 
```

Selanjutnya menghitung nilai atribut. Jumlah atribut yang sudah diberi kelas ditambah satu dan dibagi dengan jumlah kelas tertentu ditambah dengan hasil kali antara jumlah atribut dikali satu. Berikut merupakan persamaan untuk menghitung nilai `𝑃(𝑊|𝐶)` dengan menggunakan *laplacian smoothing*:

```
𝑃(𝑊|𝐶) = count + K / count(C) + |V|.1

di mana:
Count : jumlah kemunculan atribut pada kelas tertentu
K : nilai paramater
Count (c) : jumlah kemunculan kelas pada sampel c
N : jumlah total kejadian
|𝑣| : jumlah atribut pada sampel
```

## Data Understanding

Data set yang akan saya gunakan pada proyek ini merupakan dataset berisi enam puluh ribu pertanyaan pada *Stack Overflow*, lengkap dengan *tag* dan tanggal pertanyaan itu diterbitkan.

Dataset ini berisi 6 kolom berbeda: *Id*, *Title*, *Body*, *Tags*, *CreationDate*, dan *Y*. Kolom *Id* berisi identitas dari setiap baris untuk membedakan suatu baris dengan baris lainnya. Kolom *Title* berisi judul dari setiap pertanyaan. Kolom *Body* berisi detail ataupun isi dari pertanyaan yang diajukan dengan format *html*. *Tags* berisi *tags* yang berhubungan dengan pertanyaan tersebut. *CreationDate* berisi tanggal diunggahnya pertanyaan tersebut dengan format *UTC*. Dan kolom *Y* berisi tingkat kualitas pertanyaan yang dibagi menjadi 3, yaitu: *HQ*, *LQ_EDIT*, dan *LQ_CLOSE*.

*HQ* merupakan postingan pertanyaan berkualitas tinggi dengan total skor lebih dari tiga puluh dan tanpa diedit sekalipun. *LQ_EDIT* merupakan postingan berkualitas rendah dengan skor negatif dan diedit beberapa kali oleh komunitas namun tidak ditutup. Sedangkan *LQ_CLOSE* merupakan postingan berkualitas rendah yang ditutup oleh komunitas tanpa diedit sama sekali.

Data ini cukup seimbang untuk masing-masing kategori, dimana pada masing-masing kategori terdapat 20.000 data. 

![distribusi data](https://i.ibb.co/NZjRqdJ/download-4.png)

Jumlah kata pada masing-masing pertanyaan di dalam data cukup seimbang dengan nilai median masing-masing kategori sebesar 106, 85, dan 76.

![distribusi kata](https://i.ibb.co/YXPLPDp/download-5.png)

Lalu, jika kita menganalisa *unigram* pada setiap data dari masing-masing kategori. Terlihat 10 kata yang paling sering digunakan pada masing-masing kategori ada kesamaan. Masing-masing pertanyaan juga cukup bergantung pada penggunaan *stopwords*, hal ini yang menjadi pertimbangan oleh saya untuk tidak membuang *stopwords* dari data.

![unigram](https://i.ibb.co/v3RQk7D/download-6.png)

Hal tersebut juga cukup terlihat pada *bigram* masing-masing data pada setiap kategori.

![bigram](https://i.ibb.co/FbXrDhC/download-7.png)

[dataset](https://www.kaggle.com/imoore/60k-stack-overflow-questions-with-quality-rate)

## Data Preparation

Dalam memproses data sebelum digunakan untuk melatih model saya menggunakan beberapa tahapan. 

- *Label encoding*, pada tahapan ini label dari masing masing data di *encoding* menjadi data berupa angka: 0 (HQ), 1 (LQ_CLOSE), dan 2 (LQ_EDIT). Hal ini dilakukan agar label dapat dipahami oleh mesin.

- Kemudian data *title* dari masing masing data digabungkan dengan *body* dari data agar model mendapatkan informasi sebanyak mungkin dari data.

- *Data cleaning*, pada tahapan ini setiap huruf pada teks di data diubah menjadi huruf kecil, karena komputer melihat huruf kapital dan huruf kecil sangat berbeda. Lalu seperti yang sudah dijelaskan pada bagian *Data Understanding*, teks *body* dari data ini masih memiliki format *html* di dalamnya, oleh karena itu saya membersihkan setiap simbol dan *tag html* dari teks.

- *Tokenization*, pada tahapan ini setiap kata pada teks di dalam data dipisah menjadi bentuk token. Hal ini diperlukan agar model mendapatkan informasi dari masing-masing kata di dalam teks.

- *Embeddings*, pada tahapan ini data yang sudah diubah menjadi bentuk token akan diproses menjadi bentuk angka untuk mengoptimalkan model dalam berlatih maupun memprediksi. Dalam kasus ini saya menggunakan metode *TF-IDF* agar model dapat mengetahui distribusi suatu token di dalam teks. Hal tersebut juga memberikan model informasi seberapa penting suatu kata di dalam teks.

- Lalu data tersebut dibagi menjadi 2 bagian, yaitu *train dataset* dan *test dataset*. *Train dataset* digunakan untuk melatih model terhadap data, data set ini berisi 48.000 data (80% dari total data). Sedangkan *test dataset* digunakan untuk melakukan evaluasi kemampuan model terhadap data, data set ini berisi 12.000 data (20% dari total data).

## Modelling

Pada tahapan modelling, saya menggunakan dua buah model *machine learning*, yaitu *multinomial logistic regression* dan *naive bayes*. Lalu membandingkan hasil evaluasi dan prediksi dari keduanya. 

Kedua jenis model ini dilatih menggunakan *train dataset* dengan perulangan sebanyak seratus kali. Kedua model kemudian dievaluasi terhadap *test dataset* dan menghasilkan tingkat akurasi yang cukup baik. Model *multinomial logistic regression* menghasilkan tingkat akurasi sebesar 87.30%, sedangkan model *naive bayes* menghasilkan tingkat akurasi sebesar 86.37%.

Lalu saya melakukan analisa terhadap *confusion matrix* yang dihasilkan oleh model terhadap *test dataset*. Hal ini dilakukan untuk mengetahui seberapa baik model bekerja terhadap masing-masing kategori dan mendapatkan gambaran lebih jelas terhadap hasil prediksi dari model.

Pada model *multinomial logistic regression* terlihat bahwa model dapat memprediksi kategori 2 (LQ_EDIT) dengan sangat baik, tetapi model cukup kebingungan dalam memprediksi beberapa data kategori 0 (HQ) dan 1 (LQ_CLOSE).

Jika kita kembali melihat pada gambar *unigram* dan *bigram* kategori 0 (HQ) dan 1 (LQ_CLOSE) memang terlihat kemiripan diantara keduanya. Namun, jumlah data yang diprediksi oleh model dengan benar masih cukup baik.

![confusion matrix lr](https://i.ibb.co/rmNfnVL/download-2.png)

Hal ini juga dapat dilihat dari tingkat *recall*, *precision* dan *f beta score* dari kategori LQ_EDIT yang diprediksi oleh model lebih dari 98. Sedangkan pada kategori HQ dan LQ_CLOSE sebesar 81 hingga 82.

| Label    | precision | recall | f beta score |
| -------- | --------- | ------ | ------------ |
| HQ       | 82,62     | 82,37  | 82,49        |
| LQ_CLOSE | 81,17     | 81,37  | 81,27        |
| LQ_EDIT  | 98,10     | 98,15  | 98,12        |
* dalam skala 0 - 100

Lalu pada model *naive bayes* terlihat bahwa model dapat memprediksi kategori 2 (LQ_EDIT) dengan sangat baik, tetapi saat model memprediksi data kategori 0 (HQ) dan 1 (LQ_CLOSE) model cukup kebingungan dalam membedakan beberapa data kategori 0 (HQ) dan 1 (LQ_CLOSE). Dan model juga melakukan kesalahan pada data kategori 0 (HQ) dan 1 (LQ_CLOSE) dengan memprediksinya sebagai data kategori 2 (HQ_EDIT) walau lebih sedikit daripada antara kategori 0 (HQ) dan 1 (LQ_EDIT).

![confusion matrix lr](https://i.ibb.co/8dG6CsV/download-3.png)

Jika kita kembali melihat pada gambar *unigram* dan *bigram* kategori 0 (HQ) dan 1 (LQ_CLOSE) memang terlihat kemiripan diantara keduanya. Namun, jumlah data yang diprediksi oleh model dengan benar masih cukup baik.

Model *naive bayes* juga dapat memprediksi kategori LQ_EDIT dengan sangat baik dengan tingkat *recall*, *precision* dan *f beta score* antara 91 hingga 98. Sedangkan pada kategori HQ dan LQ_CLOSE memiliki nilai *recall*, *precision* dan *f beta score* antara 79 hingga 86.

| Label    | precision | recall | f beta score |
| -------- | --------- | ------ | ------------ |
| HQ       | 79,87     | 86,20  | 82,91        |
| LQ_CLOSE | 81,88     | 81,80  | 81,84        |
| LQ_EDIT  | 98,83     | 91,10  | 94,80        |
* dalam skala 0 - 100

Berdasarkan hal tersebut dapat disimpulkan bahwa model *multinomial logistic regression* dapat memprediksi data lebih baik dari pada model *multionmial naive bayes*. Hal ini dapat dilihat dari tingkat akurasi *multinomial logistic regression* yang sedikit lebih tinggi dari model *multinomial naive bayes*. Hal tersebut juga dapat dilihat dimana model *multinomial naive bayes* melakukan jauh lebih banyak kesalahan dalam memprediksi label HQ dan LQ_CLOSE menjadi label LQ_EDIT.

## Evaluation

Dalam melakukan evaluasi terhadap model saya menggunakan 4 metric berbeda, yaitu akurasi, *precision*, *recall*, dan *f beta score*. 

Yang pertama adalah akurasi, akurasi merupakan rasio prediksi Benar (positif dan negatif) dengan keseluruhan data.

```
Akurasi = (TP + TN ) / (TP+FP+FN+TN)
```

*Precision* merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan hasil yang diprediksi positf.

```
Precission = (TP) / (TP+FP)
```

*Recall* erupakan rasio prediksi benar positif dibandingkan dengan keseluruhan data yang benar positif. 

```
Recall = (TP) / (TP + FN)
```


*F beta score* sama seperti metrik *F1 score* yang merepresentasikan keseimbangan antara *precision* dan *recall*. Namun, pada *F beta score* kalkulasinya sedikit berbeda, dimana kita dapat menentukan bobot keseimbangan antara *precision* dan *recall* melalui parameter *beta* (*β*). Untuk lebih jelasnya bisa dilihat perbedaan pada *F1 score* dan *F beta score* berikut.

*F1 Score*:
```
F1 = (2 * precision * recall) / (precision + recall)
```

*F beta score*:
```
Fβ = ((1 + β^2) . (precision . recall)) / ((β^2 . precision) + recall))
```

**---Ini adalah bagian akhir laporan---**