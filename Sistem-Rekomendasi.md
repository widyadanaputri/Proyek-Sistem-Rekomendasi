# Proyek Sistem Rekomendasi (Ardiyanti Widyadana Prastuti)
## M03

Implementasi pembelajaran modul Machine Learning Terapan

## Project Overview

Perkembangan internet mendorong laju perkembangan servis daring  yang dapat diakses masyarakat luas dengan mudah, seperti movie.  Perkembangan ini juga diikuti berkembangnya data yang dihasilkan dari kegiatan pengguna. Data-data tersebut dinilai sangat berharga karena setelah diolah, dapat ditarik sebuah pola yang dilakukan pengguna dan hal ini dapat mendorong laju bisnis. Karena pola ini dapat diprediksi maka muncullah kecerdasan buatan yang disebut sistem rekomendasi.
Sistem rekomendasi adalah sistem yang dapat memberikan saran atau sugesti kepada user mengenai suatu informasi, contohnya film, buku, musik, berita dan lain-lain. Saran yang diberikan berdasarkan data yang didapatkan dari profil user seperti riwayat belanja, rating, dan lain-lain, sehingga saran yang diberikan dapat sesuai dengan selera dari pengguna tersebut. Dalam penelitian ini dijelaskan bagaimana penerapan salah satu metode yang digunakan dalam sistem rekomendasi, yaitu *Collaborative Filtering*. *Collaborative Filtering* memanfaatkan data rating yang diberikan oleh *user* terhadap suatu *item*, lalu data tersebut akan dibandingkan dengan pengguna lain untuk dicari kemiripannya. Penelitian ini dilakukan untuk mengukur tingkat akurasi prediksi dari sistem rekomendasi dengan menggunakan metode *Collaborative Filtering*.


## Business Understanding

1. Problem statements
Dari *background* di atas dapat disimpulkan beberapa rumusan masalah di antaranya:
- Bagaimana cara meningkatkan *user experience* saat mencari film untuk ditonton?
- Bagaimana cara membuat suatu sistem rekomendasi film dengan menggunakan pendekatan *collaborative filtering* atau filter kolaboratif?
2. Goals
Tujuan dari proyek ini adalah:
- Dapat meningkatkan *user experience* atau pengalaman pengguna saat mencari film untuk ditonton.
- Dapat menerapkan pendekatan *collaborative filtering* atau pemfilteran kolaboratif untuk membuat suatu sistem rekomendasi film.

3. Solution Statements
Dari *problem statements* dan *goals* yang telah dijabarkan, berikut solusi yang dapat dilakukan:
- Karena kumpulan data yang digunakan hanya berisi peringkat pengguna dan genre film, pada tugas ini akan menggunakan pendekatan *collaborative filtering* atau penyaringan kolaboratif untuk membuat suatu sistem rekomendasi film. 
- Dalam *collaborative filtering* atau pemfilteran kolaboratif, atribut yang digunakan adalah perilaku pengguna, bukan konten. Misalnya, merekomendasikan *item* berdasarkan riwayat penilaian pengguna atau pengguna lain.

## Data Understanding
Dataset yang digunakan pada proyek ini adalah dataset yang diambil dari website https://www.kaggle.com/datasets/rajmehra03/movielens100k. Dataset yang digunakan berbentuk .csv dengan total 4 file .csv. Tetapi yang digunakan hanya 3 dari 4 file .csv.
Pada movies.csv terdiri dari 9123 *rows* dan 3 *columns*.
Keterangan:
1. movieId : berisi *unique id* yang tersedia untuk tiap movie
2. title : berisi nama-nama movie beserta tahun dalam tanda kurung
3. genres : genre dari movie tersebut

Pada ratings.csv terdiri dari 10005 rows dan 4 columns.
Keterangan:
1. user id : berisi *unique id* yang tersedia untuk tiap *user*
2. movieId : berisi *unique id* yang tersedia untuk tiap movie
3. rating : berisi penilaian oleh *user* mengenai movie
4. timestamp : kode waktu dari movie tersebut

Pada tags.csv terdiri dari 1297 rows dan 4 columns.
Keterangan:
1. user id : berisi *unique id* yang tersedia untuk tiap *user*
2. movieId : berisi *unique id* yang tersedia untuk tiap movie
3. tag : berisi metadata yang telah dibuat oleh user mengenai movie
4. timestamp : kode waktu dari movie tersebut

Berikut *overview* dataset movies.csv yang telah dijadikan dataframe.
movie_df : dataset movies.csv yang dapat dilihat pada tabel 1.

Tabel 1. *Overview* movie_df
| movieId | title	                                   | genres                                      |
|---------|------------------------------------------|---------------------------------------------|
| 1       | Toy Story (1995)                         | Adventure-Animation-Children-Comedy-Fantasy |
| 2       | Jumanji (1995)                           | Adventure-Children-Fantasy                  |
| 3       | Grumpier Old Men (1995)	                 | Comedy-Romance                              |
| 4       | Waiting to Exhale (1995)	                 | Comedy-Drama-Romance                              |
| ......  | ........................................ | ........................................... |
| 162672  | Mohenjo Daro (2016)		                           | Adventure-Drama-Romance                                       |
| 163056  | Shin Godzilla (2016)      | Action-Adventure-Fantasy-SciFi                            |
| 163949  | The Beatles: Eight Days a Week - The Touring	     | Documentary                                      |

rating_df : dataset ratings.csv yang dapat dilihat pada tabel 2.

Tabel 2. *Overview* rating_df
| userId | movieId | rating |
|--------|---------|--------|
| 1	     | 31	     | 2.5    |
| 1	     | 1029	     | 3.0    |
| 1	     | 1061	     | 3.0    |
| 1	     | 1129	     | 2.0    |
| ...... | ....... | ...... |
| 671    | 6268  | 2.5    |
| 671    | 6269  | 4.0    |
| 671    | 6365  | 4.0    |

tags_df : dataset tags.csv yang dapat dilihat pada tabel 3.

Tabel 3. *Overview* tags_df
| userId | movieId | tag              | timestamp  | 
|--------|---------|------------------| ---------- |
| 15	     | 339		 | funnysandra 'boring' bullock    | 1138537770 |
| 15	     | 1955		 | dentist  | 1193435061 |
| 15	     | 7478		 | Cambodia      | 1170560997 |
| 15	     | 32892	 | Russian      | 1170626366 |
| ...... | ....... | ................ | .......... |
| 660	   | 135518			   | meaning of life           | 1436680885 |
| 660	   | 135518			   | philosophical | 1436680885 |
| 660      | 135518	 | sci-fi | 1436680885 |


## Data Preparation
Teknik *data preparation* yang dilakukan di antaranya :
1. Melakukan tahapan normalisasi yang ditujukan untuk mengubah nilai kolom numerik dalam kumpulan data ke skala umum, tanpa mendistorsi perbedaan dalam rentang nilai. Data yang dinormalisasikan yaitu data yang ada pada kolom rating pada file ratings.csv dalam hal ini digunakan metode *Min Max* yang mana metode tersebut bekerja dengan mengurangi setiap nilai fitur dengan nilai minimum fitur dan membaginya dengan *range* atau nilai maksimum dari nilai dikurangi nilai minimum fitur.
2. Melakukan *splitting* dataset. Disini dataset dibagi 2 yaitu *train data* yang digunakan sebagai model training dan *test data* diperuntukkan validasi untuk mengecek model yang digunakan sudah akurat atau belum. Perbandingan yang umum digunakan untuk *splitting* dataset yaitu 80% *train data* dan 20% *test data*. Ketika melakukan splitting dataset, tahap awal yaitu mengacak *sample data*. Dilanjutkan dengan membagi data yang mana terdapat parameter test_size yang akan digunakan saat mendefinisikan ukuran data testing. Dalam tugas ini digunakan test_size = 200000. Kemudian membagi data untuk *modelling*. Pada *modelling* tersebut digunakan *slicing* dengan format [baris, kolom] yang mana [X_train[:, 0], X_train[:, 1] yang berarti akan mengeksekusi semua baris serta kolom pertama dan kedua.

## Modelling and Result
Model yang digunakan dalam proyek ini menggunakan *embedding technique*. Proyek ini menggunakan model *Neural Collaborative Filtering* (NFC). Model *Neural Collaborative Filtering* (NFC) adalah jaringan saraf yang menyediakan penyaringan kolaboratif berdasarkan umpan balik implisit. Secara khusus, ini memberikan rekomendasi produk berdasarkan interaksi pengguna-item. Data pelatihan untuk model ini harus berisi satu set pasangan (*userid, animeid*) yang menunjukkan bahwa pengguna tertentu telah memberi peringkat atau mengklik item untuk berinteraksi dengannya. Di bawah ini adalah langkah-langkah untuk mendapatkan daftar rekomendasi film berdasarkan aktivitas pengguna berdasarkan *rate* yang ditentukan pengguna.
1. Mencari movie data yang ditonton oleh pengguna dan masukkan ke dalam *dataframe* baru. Jika parameter yang digunakan adalah *userId*, plot dengan nilai *False*, dan temp dengan nilai 1.

2. Kemudian cari rating terendah untuk film dimana parameter yang digunakan rating_df.userId yang bernilai sama dengan userId.

3. Selanjutnya, buat top_movie_reference dengan mengurutkan berdasarkan rating film. Parameter yang digunakan adalah sort_values dari 'rating' dan *ascending* dengan nilai 'False'.

4. Kemudian membuat *dataframe* baru (user_pref_df) berdasarkan *dataframe* utama (movie_df) dan memilih di mana data input adalah video yang terdapat dalam top_movie_refference.

5. Dilanjutkan dengan menghitung *rating* rata-rata pengguna. Parameter yang digunakan adalah rating_df. userId sama dengan *userId*.


- Untuk parameter BatchNormalization yang digunakan bernilai x, dimana BatchNormalization dapat mempercepat kekonvergenan serta mempercepat proses training dan meningkatkan learning rates pada saat pembuatan model. 
- Untuk parameter loss yang digunakan adalah 'binary_crossentropy', yang merupakan fungsi loss default yang digunakan untuk masalah klasifikasi biner.
- Untuk parameter optimizer menggunakan ‘adam’, karena Adam adalah algoritma optimasi pengganti untuk stochastic gradient descent untuk training model.
- Untuk function callback di berlakukan sebagai value di dalam function lain sehingga function callback akan di ekseskusi setelah function yang membungkus function callback tersebut selesai di eksekusi. 


Pada tabel 4. Berikut list dari delapan movie recommendations yang didapatkan dari tugas ini.

Tabel 4. *Movie Recommendation*
| movieId | title	                             | genres                                          |
|---------|------------------------------------|-------------------------------------------------|
| 19       | Ace Ventura: When Nature Calls (1995)		| Comedy     |
| 47     | Seven (a.k.a. Se7en) (1995)	               | Mystery-Thriller  |
| 110     | Braveheart (1995)   | Action-Drama-War  |
| 158     | Casper (1995)			| Adventure-Children  |
| 186     | Nine Months (1995)	             | Comedy-Romance |
| 204     | Under Siege 2: Dark Territory (1995)	| Action   |
| 224     | Don Juan DeMarco (1995) | Comedy-Drama-Romance                                           |
| 227  	  | Drop Zone (1994)		       | Action-Thiller  |


## Evaluation
Pada tahap evaluasi disini digunakan *mse* atau *mean squared error, precision, dan recall*.

1. *Mean Squared Error* atau MSE merupakan sebuah metode biasanya digunakan untuk memeriksa perkiraan nilai kesalahan dalam prediksi. Nilai *Mean Squared Error* yang rendah atau nilai kesalahan kuadrat rata-rata mendekati nol menunjukkan bahwa hasil prakiraan cocok dengan data aktual dan dapat digunakan untuk perhitungan prakiraan di masa mendatang. Metode *Mean Squared Error* umumnya digunakan untuk mengevaluasi metode pengukuran menggunakan regresi atau *prediction model* seperti *moving average, weighted moving average* dan analisis *trendline*. *Mean Squared Error* (MSE) dihitung dengan cara mengurangkan nilai data aktual dengan data prediksi, mengkuadratkan atau squared hasilnya, lalu menjumlahkan semuanya dan membaginya dengan jumlah data yang ada. Nilai *Mean Squared Error* yang diperoleh disini yaitu 0.0336.

![mse](https://user-images.githubusercontent.com/112928081/193192963-b23c3d34-9077-4928-85db-44ffd618d133.jpg)

#### Gambar 1. MSE *Graphic*

Dari grafik gambar 1 tersebut dihasilkan dari proses training model yang mana grafik dengan warna biru (MSE *Train*) menunjukkan penurunan. Sementara grafik dengan warna oranye (MSE *Test*) menunjukkan stabil.

2. *Precission* merupakan tingkat keakurasian antara informasi yang diminta oleh pengguna dan respon dari sistem. Nilai *precission* yang diperoleh disini yaitu 1.0000.

![precision](https://user-images.githubusercontent.com/112928081/193192981-61bb67f0-fded-43a2-b387-e74013dadd99.jpg)

#### Gambar 2. *Precission graphic*

Dari grafik gambar 2 tersebut dihasilkan dari proses training model yang mana grafik dengan warna biru (*Precission Train*) menunjukkan penurunan. Sementara grafik dengan warna oranye (*Precission Test*) menunjukkan stabil.

3. *Recall* merupakan tingkat keberhasilan dari sebuah sistem dalam menemukan kembali sebuah informasi. Nilai *recall* yang diperoleh disini yaitu 0.7143.

![recall](https://user-images.githubusercontent.com/112928081/193193002-7a772c0f-c8a3-4a07-b6c9-8473cd10a170.jpg)

#### Gambar 3. *Recall graphic*

Dari grafik gambar 3 tersebut dihasilkan dari proses training model yang mana grafik dengan warna biru (*Recall Train*) menunjukkan cenderung naik. Sementara grafik dengan warna oranye (*Recall Test*) menunjukkan cenderung turun.

## Conclusion
Dari tugas di atas dapat disimpulkan bahwa :
1. *Collaborative filtering* dapat digunakan untuk membuat suatu sistem *movie recommendation* dengan memprediksi *user rating* terhadap movie tersebut.
2. Nilai *Mean Squared Error* yang diperoleh yaitu 0.0336
3. Nilai *precission* yang diperoleh yaitu 1.0000
4. Nilai *recall* yang diperoleh yaitu 0.7143

## References
1. Arifin, Muhammad Fakhrul. 2017. Penerapan Metode Collaborative Filtering Dalam Sistem Rekomendasi Film. Universitas Pendidikan Indonesia.

