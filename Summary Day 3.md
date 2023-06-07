Perbedaan antara Tradisioinal ML & Neural Network/Deep Learning
  - Hidden layer
    * Trad. ML -> Tidak ada Hidden Layer
    * NN/DL -> Memiliki Hidden Layer
    
Kapan waktu yang tepat untuk menggunakan NN/DL
  - Ketika kita ingin memprediksi data unstructured
    * Image
    * Video
    * Suara
    * Text
  - Walaupun peruntuhannya untuk data unstructured, NN/DL juga bisa digunakan untuk data structured
    * Ketika memang model tradisional ML tidak bisa memberikan hasil yang kita inginkan
    
Bagaimanakan proses belajar NN/DL
  - Step 1: Inisiasi bobot secara random, untuk mengunci sifat randomnya kita berikan parameter set.seed()
  - Step 2: Feed Foward, Input Layer -> Hidden Layer -> Output Layer
    * Ketika input dibawa ke Hidden Layer, pada bagian Hidden Layer dilakukan Scalling ataupun Activation Function
    * Dari hidden layer dibawa ke bagian output layer, pada bagian tersebut nilai errornya akan dihitung.
  - Step 3: Back Propagation, bobot yang ada akan diupdate berdasarkan perhitungan turunan parsialnya.
    * Hasil dari turunan parsialnya akan dikalikan dengan learning rate lalu dikurangin dengan bobot awalnya.
    * Learning rate memiliki tujuan untuk mengatur seberapa cepat model kita mencari titip global optimum/nilai error paling rendah
  - Step 4: Semua langkah di atas akan diulangi terus menerus. Sehingga model kita sampai ke nilai global optimum.

Fungsi yang kita gunakan dalam membuat deep learning masih sama yaitu `neuralnet()`
  - Agar model kita bisa menjadi deep learning, pada parameter `hidden =` kita akan menggunakan fungsi `c()`
    * Sebagai contoh: c(5, 3) 
      + Akan terdapat 2 hidden layer
      + Hidden layer 1 terdaapt 5 nodes
      + Hidden layer 2 terdapat 3 nodes
      
Setelah membuat model kita dapat melakukan prediksi dengan menggunakan fungsi `compute()`
    
Dalam membuat model klasifikasi denan menggunakan NN/DL, terdapat tahapan Pre-Processing data yang harus kita lakukan terlebih dahulu, yaitu membuat dummy variable secara manual.
  - Tahapan pembuatan dummy variable secara manual sering juga disebut One Hot Encoding
    * Fungsi yang kita gunakan adalah `model.matrix()`
      + Setelah berhasil membuat dummy variable terdapat 3 hal lain yang harus kita lakukan yaitu
        ~ Mengubah menjadi bentuk data frame
        ~ Menghilangkan nilai interceptnya
        ~ Menambahkan kolom taget variablenya
        
  - Setelah berhasil membuat dummy variable, nama kolom yang dihasilkan harus dicek kemballi karena jika terdapat simbol, model NN/ DL tidak bisa menerima data seperti itu
    * Fungsi yan dapat kita gunakan ada `gsub()`

Deep Learning with `Keras`:

  - Keras merupakan sebuah framework yang dikembangkan oleh Google yang dapat kita manfaatkan untuk membuat arsitektur Deep Learning yang lebih flexible dan powerfull.
  - Dalam mengimplementasikan Keras kita akan meminjamnya melalui Anaconda Prompt/Python dan akan kita sambungkan dengan aplikasi R kita.

Implementasi Deep Learning denan Keras untuk mengolah data gambar.

Data Pre-processing untuk data gambar di Keras:

  - Data dalam bentuk `data.frame` diubah terlebih dahulu menjadi `array` karena keras menerima input dalam bentuk array. Tahapan: data.frame -> matrix -> array.
    * Kita harus mengubahnya menjadi bentuk array karena python hanya dapat menerima bentuk data dalam bentuk array.
  - Pengubahan data dalam bentuk matrix ke array:
    * data prediktor diubah menggunakan fungsi `array_reshape()`
    * data target (label) diubah menggunakan fungsi `to_categorical()`

Cross Validation (Additional):

  - Pada tahapan Cross Validation, kita dapat membagi data kita menjadi 3 bagian yaitu:
    * Data Train: 
      ~ data ini akan kita manfaatkan untuk melatih model yang sudah kita buat
    * Data Validation: 
      ~ data ini akan kita manfaatkan untuk verifikasi pertama terhadap model yang sudah kita buat
    * Data Test: 
      ~ data ini akan kita manfaatkan untuk verifikasi kedua terhadap model
      ~ hasil dari data test ini dapat kita bandingkan terhadap hasil evaluasi dari data validation, untuk memastikan apakah model kita Overfit, Underfit atau JustRight
      
Pembuatan Model di Keras:

  1. Define Model Architecture:
  
    - Tahapan pertama adalah mempersiapkan arsitektur dari model Deep Learning yang akan ingin digunakan
      * Hal pertama yang akan dilakukan pada bagian ini adalah membangun pondasi dengan menggunakan fungsi `model_keras_sequential()`
        ~ Setiap kita ingin memperbaiki arsitektur dari model kita, kita harus mengulangnya dari bagian ini
      * Selanjutnya adalah membuat Input, Hidden & Output layer dengan menggunakan fungsi `layer_dense()`
      * Sitem pembuatannya menerapkan konsep *layer by layer*.
        ~ Penempatan fungsi `layer_dense()` paling pertama akan di definisikan sebagai Input Sekaligus Hidden Layer 1
        ~ Penempatan fungsi `layer_dense()` seterusnya akan di definisikan sebagai Hidden Layer 2, dst
        ~ Penempatan fungsi `layer_dense()` paling akhir akan menjadi Output layer
      * Setiap Nodes pada Hidden & Output layer dapat kita atur Activation Functionnya
      
  2. Compile Model:
  
    - Tahapan kedua yang akan dilakukan adalah membuat Compile Model 
      * Pada tahapan ini, kita akan memberikan model kita metode untuk melakukan Back Propagation, dengan cara menambahkan fungsi `compile()`
        ~ Untuk memberikan perhitungan error, kita akan menambahkan parameter `loss = `
        ~ Untuk memberikan pembobotan, kita akan menambahkan parameter `optimizer = `
          + Setiap fungsi optimizer yang kita gunakan, dapat kita isi dengan perameter `learning_rate = `, untuk mengatur seberapa cepat peng-updatean bobot.
        ~ Parameter terakhir yang dapat kita gunakan adalah `metrics = `. Parameter tersebut dapat kita isi denagn `accuracy` untuk melihat nilai accuracy-nya yang dihasilkan.

  3. Fit Model:
  
  - Pada tahapan terakhir adalah tahapan dimana kita melakukan training model
    * Dalam meakukan training ini, kita akan menggunakan fungsi `fit()`.
  - Pada keras, model yang dibuat hanya 1 buah, namun proses trainingnya dapat dilakukan beberapa kali. Pada tiap proses training, dilakukan pembagian data ke beberapa **batch** melalui random sampling. Model akan ditraining menggunakan data pada batch 1 terlebih dahulu, kemudian batch selanjutnya, hingga digunakan seluruh data (1 **epoch**). 
  
  
  
  
  
  
  
  
  
  
  
  
  