package com.example.imageclassification

import android.app.Activity
import android.app.ComponentCaller
import android.content.ContentValues.TAG
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import com.example.imageclassification.databinding.ActivityMainBinding
import com.example.imageclassification.ml.MobilenetV110224Quant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {
    lateinit var bitmap: Bitmap
    lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        val view = binding.root
        setContentView(view)

        binding.uploadbtn.setOnClickListener(View.OnClickListener {
            var intent:Intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"
            startActivityForResult(intent,100)
            binding.detailstxt.text = "Tahmin: "
        })

        binding.searchbtn
            .setOnClickListener(View.OnClickListener {
            if (::bitmap.isInitialized) { // Resim seçilip seçilmediğini kontrol edin
                val resized: Bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

                val model = MobilenetV110224Quant.newInstance(this)

                // Model girişini hazırlayın
                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
                val tensorImage = TensorImage(DataType.UINT8)
                tensorImage.load(resized)
                inputFeature0.loadBuffer(tensorImage.buffer)

                // Modeli çalıştırın ve sonuç alın
                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer

                // Çıkışın maksimum değere sahip indeksini bulun
                val floatArray = outputFeature0.floatArray
                val maxIndex = floatArray.indices.maxByOrNull { floatArray[it] } ?: -1

                // Etiketleri yükleyin ve sınıf adını bulun
                val labels = loadLabels()
                val predictedLabel = if (maxIndex >= 0 && maxIndex < labels.size) labels[maxIndex] else "Bilinmeyen"

                // Sonucu TextView'de gösterin
                binding.detailstxt.text = "Tahmin: $predictedLabel"

                // Modeli kapatın
                model.close()
            } else {
                Toast.makeText(this, "Lütfen bir resim seçin", Toast.LENGTH_SHORT).show()
            }
        })
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        try {
            // Kullanıcının resim seçimi başarıyla gerçekleşti mi?
            if (resultCode == Activity.RESULT_OK && requestCode == 100) {
                // Seçilen resmi ImageView'de göster
                val selectedImageUri: Uri? = data?.data
                binding.image1.setImageURI(selectedImageUri)

                // Seçilen resmi bitmap olarak al
                if (selectedImageUri != null) {
                    bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, selectedImageUri)
                }
            } else {
                Toast.makeText(this, "Resim seçimi iptal edildi veya hata oluştu", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Resim seçimi sırasında hata oluştu: ", e)
            Toast.makeText(this, "Bir hata oluştu", Toast.LENGTH_SHORT).show()
        }
    }
    private fun loadLabels(): List<String> {
        val labels = mutableListOf<String>()
        try {
            val inputStream = assets.open("labels_mobilenet_dataset.txt")
            inputStream.bufferedReader().useLines { lines -> lines.forEach { labels.add(it) } }
        } catch (e: Exception) {
            Log.e(TAG, "Etiketler yüklenirken hata oluştu: ${e.message}")
        }
        return labels
    }


}