# Parmak İzi Tanıma Sistemi (CNN ile Kişi Sınıflandırması)

Bu proje, **SOCOFing** parmak izi veri setini kullanarak **Convolutional Neural Network (CNN)** ile kişi sınıflandırması yapan bir makine öğrenmesi sistemidir. Sistem, hem gerçek hem de manipüle edilmiş parmak izi görsellerini analiz ederek kişi kimlik doğrulaması yapabilir.

## İçindekiler
- [Proje Hakkında](#proje-hakkında)
- [Model Mimarisi](#model-mimarisi)
- [Eğitim Süreci](#eğitim-süreci)
- [Sonuçlar](#sonuçlar)
- [Kurulum ve Kullanım](#kurulum-ve-kullanım)
- [Dosya Yapısı](#dosya-yapısı)

## Proje Hakkında

Bu proje, biyometrik kimlik doğrulama alanında CNN tabanlı derin öğrenme yaklaşımını kullanarak:
- **600 farklı kişiyi** parmak izi görsellerinden tanıyabilir
- **Gerçek ve manipüle edilmiş** parmak izi görsellerini işleyebilir
- **%76+ doğruluk oranı** ile kişi sınıflandırması yapar
- **Interaktif test arayüzü** ile performans analizi sunar

### Teknik Özellikler
- **CNN Mimarisi**: 2 konvolüsyonel + 2 tam bağlantılı katman
- **Görsel Boyutu**: 64x64 piksel (RGB)
- **Optimizasyon**: Adam optimizer (learning_rate=0.001)
- **Regularization**: Dropout (%50)

### Veri Seti
**SOCOFing (Sokoto Coventry Fingerprint Dataset)** kullanılmıştır:
- **Toplam Kişi**: 600 (1-600 arası)
- **Cinsiyet**: Erkek (M), Kadın (F)
- **El**: Sol (Left), Sağ (Right)
- **Parmak**: Başparmak, işaret, orta, yüzük, serçe
- **Manipülasyon Türleri**: CR, OBL, ZCUT

## Model Mimarisi

![Model Özeti](resimler/model_summary.png)

```python
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)             (None, 62, 62, 16)        448       
max_pooling2d (MaxPooling2D) (None, 31, 31, 16)       0         
conv2d_1 (Conv2D)           (None, 29, 29, 32)        4640      
max_pooling2d_1 (MaxPooling2D)(None, 14, 14, 32)      0         
flatten (Flatten)           (None, 6272)              0         
dense (Dense)               (None, 64)                401472    
dropout (Dropout)           (None, 64)                0         
dense_1 (Dense)             (None, 600)               39000     
=================================================================
Total params: 445,560
Trainable params: 445,560
Non-trainable params: 0
```

### Katman Detayları
- **Conv2D (16 filtre)**: 3x3 kernel, ReLU aktivasyon
- **MaxPooling2D**: 2x2 pooling
- **Conv2D (32 filtre)**: 3x3 kernel, ReLU aktivasyon
- **MaxPooling2D**: 2x2 pooling
- **Flatten**: 2D'den 1D'ye dönüştürme
- **Dense (64 nöron)**: ReLU aktivasyon
- **Dropout (0.5)**: Overfitting önleme
- **Dense (600 nöron)**: Softmax aktivasyon (çıkış)

## Eğitim Süreci

### Eğitim Parametreleri
- **Epoch Sayısı**: 30
- **Batch Size**: 8
- **Validation Split**: %10
- **Train-Test Split**: %80-%20

![Eğitim Grafikleri](resimler/training_graphs.png)

*Model eğitimi boyunca accuracy ve loss değerlerinin değişimi*

## Sonuçlar

### Performans Metrikleri
- **Genel Test Doğruluğu**: %76.05
- **Gerçek Görsellerde**: %100.0 (20/20)
- **Manipüle Görsellerde**: %75.0 (15/20)
- **Performans Farkı**: %25.0

![Test Sonuçları](resimler/test_accuracy.png)

### Analiz
- **Güçlü Yönler**: Temiz parmak izlerinde mükemmel performans
- **Gelişim Alanı**: Manipüle görsellerde dayanıklılık artırılabilir
- **Sonuç**: 600 kişilik sınıflandırma için başarılı sonuç

### Test Sonuçları Örnekleri

![Test Sonuçları 1](resimler/test_results_1.png)

*Gerçek test oturumlarından alınmış ekran görüntüleri*

## Kurulum ve Kullanım

### Gereksinimler
```bash
pip install tensorflow pandas scikit-learn pillow matplotlib numpy
```

### Kullanım Adımları

#### 1. Veri Seti Hazırlığı
- SOCOFing veri setini indirin
- Proje ana dizinine `SOCOFing/` klasörü olarak yerleştirin

#### 2. Model Eğitimi
```bash
cd src
python CNN.py
```

#### 3. İnteraktif Test
```bash
python test_model.py
```

**Test menüsü seçenekleri:**
1. Random karşılaştırma: Rastgele iki görsel
2. Aynı kişi analizi: Bir kişinin farklı parmakları
3. Gerçek vs manipüle: Normal ve bozuk görsel karşılaştırması
4. Manuel seçim: Kendiniz görsel seçin
5. Çıkış

## Dosya Yapısı

```
fingerprint-recognition-cnn/
├── README.md                    # Bu dokümantasyon
├── SOCOFing/                   # Veri seti (ayrıca indirilmeli)
│   ├── Real/                   # 6000+ gerçek parmak izi
│   └── Altered/               # Manipüle edilmiş görseller
├── src/
│   ├── CNN.py                 # Ana model eğitim scripti
│   └── test_model.py          # İnteraktif test arayüzü
├── resimler/                  # Proje görselleri
│   ├── model_summary.png      # Model katman özeti
│   ├── training_graphs.png    # Eğitim grafikleri
│   ├── test_accuracy.png      # Test sonuçları
│   └── test_results_1.png     # Örnek test ekranı
└── fingerprint_model.h5       # Eğitilmiş model (oluşturulacak)
```

---

## Proje Özeti

Bu proje, **600 farklı kişinin parmak izi görsellerini CNN ile sınıflandıran** kapsamlı bir makine öğrenmesi sistemidir. Sistem, hem **gerçek hem de manipüle edilmiş görselleri** işleyerek **%76+ doğruluk oranı** elde etmiştir.

**Ana Başarılar:**
- 600 sınıflı karmaşık problem çözümü
- Gerçek görsellerde %100 başarı
- İnteraktif test sistemi geliştirme
- Manipülasyon dayanıklılığı analizi

Bu sistem, **biyometrik güvenlik**, **kimlik doğrulama** ve **forensik analiz** alanlarında kullanılabilir.
