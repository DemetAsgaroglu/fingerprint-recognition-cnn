# 🔍 Parmak İzi Tanıma Sistemi (CNN ile Kişi Sınıflandırması)

Bu proje, **SOCOFing** parmak izi veri setini kullanarak **Convolutional Neural Network (CNN)** ile kişi sınıflandırması yapan bir makine öğrenmesi sistemidir. Sistem, hem gerçek hem de manipüle edilmiş parmak izi görsellerini analiz ederek kişi kimlik doğrulaması yapabilir.

## 📋 İçindekiler
- [Proje Hakkında](#proje-hakkında)
- [Özellikler](#özellikler)
- [Veri Seti](#veri-seti)
- [Veri Ön İşleme](#veri-ön-işleme)
- [Model Geliştirme Süreci](#model-geliştirme-süreci)
- [Model Mimarisi](#model-mimarisi)
- [Eğitim Süreci](#eğitim-süreci)
- [Test ve Değerlendirme](#test-ve-değerlendirme)
- [Sonuçlar ve Analiz](#sonuçlar-ve-analiz)
- [Kurulum ve Kullanım](#kurulum-ve-kullanım)
- [Dosya Yapısı](#dosya-yapısı)

## 🎯 Proje Hakkında

Bu proje, biyometrik kimlik doğrulama alanında CNN tabanlı derin öğrenme yaklaşımını kullanarak:
- **600 farklı kişiyi** parmak izi görsellerinden tanıyabilir
- **Gerçek ve manipüle edilmiş** parmak izi görsellerini işleyebilir
- **%76+ doğruluk oranı** ile kişi sınıflandırması yapar
- **Interaktif test arayüzü** ile performans analizi sunar

### 🔬 Bilimsel Amaç
- Biyometrik güvenlik sistemlerinde CNN'lerin etkinliğini test etmek
- Manipüle edilmiş parmak izi görsellerine karşı dayanıklılık analizi
- Farklı parmak türleri ve el pozisyonlarının tanıma başarısına etkisini incelemek

## ✨ Özellikler

### 🤖 Model Özellikleri
- **CNN Mimarisi**: 2 konvolüsyonel katman + 2 tam bağlantılı katman
- **Görsel Boyutu**: 64x64 piksel (RGB)
- **Sınıf Sayısı**: 600 farklı kişi
- **Optimizasyon**: Adam optimizer (learning_rate=0.001)
- **Regularization**: Dropout (%50)

### 📊 Veri İşleme
- **Otomatik veri yükleme** ve ön işleme
- **Label encoding** ile kişi ID'lerinin sayısallaştırılması
- **Train-test split** (%80-%20 oranında)
- **Normalizasyon** ([0,1] aralığına)

### 🧪 Test Sistemi
- **İnteraktif menü** ile farklı test seçenekleri
- **Gerçek vs manipüle edilmiş görsel** karşılaştırması
- **Aynı kişinin farklı parmakları** arasında test
- **Güven skoru analizi** ve çapraz kontrol

## 📦 Veri Seti

**SOCOFing (Sokoto Coventry Fingerprint Dataset)** kullanılmıştır:

### 📂 Veri Yapısı
```
SOCOFing/
├── Real/                    # Gerçek parmak izi görselleri
│   ├── 1__M_Left_index_finger.BMP
│   ├── 1__M_Left_thumb_finger.BMP
│   └── ... (6000 görsel)
└── Altered/                 # Manipüle edilmiş görseller
    ├── Altered-Easy/        # Kolay seviye manipülasyon
    ├── Altered-Medium/      # Orta seviye manipülasyon
    └── Altered-Hard/        # Zor seviye manipülasyon
```

### 📝 Dosya Adlandırma
```
<KişiID>__<Cinsiyet>_<El>_<Parmak>_[Manipülasyon].BMP
Örnek: 100__M_Left_index_finger_CR.BMP
```

### 📈 Veri İstatistikleri
- **Toplam Kişi**: 600 (1-600 arası)
- **Cinsiyet**: Erkek (M), Kadın (F)
- **El**: Sol (Left), Sağ (Right)
- **Parmak**: Başparmak, işaret, orta, yüzük, serçe
- **Manipülasyon Türleri**: CR, OBL, ZCUT

## � Veri Ön İşleme

### 📋 Dosya Adı Analizi
Projede ilk olarak parmak izi dosyalarının adlarından önemli bilgileri çıkaran bir `parse_filename()` fonksiyonu geliştirildi:

```python
def parse_filename(file_name):
    # Örnek: "100__M_Left_index_finger_CR.BMP"
    # Çıktı: subject_id=100, gender=M, hand=Left, finger=index, alteration=CR
```

**Çıkarılan Özellikler:**
- 🆔 **Kişi ID'si**: 1-600 arası benzersiz tanımlayıcı
- 👥 **Cinsiyet**: M (Erkek), F (Kadın)
- ✋ **El**: Left (Sol), Right (Sağ)
- 👆 **Parmak Türü**: index, middle, ring, little, thumb
- 🔀 **Manipülasyon**: CR, OBL, ZCUT (bozuk görseller için)

### 📊 Veri Setini Yükleme
`load_dataset()` fonksiyonu ile:
- Real klasöründen **gerçek parmak izi görselleri**
- Altered klasörlerinden **manipüle edilmiş görseller**
- Toplam **6000+ görsel** otomatik olarak yüklendi

### 🖼️ Görsel Ön İşleme
Her görsel için uygulanan işlemler:
1. **Format Dönüşümü**: BMP → RGB (3 kanal)
2. **Boyut Standardizasyonu**: Farklı boyutlar → 64x64 piksel
3. **Normalizasyon**: [0-255] → [0.0-1.0] aralığı
4. **Array Dönüşümü**: PIL Image → NumPy array

### 🏷️ Etiket İşleme (Label Encoding)
```python
# Kişi ID'lerini sayısal sınıflara dönüştürme
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # "100" → 99 (0-indexed)
y_categorical = to_categorical(y_encoded)   # One-hot encoding
```

## 🏗️ Model Geliştirme Süreci

### 🎯 Problem Tanımı
- **Problem Türü**: Çok sınıflı sınıflandırma (Multi-class Classification)
- **Sınıf Sayısı**: 600 farklı kişi
- **Girdi**: 64x64x3 RGB parmak izi görselleri
- **Çıktı**: Kişi kimliği (0-599 arası sınıf)

### 🧠 CNN Mimarisi Tasarımı
Hafif ve etkili bir CNN modeli tasarlandı:

**Katman 1-2: Özellik Çıkarma**
- Conv2D(16 filtre) + MaxPooling → Temel kenar ve doku özelliklerini yakalar
- Conv2D(32 filtre) + MaxPooling → Daha karmaşık desenleri öğrenir

**Katman 3-4: Sınıflandırma**
- Flatten → 2D'yi 1D'ye dönüştürür
- Dense(64) + Dropout → Özellik birleştirme ve overfitting önleme

**Çıkış Katmanı**
- Dense(600) + Softmax → 600 kişi için olasılık dağılımı

### ⚙️ Optimizasyon Stratejisi
- **Optimizer**: Adam (adaptive learning rate)
- **Learning Rate**: 0.001 (deneysel olarak belirlendi)
- **Loss Function**: Categorical Crossentropy
- **Regularization**: Dropout (%50) overfitting'i önlemek için

## 🏗️ Model Mimarisi

### Gereksinimler
```bash
pip install tensorflow
pip install opencv-python
pip install pillow
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install numpy
```

### Proje Klonlama
```bash
git clone https://github.com/[username]/fingerprint-recognition-cnn.git
cd fingerprint-recognition-cnn
```

### Veri Seti İndirme
1. [SOCOFing Dataset](https://www.kaggle.com/datasets/ruizgara/socofing) adresinden veri setini indirin
2. `SOCOFing` klasörünü proje ana dizinine yerleştirin

## 🚀 Kullanım

### 1. Model Eğitimi
```bash
cd src
python CNN.py
```

Bu komut:
- Veri setini yükler ve ön işleme yapar
- CNN modelini eğitir (30 epoch)
- Modeli `fingerprint_model.h5` olarak kaydeder
- Eğitim grafiklerini gösterir
- Test sonuçlarını analiz eder

### 2. İnteraktif Test
```bash
python test_model.py
```

Test menüsü seçenekleri:
1. **Random karşılaştırma**: Rastgele iki görsel
2. **Aynı kişi analizi**: Bir kişinin farklı parmakları
3. **Gerçek vs manipüle**: Normal ve bozuk görsel karşılaştırması
4. **Manuel seçim**: Kendiniz görsel seçin
5. **Çıkış**

## 🏗️ Model Mimarisi

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

![Model Mimarisi Diyagramı](resimler/model_architecture.png)

### Katman Detayları
- **Conv2D (16 filtre)**: 3x3 kernel, ReLU aktivasyon
- **MaxPooling2D**: 2x2 pooling
- **Conv2D (32 filtre)**: 3x3 kernel, ReLU aktivasyon
- **MaxPooling2D**: 2x2 pooling
- **Flatten**: 2D'den 1D'ye dönüştürme
- **Dense (64 nöron)**: ReLU aktivasyon
- **Dropout (0.5)**: Overfitting önleme
- **Dense (600 nöron)**: Softmax aktivasyon (çıkış)

## 🎓 Eğitim Süreci

### 📊 Veri Bölümü
```python
# %80 eğitim, %20 test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded  # Her sınıftan eşit oranda örnekleme
)
```

### 🔧 Eğitim Parametreleri
- **Epoch Sayısı**: 30 (deneysel olarak optimize edildi)
- **Batch Size**: 8 (bellek optimizasyonu için küçük tutuldu)
- **Validation Split**: %10 (eğitim setinin %10'u validasyon için)
- **Verbose**: 1 (eğitim ilerlemesini görmek için)

### 📈 GPU Optimizasyonu
```python
# GPU bellek büyümesini etkinleştirme
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### 🎯 Eğitim Stratejisi
1. **Bellek Yönetimi**: 64x64 görsel boyutu ile GPU bellek kullanımını optimize ettik
2. **Batch Boyutu**: Küçük batch size (8) ile stabil eğitim sağlandı
3. **Validation**: Her epoch'ta validasyon kaybı ve doğruluğu izlendi
4. **Early Stopping**: Manual kontrol ile overfitting engellendi

### 📊 Eğitim İlerlemesi

![Eğitim İlerlemesi](resimler/training_progress.png)

### 📈 Eğitim Grafikleri

![Eğitim Grafikleri](resimler/training_graphs.png)

*Model eğitimi boyunca accuracy ve loss değerlerinin değişimi*

### ✅ Eğitim Tamamlanması

![Eğitim Sonuçları](resimler/training_results.png)

*30 epoch sonunda elde edilen nihai eğitim sonuçları*

## 🧪 Test ve Değerlendirme

### 🔍 Test Sistemi Geliştirme
İnteraktif bir test arayüzü (`test_model.py`) geliştirildi:

#### 1. **Model Yükleme Sistemi**
```python
model = load_model('fingerprint_model.h5')  # Eğitilmiş modeli yükle
df, label_encoder = load_dataset_info()     # Veri seti bilgilerini yükle
```

#### 2. **Tahmin Fonksiyonu**
```python
def predict_person(image_path, model, label_encoder):
    # Görsel yükleme ve ön işleme
    # Model tahmini
    # Güven skoru hesaplama
    return person_id, confidence
```

#### 3. **Karşılaştırma Sistemi**
Her test için şu analizler yapılır:
- **Temel Tahmin**: Hangi kişiye ait olduğu
- **Güven Skoru**: Model ne kadar emin (%0-100)
- **Çapraz Kontrol**: Yanlış tahminlerde alternatifleri gösterme

### 📋 Test Kategorileri

#### 🎲 1. Random Karşılaştırma
- Rastgele iki görsel seçilerek karşılaştırılır
- Aynı kişi mi, farklı kişiler mi analiz edilir

#### 👤 2. Aynı Kişi Analizi  
- Belirli bir kişinin farklı parmak izleri test edilir
- Model tutarlılığı ölçülür

#### 🆚 3. Gerçek vs Manipüle Görsel
- Aynı kişinin normal ve bozuk görselleri karşılaştırılır
- Manipülasyona karşı dayanıklılık test edilir

#### ✋ 4. Manuel Test
- Kullanıcı istediği görselleri seçebilir
- Özel durumlar için esneklik sağlar

## 📊 Sonuçlar ve Analiz

### 🎯 Performans Metrikleri
- **Genel Test Doğruluğu**: %76.05
- **Gerçek Görsellerde**: %100.0 (20/20)
- **Manipüle Görsellerde**: %75.0 (15/20)
- **Performans Farkı**: %25.0

![Test Sonuçları](resimler/test_accuracy.png)

### 📈 Analiz
- ✅ **Güçlü Yönler**: Temiz parmak izlerinde mükemmel performans
- ⚠️ **Gelişim Alanı**: Manipüle görsellerde dayanıklılık artırılabilir
- 🔍 **Sonuç**: 600 kişilik sınıflandırma için başarılı sonuç

![Performans Analizi](resimler/performance_analysis.png)

### 📉 Eğitim Grafikleri
Model eğitimi sırasında accuracy ve loss grafikleri otomatik olarak gösterilir.

### 🔬 Detaylı Performans Analizi

#### ✅ **Başarılı Yönler**
1. **Yüksek Genel Doğruluk**: %76.05 (600 sınıf için mükemmel)
2. **Gerçek Görsellerde Mükemmellik**: %100 başarı oranı
3. **Hızlı Tahmin**: Her görsel için ~20-50ms süre
4. **Tutarlı Güven Skorları**: Doğru tahminlerde yüksek güven

#### ⚠️ **Gelişim Alanları**
1. **Manipüle Görsellerde Düşüş**: %25 performans kaybı
2. **Belirli Manipülasyon Türlerinde Zorluk**: CR, OBL, ZCUT'ta farklı başarı
3. **Bellek Kullanımı**: Büyük veri setlerinde optimizasyon gereksinimi

#### 🎯 **İstatistiksel Bulgular**
- **En Başarılı**: Temiz, kaliteli parmak izi görselleri
- **En Zor**: ZCUT manipülasyonu (kesik/yırtık simülasyonu)
- **Orta Zorluk**: CR (Merkezi Döndürme), OBL (Oblik Deformasyon)

### 🧠 **Model Davranış Analizi**
```
Test Örneği:
Görsel: 100__M_Left_index_finger.BMP → Tahmin: 100 (%100 güven) ✅
Görsel: 100__M_Left_index_finger_CR.BMP → Tahmin: 100 (%85 güven) ⚠️
```

**Gözlem**: Model manipüle görsellerde doğru tahmin yapıyor ancak güven azalıyor.

### 🖼️ Test Sonuçları Örnekleri

![Test Örnekleri](resimler/test_examples.png)

*Farklı test kategorilerinden örnek sonuçlar*

## 🛠️ Kurulum ve Kullanım

### Gereksinimler
```bash
pip install tensorflow pandas scikit-learn pillow matplotlib numpy
```

### Kullanım Adımları

#### 1. **Veri Seti Hazırlığı**
- SOCOFing veri setini indirin
- Proje ana dizinine `SOCOFing/` klasörü olarak yerleştirin

#### 2. **Model Eğitimi**
```bash
cd src
python CNN.py
```
**Çıktılar:**
- Eğitim sürecinin canlı takibi
- Accuracy/Loss grafikleri
- Test sonuçları ve analiz
- `fingerprint_model.h5` model dosyası

#### 3. **İnteraktif Test**
```bash
python test_model.py
```
**Özellikler:**
- Menü tabanlı kolay kullanım
- Gerçek zamanlı tahmin
- Detaylı karşılaştırma analizi

### 📊 Test Çıktı Örnekleri

#### ✅ **Başarılı Karşılaştırma**
```
IKI GORSEL KARSILASTIRMASI:
============================================================
Gorsel 1: 100__M_Left_index_finger.BMP
   Tahmin: Kisi 100 (Guven: %100.0)
Gorsel 2: 100__M_Right_thumb_finger.BMP  
   Tahmin: Kisi 100 (Guven: %98.5)
AYNI KISI! Her iki gorsel de 100 numarali kisiye ait
Guven farki: %1.5
Model her iki gorselde de cok emin!
```

#### ❌ **Farklı Kişiler**
```
Gorsel 1: 150__F_Left_middle_finger.BMP
   Tahmin: Kisi 150 (Guven: %95.2)
Gorsel 2: 200__M_Right_index_finger_CR.BMP
   Tahmin: Kisi 200 (Guven: %78.3)
FARKLI KISILER! Gorsel 1: 150, Gorsel 2: 200
Capraz analiz:
   Gorsel 1'in 200 olma ihtimali: %2.1
   Gorsel 2'nin 150 olma ihtimali: %3.4
```

### 🧪 İnteraktif Test Sonuçları

![Test Sonuçları 1](resimler/test_results_1.png)

![Test Sonuçları 2](resimler/test_results_2.png)

*Gerçek test oturumlarından alınmış ekran görüntüleri*

## 📁 Dosya Yapısı

```
fingerprint-recognition-cnn/
├── README.md                    # Bu dokümantasyon
├── SOCOFing/                   # Veri seti (ayrıca indirilmeli)
│   ├── Real/                   # 6000+ gerçek parmak izi
│   └── Altered/               # Manipüle edilmiş görseller
│       ├── Altered-Easy/      # Hafif manipülasyon
│       ├── Altered-Medium/    # Orta seviye manipülasyon  
│       └── Altered-Hard/      # Yoğun manipülasyon
├── src/
│   ├── CNN.py                 # 🎯 Ana model eğitim scripti
│   └── test_model.py          # 🧪 İnteraktif test arayüzü
└── fingerprint_model.h5       # 💾 Eğitilmiş model (oluşturulacak)
```

### 🔧 **Teknik Detaylar**

#### `CNN.py` - Ana Eğitim Scripti
- **Veri yükleme**: `load_dataset()` fonksiyonu
- **Ön işleme**: Görsel boyutlandırma ve normalizasyon  
- **Model tanımı**: Sequential CNN mimarisi
- **Eğitim**: 30 epoch ile optimize edilmiş eğitim
- **Değerlendirme**: Test seti üzerinde performans analizi
- **Görselleştirme**: Accuracy/Loss grafikleri
- **Model kaydetme**: `fingerprint_model.h5` formatında

#### `test_model.py` - Test Arayüzü
- **Model yükleme**: Kaydedilmiş modeli kullanma
- **İnteraktif menü**: 5 farklı test seçeneği
- **Karşılaştırma**: İki görsel arasında detaylı analiz
- **Güven skoru**: Tahmin güvenilirliği hesaplama
- **Çapraz kontrol**: Alternatif tahminlerin analizi

---

## 🎓 **Proje Özeti**

Bu proje, **600 farklı kişinin parmak izi görsellerini CNN ile sınıflandıran** kapsamlı bir makine öğrenmesi sistemidir. Sistem, hem **gerçek hem de manipüle edilmiş görselleri** işleyerek **%76+ doğruluk oranı** elde etmiştir.

**Ana Başarılar:**
- ✅ 600 sınıflı karmaşık problem çözümü
- ✅ Gerçek görsellerde %100 başarı
- ✅ İnteraktif test sistemi geliştirme
- ✅ Manipülasyon dayanıklılığı analizi

**Teknik Yenilikler:**
- 🔧 Optimized CNN mimarisi (445K parametre)
- 🔧 Bellek-verimli eğitim stratejisi  
- 🔧 Comprehensive test framework
- 🔧 Real-time performance analysis

Bu sistem, **biyometrik güvenlik**, **kimlik doğrulama** ve **forensik analiz** alanlarında kullanılabilir.
