import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# CNN modelini oluşturmak için TensorFlow ve Keras'tan gerekli modülleri yüklüyoruz
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def parse_filename(file_name):
    file_name = file_name.replace('.BMP', '').replace('.bmp', '')
    parts = file_name.split('__')
    subject_id = parts[0]
    rest = parts[1]

    tokens = rest.split('_')
    gender = tokens[0]
    hand = tokens[1]
    finger = tokens[2]

    # Eğer 5 parça varsa: CR/OBL/ZCUT en sonda olur
    if len(tokens) == 5 and tokens[4].lower() in ['cr', 'obl', 'zcut']:
        alteration = tokens[4].upper()
        label = 'altered'
    else:
        alteration = None
        label = 'real'

    return subject_id, gender, hand, finger, label, alteration


def load_dataset(base_path):
    data = []

    # Real images
    real_path = os.path.join(base_path, "Real")
    for file_name in os.listdir(real_path):
        full_path = os.path.join(real_path, file_name)
        subject_id, gender, hand, finger, label, alteration = parse_filename(file_name)
        data.append([full_path, subject_id, gender, hand, finger, label, alteration])

    # Altered images (Easy, Medium, Hard)
    for level in ["Altered-Easy", "Altered-Medium", "Altered-Hard"]:
        altered_path = os.path.join(base_path, "Altered", level)
        for file_name in os.listdir(altered_path):
            full_path = os.path.join(altered_path, file_name)
            subject_id, gender, hand, finger, label, alteration = parse_filename(file_name)
            data.append([full_path, subject_id, gender, hand, finger, label, alteration])

    df = pd.DataFrame(data, columns=["image_path", "subject_id", "gender", "hand", "finger", "label", "alteration"])
    return df
dataset_path = "./SOCOFing"
df = load_dataset(dataset_path)
print(df.head())

print(df['label'].value_counts())
print(df['alteration'].value_counts())

img = Image.open(df.iloc[0]['image_path'])
plt.imshow(img, cmap='gray')
plt.title(f"{df.iloc[0]['label']} - {df.iloc[0]['subject_id']}")
plt.axis('off')
plt.show()

# GPU bellek ayarları
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # GPU bellek büyümesini etkinleştir
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


IMG_SIZE = (64, 64)  

# X: görsellerin array halleri
# y: her görselin subject_id'si (etiketi)
X=[]
y=[]

for idx, row in df.iterrows():
    try:
        
        img = Image.open(row['image_path'])

        img = img.convert("RGB")

        # Görseli yeniden boyutlandır (64x64)
        img = img.resize(IMG_SIZE)

        # Görseli NumPy array'e çevir ve X'e ekle
        X.append(np.array(img))

        # Etiketi (subject_id) y'ye ekle
        y.append(row['subject_id'])

    except Exception as e:
        # Eğer görsel okunamazsa hata verir
        print(f"Error with image {row['image_path']}: {e}")
        continue

X = np.array(X)
X = X.astype('float32') / 255.0  # [0,1] aralığına normalize et
y=np.array(y)

# Boyut kontrolü
print("Görsel verisi boyutu:", X.shape)
print("Etiket verisi boyutu:", y.shape)

#Etiketleri sayısal değere çevirme
label_encoder= LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

y_categorical = to_categorical(y_encoded)

# Sınıf sayısını da saklıyoruz, model için kullanacağız
num_classes = y_categorical.shape[1]
print("Sınıf sayısı:", num_classes)

X_train, X_test, y_train, y_test = train_test_split(X,y_categorical , test_size=0.2, random_state=42,stratify=y_encoded
)

print("Eğitim verisi boyutu:", X_train.shape)
print("Test verisi boyutu:", X_test.shape)

# Daha hafif CNN modeli
model= Sequential()
model.add(Conv2D(16,(3,3),activation='relu',input_shape=(64,64,3)))  # 32 yerine 16
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3, 3), activation='relu'))  # 64 yerine 32
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten: 2B veriyi düzleştirip dense katmanlara hazırlıyoruz
model.add(Flatten())

# Tam bağlantılı (dense) katman: 64 nöronlu (128 yerine)
model.add(Dense(64, activation='relu'))

# Overfitting'i azaltmak için Dropout uyguluyoruz (%50)
model.add(Dropout(0.5))

# Çıkış katmanı: Sınıf sayısı kadar nöron, softmax ile sınıf olasılıkları
model.add(Dense(num_classes, activation='softmax'))

# Modeli derliyoruz: kayıp fonksiyonu, optimizer ve metriği belirtiyoruz
model.compile(
    loss='categorical_crossentropy',  # Çok sınıflı sınıflandırmalarda kullanılır
    optimizer=Adam(learning_rate=0.001),  # Öğrenme oranını düşürdük
    metrics=['accuracy']  # Doğruluk oranını takip ediyoruz
)

# Modeli eğitiyoruz - daha küçük batch size
history = model.fit(
    X_train, y_train,             # Eğitim verileri
    validation_split=0.1,         # %10'unu validasyon için ayır
    epochs=50,                    # 30 kez tüm veriyi dolaşacak
    batch_size=8,                 # 16 yerine 8 - daha az bellek kullanımı
    verbose=1                     # Eğitim sürecini ekrana yazdır
)
# Test verisiyle başarı oranını ölçüyoruz
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test doğruluğu (accuracy):", test_acc)

print("Benzersiz kişi sayısı:", len(label_encoder.classes_))
print("İlk 10 kişi:", label_encoder.classes_[:10])

# Eğitim grafiklerini çizin
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Validasyon Doğruluğu')
plt.title('Model Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Validasyon Kaybı')
plt.title('Model Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.tight_layout()
plt.show()

# Model özetini göster
model.summary()

# Modeli kaydet
model.save('fingerprint_model.h5')
print("Model fingerprint_model.h5 olarak kaydedildi!")

# Tahmin fonksiyonu
def predict_person(image_path, model, label_encoder):
    """Bir parmak izi görselinden kişi tahmini yapar"""
    try:
        # Görseli yükle ve işle
        img = Image.open(image_path)
        img = img.convert("RGB")
        img = img.resize((64, 64))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Batch dimension ekle
        
        # Tahmin yap
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Kişi ID'sini al
        person_id = label_encoder.inverse_transform([predicted_class])[0]
        
        return person_id, confidence
    except Exception as e:
        return None, 0.0

# Test için gerçek ve bozuk görselleri test etme
print("\n=== Test Tahminleri ===")

# Gerçek görselleri test et
print("GERÇEK GÖRSELLER:")
real_images = df[df['label'] == 'real'].head(3)
for idx, row in real_images.iterrows():
    test_image = row['image_path']
    true_person = row['subject_id']
    
    predicted_person, confidence = predict_person(test_image, model, label_encoder)
    
    print(f"Görsel: {test_image.split(os.sep)[-1]}")
    print(f"Gerçek kişi: {true_person}")
    print(f"Tahmin: {predicted_person} (Güven: %{confidence*100:.1f})")
    print(f"Doğru mu? {'✅' if predicted_person == true_person else '❌'}")
    print("-" * 50)

# Bozuk görselleri test et
print("\nBOZUK GÖRSELLER:")
altered_images = df[df['label'] == 'altered'].head(3)
for idx, row in altered_images.iterrows():
    test_image = row['image_path']
    true_person = row['subject_id']
    alteration_type = row['alteration']
    
    predicted_person, confidence = predict_person(test_image, model, label_encoder)
    
    print(f"Görsel: {test_image.split(os.sep)[-1]}")
    print(f"Bozuk türü: {alteration_type}")
    print(f"Gerçek kişi: {true_person}")
    print(f"Tahmin: {predicted_person} (Güven: %{confidence*100:.1f})")
    print(f"Doğru mu? {'✅' if predicted_person == true_person else '❌'}")
    print("-" * 50)

# Her zorluk seviyesinden bir örnek test et
print("\nZORLUK SEVİYELERİNE GÖRE TEST:")
for level in ['CR', 'OBL', 'ZCUT']:
    if level in df['alteration'].values:
        level_images = df[df['alteration'] == level].head(2)
        print(f"\n--- {level} Seviyesi ---")
        for idx, row in level_images.iterrows():
            test_image = row['image_path']
            true_person = row['subject_id']
            
            predicted_person, confidence = predict_person(test_image, model, label_encoder)
            
            print(f"Görsel: {test_image.split(os.sep)[-1]}")
            print(f"Gerçek kişi: {true_person}")
            print(f"Tahmin: {predicted_person} (Güven: %{confidence*100:.1f})")
            print(f"Doğru mu? {'✅' if predicted_person == true_person else '❌'}")
            print("-" * 30)

# Genel performans analizi
print("\n GENEL PERFORMANS ANALİZİ:")
print("=" * 50)

# Gerçek görsellerde başarı oranı
real_test = df[df['label'] == 'real'].sample(min(20, len(df[df['label'] == 'real'])))
real_correct = 0
for idx, row in real_test.iterrows():
    predicted_person, confidence = predict_person(row['image_path'], model, label_encoder)
    if predicted_person == row['subject_id']:
        real_correct += 1

real_accuracy = real_correct / len(real_test) * 100
print(f"Gerçek görsellerde başarı oranı: %{real_accuracy:.1f} ({real_correct}/{len(real_test)})")

# Bozuk görsellerde başarı oranı
if len(df[df['label'] == 'altered']) > 0:
    altered_test = df[df['label'] == 'altered'].sample(min(20, len(df[df['label'] == 'altered'])))
    altered_correct = 0
    for idx, row in altered_test.iterrows():
        predicted_person, confidence = predict_person(row['image_path'], model, label_encoder)
        if predicted_person == row['subject_id']:
            altered_correct += 1
    
    altered_accuracy = altered_correct / len(altered_test) * 100
    print(f"Bozuk görsellerde başarı oranı: %{altered_accuracy:.1f} ({altered_correct}/{len(altered_test)})")
    
    # Performans farkı
    performance_diff = real_accuracy - altered_accuracy
    print(f"Performans farkı: %{performance_diff:.1f}")
    
    if performance_diff > 10:
        print("Model bozuk görsellerde zorlanıyor!")
    elif performance_diff > 5:
        print("Model bozuk görsellerde biraz zorlanıyor.")
    else:
        print("Model bozuk görsellerde de iyi performans gösteriyor!")
