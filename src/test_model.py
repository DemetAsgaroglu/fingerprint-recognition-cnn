import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def parse_filename(file_name):
    """Dosya adından bilgileri çıkarır"""
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

def load_dataset_info(base_path):
    """Veri setini yükler ve label encoder'ı hazırlar"""
    data = []

    # Real images
    real_path = os.path.join(base_path, "Real")
    if os.path.exists(real_path):
        for file_name in os.listdir(real_path):
            if file_name.endswith('.BMP') or file_name.endswith('.bmp'):
                full_path = os.path.join(real_path, file_name)
                subject_id, gender, hand, finger, label, alteration = parse_filename(file_name)
                data.append([full_path, subject_id, gender, hand, finger, label, alteration])

    # Altered images (Easy, Medium, Hard)
    for level in ["Altered-Easy", "Altered-Medium", "Altered-Hard"]:
        altered_path = os.path.join(base_path, "Altered", level)
        if os.path.exists(altered_path):
            for file_name in os.listdir(altered_path):
                if file_name.endswith('.BMP') or file_name.endswith('.bmp'):
                    full_path = os.path.join(altered_path, file_name)
                    subject_id, gender, hand, finger, label, alteration = parse_filename(file_name)
                    data.append([full_path, subject_id, gender, hand, finger, label, alteration])

    df = pd.DataFrame(data, columns=["image_path", "subject_id", "gender", "hand", "finger", "label", "alteration"])
    
    if len(df) == 0:
        print("UYARI: Hic gorsel bulunamadi!")
        print(f"Kontrol edilen yol: {base_path}")
        return None, None
    
    # Label encoder'ı hazırla
    label_encoder = LabelEncoder()
    label_encoder.fit(df['subject_id'])
    
    return df, label_encoder

def predict_person(image_path, model, label_encoder, img_size=(64, 64)):
    """Bir parmak izi görselinden kişi tahmini yapar"""
    try:
        # Görseli yükle ve işle
        img = Image.open(image_path)
        img = img.convert("RGB")
        img = img.resize(img_size)
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Batch dimension ekle
        
        # Tahmin yap
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Kişi ID'sini al
        person_id = label_encoder.inverse_transform([predicted_class])[0]
        
        return person_id, confidence, predictions[0]
    except Exception as e:
        print(f"Hata: {e}")
        return None, 0.0, None

def test_specific_images(model_path, dataset_path):
    """Belirli görselleri test eder"""
    
    print("Model ve veri seti yukleniyor...")
    
    # Modeli yükle
    model = load_model(model_path)
    print(f"Model yuklendi: {model_path}")
    
    # Veri setini ve label encoder'ı hazırla
    df, label_encoder = load_dataset_info(dataset_path)
    
    if df is None:
        print("Veri seti yuklenemedi!")
        return None, None, None
        
    print(f"Veri seti yuklendi: {len(df)} gorsel bulundu")
    print(f"Toplam kisi sayisi: {len(label_encoder.classes_)}")
    
    return model, df, label_encoder

def compare_images(model, label_encoder, image1_path, image2_path):
    """İki görseli karşılaştırır"""
    print("\nIKI GORSEL KARSILASTIRMASI:")
    print("=" * 60)
    
    # İlk görsel
    person1, conf1, pred1 = predict_person(image1_path, model, label_encoder)
    print(f"Gorsel 1: {os.path.basename(image1_path)}")
    print(f"   Tahmin: Kisi {person1} (Guven: %{conf1*100:.1f})")
    
    # İkinci görsel
    person2, conf2, pred2 = predict_person(image2_path, model, label_encoder)
    print(f"Gorsel 2: {os.path.basename(image2_path)}")
    print(f"   Tahmin: Kisi {person2} (Guven: %{conf2*100:.1f})")
    
    # Karşılaştırma
    if person1 == person2:
        print(f"AYNI KISI! Her iki gorsel de {person1} numarali kisiye ait")
        
        # Güven farkı
        conf_diff = abs(conf1 - conf2) * 100
        print(f"Guven farki: %{conf_diff:.1f}")
        
        if conf_diff < 5:
            print("Model her iki gorselde de cok emin!")
        elif conf_diff < 15:
            print("Model makul seviyede emin.")
        else:
            print("Model guven seviyeleri arasinda buyuk fark var.")
            
    else:
        print(f"FARKLI KISILER! Gorsel 1: {person1}, Gorsel 2: {person2}")
        
        # Çapraz güven analizi
        person1_idx = label_encoder.transform([person1])[0]
        person2_idx = label_encoder.transform([person2])[0]
        
        cross_conf1 = pred1[person2_idx] * 100  # Görsel 1'in person2'ye güveni
        cross_conf2 = pred2[person1_idx] * 100  # Görsel 2'nin person1'e güveni
        
        print(f"Capraz analiz:")
        print(f"   Gorsel 1'in {person2} olma ihtimali: %{cross_conf1:.1f}")
        print(f"   Gorsel 2'nin {person1} olma ihtimali: %{cross_conf2:.1f}")

def interactive_test():
    """Etkileşimli test modulu"""
    
    # Model ve veri setini yükle
    model_path = "fingerprint_model.h5"
    dataset_path = "./SOCOFing"  # Mevcut dizinden SOCOFing klasörü

    
    if not os.path.exists(model_path):
        print(f"Model dosyasi bulunamadi: {model_path}")
        print("Once CNN.py'yi calistirarak modeli egitin!")
        return
        
    model, df, label_encoder = test_specific_images(model_path, dataset_path)
    
    # Veri seti kontrol
    if df is None or len(df) == 0 or model is None:
        print("HATA: Model veya veri seti yuklenemedi!")
        print("Lutfen:")
        print("1. fingerprint_model.h5 dosyasinin mevcut oldugunu kontrol edin")
        print("2. SOCOFing klasorunun dogru yerde oldugunu kontrol edin")
        return
    
    while True:
        print("\n" + "="*50)
        print("PARMAK IZI TEST MENUSU")
        print("="*50)
        print("1. Random iki gorsel karsilastir")
        print("2. Belirli bir kisinin farkli parmak izlerini karsilastir")
        print("3. Gercek vs Bozuk gorsel karsilastir")
        print("4. Manual gorsel sec ve test et")
        print("5. Cikis")
        
        choice = input("\nSeciminizi yapin (1-5): ").strip()
        
        if choice == "1":
            # Random iki görsel seç
            sample1 = df.sample(1).iloc[0]
            sample2 = df.sample(1).iloc[0]
            compare_images(model, label_encoder, sample1['image_path'], sample2['image_path'])
            
        elif choice == "2":
            # Belirli bir kişinin farklı parmak izleri
            person = input("Kisi ID'si girin (orn: 100): ").strip()
            person_images = df[df['subject_id'] == person]
            
            if len(person_images) < 2:
                print(f"{person} numarali kisi icin yeterli gorsel bulunamadi!")
                continue
                
            sample1 = person_images.sample(1).iloc[0]
            sample2 = person_images.sample(1).iloc[0]
            
            print(f"\n{person} numarali kisinin farkli parmak izleri karsilastiriliyor:")
            compare_images(model, label_encoder, sample1['image_path'], sample2['image_path'])
            
        elif choice == "3":
            # Gerçek vs bozuk
            person = input("Kisi ID'si girin (orn: 100): ").strip()
            
            real_img = df[(df['subject_id'] == person) & (df['label'] == 'real')]
            altered_img = df[(df['subject_id'] == person) & (df['label'] == 'altered')]
            
            if len(real_img) == 0 or len(altered_img) == 0:
                print(f"{person} numarali kisi icin gercek veya bozuk gorsel bulunamadi!")
                continue
                
            real_sample = real_img.sample(1).iloc[0]
            altered_sample = altered_img.sample(1).iloc[0]
            
            print(f"\n{person} numarali kisinin gercek vs bozuk gorseli:")
            compare_images(model, label_encoder, real_sample['image_path'], altered_sample['image_path'])
            
        elif choice == "4":
            print("\nMevcut gorseller:")
            for i, (idx, row) in enumerate(df.head(10).iterrows()):
                print(f"{i+1}. {os.path.basename(row['image_path'])} (Kisi: {row['subject_id']})")
            
            try:
                img1_idx = int(input("Ilk gorsel numarasi (1-10): ")) - 1
                img2_idx = int(input("Ikinci gorsel numarasi (1-10): ")) - 1
                
                img1_path = df.iloc[img1_idx]['image_path']
                img2_path = df.iloc[img2_idx]['image_path']
                
                compare_images(model, label_encoder, img1_path, img2_path)
                
            except (ValueError, IndexError):
                print("Gecersiz numara!")
                
        elif choice == "5":
            print("Test tamamlandi!")
            break
            
        else:
            print("Gecersiz secim!")

interactive_test()
