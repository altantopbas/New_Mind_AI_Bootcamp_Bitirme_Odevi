import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

comments_set=pd.read_csv('urun_yorumlari.csv',on_bad_lines="skip",delimiter=";")
# on_bad_lines = "skip": Hatalı satırları atlar ve yalnızca doğru formatlı satırları işler.
# delimiter = ";": CSV dosyasındaki ayırıcı karakterin noktalı virgül (;) olduğunu belirtir.

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

comments_set.head() # Veri seti kontrol edilir.

"""Veri seti incelendi.
1: Pozitif,
0: Negatif
2: Nötr
"""

comments_set.isnull().sum() # Veri setinde herhangi bir boşluk var mı kontrol edilir.

comments_set.dtypes # Veri setinde bulunan sütunların özellikleri kontrol edilir.

# Dil işleme için gerekli kütüphaneler indirilir.
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Metin Temizleme Fonksiyonu
def clean_text(text):
    # 1. Küçük harfe çevir
    text = text.lower()

    # 2. Noktalama işaretlerini kaldır
    text = re.sub(r"[^\w\s]", '', text)

    # 3. Tokenize et
    tokens = word_tokenize(text)

    # 4. Stop Words'leri kaldır
    stop_words = set(stopwords.words('turkish'))  # Türkçe stop words
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # 6. Temizlenmiş kelimeleri birleştir
    cleaned_text = ' '.join(tokens)

    return cleaned_text

df = comments_set.copy()

# Temizleme işlemi
df["Metin"] = df["Metin"].apply(clean_text)

# Numerik İfadeleri Etiketleme
def katbul(sutun):
    kat={
        1:"Olumlu",
        0:"Olumsuz",
        2:"Nötr"
    }
    label=kat.get(sutun)
    return label

df['Label']=df['Durum'].apply(katbul)

df.head() # Metin ön işlemesi yapıldıktan sonra veri seti kontrol edilir.

df['Label'].unique() # Etiketleme işlemi kontrol edilir.

X = df['Metin'] # Features olarak, X belirlenir.
y = df['Durum'] # Target olarak, Y belirlenir.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)
 # X ve y test ve train için ayrılır.
 # Veri setinin %80'i eğitim, %20'si test için ayrılmıştır.

tfidf_vectorizer = TfidfVectorizer(max_features=1000) #TF-IDF ile vektörleştirilme işlemi yapılır.

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Temiz metinleri sayısal bir vektöre dönüştürürüz. Böylelikle üzerlerinde işlem yapılabilir hale getiririz.

rf_model = RandomForestClassifier(n_estimators=500)
rf_model.fit(X_train_tfidf, y_train)
# 500 ağaçlı bir RandomForest Modeli kullanarak eğitime başlarız.

# RandomForest

# Test verisi ile doğruluğu hesaplama
y_pred_rf = rf_model.predict(X_test_tfidf)
accuracy_rf = rf_model.score(X_test_tfidf, y_test)

# Modelin performans raporu
print(" *** RandomForest Model *** ")
print("\nAccuracy:", accuracy_rf)
print("\nclassification_report: \n", classification_report(y_test, y_pred_rf))
print("Confussion Matrix: \n", confusion_matrix(y_test, y_pred_rf))

!pip install xgboost

from xgboost import XGBClassifier
import xgboost as xgb

model_xg = XGBClassifier(n_estimators=500)
model_xg.fit(X_train_tfidf, y_train)
# Aynı veri setini XGBoost modeli ile de eğitiriz.

# XGBoost

# Test verisi ile doğruluğu hesaplama
y_pred_xg = model_xg.predict(X_test_tfidf)
accuracy_xg = model_xg.score(X_test_tfidf, y_test)

# Modelin performans raporu
print(" *** XGBoost Model *** ")
print("\nAccuracy:", accuracy_xg)
print("\nclassification_report: \n", classification_report(y_test, y_pred_xg))
print("Confussion Matrix: \n", confusion_matrix(y_test, y_pred_xg))

# İki farklı model kullanılarak performansları ölçeriz.
# Sonuçlara bakacak olursak XGBoost modelinin bu veri seti için daha iyi bir sonuç verdiğini söyleyebiliriz.

import joblib

# Modeli kaydetme
joblib.dump(model_xg, 'xgboost_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Modeli yükleme
loaded_model = joblib.load('xgboost_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

model = model_xg

# Modeli pickle ile kaydetmek
with open('rf_model_SA.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model başarıyla kaydedildi!")

# Modeli pickle ile yüklemek
with open('rf_model_SA.pkl', 'rb') as file:
    rf_model_SA = pickle.load(file)

print("Model başarıyla yüklendi!")



