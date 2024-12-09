#Gerekli kütüphaneler yüklendi.
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer

# Model oluşturma için gerekli eğitim, gpu üzerinden yapıldı.
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
device

df=pd.read_csv('urun_yorumlari.csv',on_bad_lines="skip",delimiter=";")
# on_bad_lines = "skip": Hatalı satırları atlar ve yalnızca doğru formatlı satırları işler.
# delimiter = ";": CSV dosyasındaki ayırıcı karakterin noktalı virgül (;) olduğunu belirtir.

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


    # 5. Temizlenmiş kelimeleri birleştir
    cleaned_text = ' '.join(tokens)

    return cleaned_text
    
    # Temizleme işlemi
df["Metin"] = df["Metin"].apply(clean_text)

# Numerik İfadeleri Etiketleme
def katbul(sutun):
    kat={
        1:"Pozitif",
        0:"Negatif",
        2:"Nötr"
    }
    label=kat.get(sutun)
    return label
df['Label']=df['Durum'].apply(katbul)
# Veri setinde durumlar 0, 1 ve 2 şeklinde numerik olarak belirtildiği için label ataması yapılır.

X=df['Metin'] #Tahmin edilmesi için gerekli features
y=df['Durum'] #Tahmin edilmesi istenen sütun

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42) # %20 test için ayrılmıştır.

#Sınıflandırma modeli için bert tabanlı türkçe model seçilir.
model_name = "dbmdz/bert-base-turkish-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Veriyi tokenlaştırma işlemi yapılır.
def tokenize_function(texts, labels):
    tokens = tokenizer(list(texts), truncation=True, padding=True, max_length=128)
    tokens['labels'] = list(labels)
    return tokens
    
 train_encodings = tokenize_function(X_train, y_train)
test_encodings = tokenize_function(X_test, y_test)

# Basit bir Dataset sınıfı
class ReviewDataset:
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: np.array(val[idx]) for key, val in self.encodings.items()}

train_dataset = ReviewDataset(train_encodings)
test_dataset = ReviewDataset(test_encodings)

# Model oluşturma
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(df['Durum'].unique())).to(device) 
# Oluşturulan modelin label sayısı, veri setindeki durumların unique sayısı kadar olmalıdır, yani 3 adet.
# .to(device) komutu ile gpu'da çalışması sağlanmıştır.

# Eğitim ayarları
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,
    report_to="none",
)
# Eğitim parametreleri burada ayarlanır.
# Modelin ismi, raporlama yapılıp yapılmayacağı, en iyi halinin çıkarılacağı, eğitim için kaç kez deneme yapılacağı ve nerede (cpu ya da gpu) da olup olmayacağı gibi parametreler ayarlanır.

!pip install -q evaluate # Değerlendirme fonksiyonları için evaluate kütüphanesinin kurulması gerekmektedir.

# Değerlendirme fonksiyonu
import evaluate
accuracy = evaluate.load("accuracy")
def compute_metrics(pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
# Değerlendirme fonksiyonları burada tanımlanır.

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

#Ayarlanan eğitim parametreleri burada eğitim için kullanılır.

# Modeli eğit
trainer.train() # Burada eğitime başlanır.

# Test sonuçları elde edilir.
predictions = trainer.predict(test_dataset)
print("Test Sonuçları:", predictions.metrics)

from huggingface_hub import login
login()
#Huggingface token'i ile Huggingface hesabı bağlanır.

trainer.push_to_hub(commit_message="Training Completed") #Eğitilen model burada paylaşılır.

# Test amaçlı modelin kullanımı aşağıda gerçekleşmiştir.
# Model ve tokenizer yüklenir. 
model_name = "altan01/results"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device) 

# Örnek ifadeler hazırlanır.
test_texts = [
    "Ürün çok güzel, kaliteli ve hızlı teslimat.",  # Pozitif yorum
    "Maalesef ürün istediğim gibi değildi, hayal kırıklığı yaşadım.",  # Negatif yorum
    "Ürün fena değil, idare eder.",  # Nötr yorum
]   

# Örnek metinleri tokenlaştırma
inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Giriş verileri cihaza(cpu ya da gpu) taşınır
inputs = {key: val.to(device) for key, val in inputs.items()}

# Modeli kullanarak tahmin yapma
with torch.no_grad():
    outputs = model(**inputs)
    
# Çıkışları işleme
predictions = torch.argmax(outputs.logits, dim=1)

# Sınıflandırma sonuçlarını yazdırma işlemi gerçekleşir.
labels = ["Negatif", "Pozitif", "Nötr"]
for text, pred in zip(test_texts, predictions):
    print(f"Metin: {text}\nTahmin Edilen Sınıf: {labels[pred]}\n")