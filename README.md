
# New Mind AI Bootcamp Bitirme Projesi

## Duygu Durum Analizi Sonrası LLM Üzerinden Sonuç Üretme💻 

Projemin amacı, ürünlere yapılacak olan yorumların duygu durumunu analiz edip bu duygu durumuna göre LLM üzerinden cevap üretmektir. 🚀

Veri seti olarak herhangi bir e-ticaret firmasına ait ürün yorumları ve yorumların ait olduğu duygu sınıfları yer almaktadır.📝

Projede Kullanılan Sınıflandırma Türleri
- 
Projede üç farklı sınıflandırma kullanılmıştır. Bunlar:

-Random Forest

-XGBoost

-Bert Tabanlı Sınıflandırma

Sınıflandırma Türlerinin Sonuç Parametreleri:
-
```bash
 *** RandomForest Model *** 

Accuracy: 0.8605800922874094

classification_report: 
               precision    recall  f1-score   support

           0       0.88      0.91      0.89      1361
           1       0.84      0.95      0.89      1368
           2       0.79      0.28      0.41       305

    accuracy                           0.86      3034
   macro avg       0.84      0.71      0.73      3034
weighted avg       0.86      0.86      0.84      3034

Confussion Matrix: 
 [[1233  112   16]
 [  68 1294    6]
 [  94  127   84]]
```
```bash
 *** XGBoost Model *** 

Accuracy: 0.8744232036914964

classification_report: 
               precision    recall  f1-score   support

           0       0.90      0.92      0.91      1361
           1       0.89      0.92      0.90      1368
           2       0.63      0.47      0.54       305

    accuracy                           0.87      3034
   macro avg       0.81      0.77      0.78      3034
weighted avg       0.87      0.87      0.87      3034

Confussion Matrix: 
 [[1258   61   42]
 [  73 1252   43]
 [  73   89  143]]
```
```bash
*** Bert Tabanlı Sınıflandırma ***
Loss: 0.3292
Accuracy: 0.9262
Precision: 0.9262
Recall: 0.9262
F1: 0.9262
```

---
Projeye Ait Çıktı Görselleri ve Proje Kodları:
-
Projeye ait tüm kodlara [buradan](https://github.com/altantopbas/New_Mind_AI_Bootcamp_Bitirme_Odevi/tree/main/Proje_kodlari) erişebilirsiniz.

Projeye ait çıktı görsellerine [buradan](https://github.com/altantopbas/New_Mind_AI_Bootcamp_Bitirme_Odevi/tree/main/Sonuc_gorselleri) erişebilirsiniz.
## Özellikler

- Yazılan yorumdan anlık olarak pozitif, negatif, nötr şeklinde duygu durum analizi yapılması
- Yazılan yoruma cevap olarak LLM üzernden cevap üretilmesi
- Google Colab ile çalıştırılması
- Türkçe GPT2 Modeli kullanılması

  
## Bilgisayarınızda Çalıştırın

Projeyi klonlayın

```bash
  git clone https://github.com/altantopbas/New_Mind_AI_Bootcamp_Bitirme_Odevi.git
```

Proje dizinine gidin

```bash
  cd Proje_kodlari
```

Python dosyasını çalıştırın

```bash
  python new_mind_ai_bitirme_llm_cevap_uretme.py
```


## Ekran Görüntüleri

Projenin çalışırken ki ekran görüntüleri aşağıda yer almaktadır.
### Negatif Yoruma Üretilen Cevap
![negatif_sonuc](https://github.com/altantopbas/New_Mind_AI_Bootcamp_Bitirme_Odevi/blob/main/Sonuc_gorselleri/negatif_sonuc.png)
  


 ### Pozitif Yoruma Üretilen Cevap
![pozitif_sonuc](https://github.com/altantopbas/New_Mind_AI_Bootcamp_Bitirme_Odevi/blob/main/Sonuc_gorselleri/pozitif_sonuc.png)
  
## Modelde kullanılan optimizasyonlar 
### Model tanımlaması:
```bash
llm_model_name = "ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1"

llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

llm_model = GPT2LMHeadModel.from_pretrained(llm_model_name, device_map="cpu", torch_dtype="auto"
)
```
### Model parametreleri:
``` bash
pipe = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer)

generation_args = {
"max_new_tokens": 150,
"return_full_text": False,
"temperature": 0.0,        
"do_sample": False
}
```
### Modele verilecek prompt:
```bash
def gen_response(comment_class, user_comment):

    prompt_text = f"""
      Sen bir ürünün yorumlarını inceleyen bir analistsin. Kullanıcı yorumuna cevap olarak kullanıcıya uygun kısa ve nazik bir yanıt oluştur.
    
      Yorum: {user_comment}
      Sınıf: {comment_class}

      Yanıt:
      """
    instruction_prompt = f"### Kullanıcı:\n{prompt_text}\n### Asistan:\n"
    generated_text = pipe(instruction_prompt, **generation_args)
    result = generated_text[0]["generated_text"]
    return result
```    

## Testler
### Modeli test etmek için:
```bash
# Kullanıcı yorumu
user_comment = input("Lütfen Yorumunuzu Girin: ")

# Yorumun sınıfını belirle
comment_class = classify_comment(user_comment)

# Metin üretimi
result = gen_response(comment_class, user_comment)
print("Üretilen Yanıt:\n", result)
```

  
## Geri Bildirim

Herhangi bir geri bildiriminiz varsa, aşağıdaki iletişim adreslerime ulaşabilirsiniz

[E-Mail](altantopbas5@gmail.com)

[LinkedIn](https://www.linkedin.com/in/altantopbas/)

  
## Lisans

[MIT](https://choosealicense.com/licenses/mit/)

  
