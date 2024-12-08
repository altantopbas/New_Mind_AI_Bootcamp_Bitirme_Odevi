# Sınıflandırma modeli (altan01/results) tanımlanır.
classification_model_name = "altan01/results"
classification_tokenizer = AutoTokenizer.from_pretrained(classification_model_name)
classification_model = AutoModelForSequenceClassification.from_pretrained(classification_model_name)

# Türkçe dil modeli (Hugging Face üzerinde Türkçe GPT-2 örneği) eklenir.
llm_model_name = "ytu-ce-cosmos/turkish-gpt2-large"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)

  # Pad token eksikliğini giderme işlemi yapılır.
if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
    
def classify_comment(user_comment):
    # Yorumu sınıflandırma ve sınıfa döndürme işlemi yapılır
    # Yorumu tokenize etme ve sınıf tahmini yapılır
    inputs = classification_tokenizer(user_comment, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = classification_model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Sınıf numarasını metne dönüştürme
    label_map = {0: "Negatif", 1: "Pozitif", 2: "Nötr"}
    return label_map.get(predicted_class, "Nötr")
    
#Olumlu yorum için teşekkür eden kısa bir yanıt yaz:\n
#Kötü yorum yapıldığı için özür dile neden kötü olduğunu sor:\n
#Yorumdan hareketle uygun bir cevap yaz:\n

def generate_response(comment_class, user_comment):
    # Sınıfa ve yoruma göre metin üret
    if comment_class == "Pozitif":
        prompt = f"Bu kullanıcı ürünü çok beğenmiş.\n Yorum: {user_comment} \n"
    elif comment_class == "Negatif":
        prompt = f"Bu kullanıcı üründen memnun kalmamış.\n Yorum: {user_comment}\n"
    else:
        prompt = f"Kullanıcı ürüne nötr yaklaşmış.\n Yorum: {user_comment}\n"

    # Tokenize et ve metin üret
    inputs = llm_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=100)
    outputs = llm_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask, # Attention mask ekledik
        max_length=150,  # Üretilecek maksimum kelime uzunluğu
        num_return_sequences=1,  # Kaç farklı metin üretilsin
        do_sample=True,
        temperature=0.7,  # Çeşitlilik kontrolü için
        top_p=0.92,  # Daha olasılığı yüksek kelimelere odaklan
        top_k=0,  # Kelime havuzunu daralt
        repetition_penalty=2.25,  # Tekrarlamaları azaltmak için
        pad_token_id=llm_tokenizer.pad_token_id,
        eos_token_id=llm_tokenizer.eos_token_id
    )
    generated_text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Kullanıcı yorumu
user_comment = input("Lütfen Yorumunuzu Girin: ")

# Yorumun sınıfını belirle
comment_class = classify_comment(user_comment)

# Metin üretimi
result = generate_response(comment_class, user_comment)
print("Üretilen Yanıt:\n", result)