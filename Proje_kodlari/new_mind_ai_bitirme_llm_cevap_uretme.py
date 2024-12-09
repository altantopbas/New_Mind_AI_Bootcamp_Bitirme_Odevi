from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers import pipeline
import torch

# Sınıflandırma modeli (altan01/results) tanımlanır.
classification_model_name = "altan01/results"
classification_tokenizer = AutoTokenizer.from_pretrained(classification_model_name)
classification_model = AutoModelForSequenceClassification.from_pretrained(classification_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_id = 0 if torch.cuda.is_available() else -1

# Türkçe dil modeli (Hugging Face üzerinde Türkçe GPT-2 örneği) eklenir.
llm_model_name = "ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = GPT2LMHeadModel.from_pretrained(
    llm_model_name,
    device_map="cpu",
    torch_dtype="auto",
    #trust_remote_code=True,
)

pipe = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer)
generation_args = {
    "max_new_tokens": 150,
    "return_full_text": False,
    "temperature": 0.0,  # Daha yaratıcı yanıtlar için
    #"top_p": 0.9,        # Örnekleme
    "do_sample": False    # Sampling etkin
}

# Pad token eksikliğini giderme işlemi yapılır.
if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

def classify_comment(user_comment):
    # Yorumu sınıflandırma ve sınıfa döndürme işlemi yapılır
    # Yorumu tokenize etme ve sınıf tahmini yapılır
    inputs = classification_tokenizer(user_comment, return_tensors="pt", truncation=True, padding=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = classification_model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Sınıf numarasını metne dönüştürme
    label_map = {0: "Negatif", 1: "Pozitif", 2: "Nötr"}
    return label_map.get(predicted_class, "Nötr")

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

# Kullanıcı yorumu
user_comment = input("Lütfen Yorumunuzu Girin: ")

# Yorumun sınıfını belirle
comment_class = classify_comment(user_comment)

# Metin üretimi
result = gen_response(comment_class, user_comment)
print("Üretilen Yanıt:\n", result)



