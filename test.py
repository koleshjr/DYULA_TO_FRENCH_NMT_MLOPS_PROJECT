from transformers import pipeline,  T5Tokenizer
repository_id = "saved_models/Koleshjrflan-t5-base-finetuned-translation-v5"
tokenizer = T5Tokenizer.from_pretrained(repository_id)
pipe_ft = pipeline("translation", model = repository_id, max_length=tokenizer.model_max_length, device_map="auto")
        

if __name__ == "__main__":
    input_text = 'Translate the following sentence from Dyula to French: "i tɔgɔ bi cogodɔ"'
    translation = pipe_ft([input_text])[0]['translation_text']
    print(f"Translated to French: {translation}")