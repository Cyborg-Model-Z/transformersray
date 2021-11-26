from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

# Sentiment analysis pipeline
pipeline('sentiment-analysis')

# Question answering pipeline, specifying the checkpoint identifier
pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased')

# Named entity recognition pipeline, passing in a specific model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
pipeline('ner', model=model, tokenizer=tokenizer)