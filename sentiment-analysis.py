from transformers import pipeline
classifier = pipeline("text-generation", model="distilgpt2")
classifier(
"In this course we will", 
max_length=30,num_return_sequences=2,)