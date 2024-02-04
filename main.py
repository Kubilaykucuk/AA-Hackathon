#Kütüphaneler importlanır
from transformers import BertTokenizerFast, EncoderDecoderModel
import tkinter as tk
import nltk
import networkx as nx
from nltk.tokenize import sent_tokenize
import random
import numpy as np

#Punkt indirilir
nltk.download('punkt')

#Bert modeli importlanır
tokenizer = BertTokenizerFast.from_pretrained("mrm8488/bert2bert_shared-turkish-summarization")
model = EncoderDecoderModel.from_pretrained("mrm8488/bert2bert_shared-turkish-summarization")

#Cümleler düzeltilir
def rearrange_sentences(text):
    sentences = sent_tokenize(text, language='turkish')
    
    random.shuffle(sentences)
    
    rearranged_text = ' '.join(sentences)
    return rearranged_text

#Özetler oluşturulur
def generate_headlines(text):
    #Tokenler oluşturulur ve maskelerden geçer
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    #Ön ayarlar yapılır
    headlines = []
    settings = [
        {"temperature": 0.001, "top_k": 10, "top_p": 0.9},
        {"temperature": 0.9, "top_k": 20, "top_p": 0.92}, 
        {"temperature": 1.1, "top_k": 30, "top_p": 0.94},
    ]


    #Her bir ayara göre özet üretilir
    for setting in settings:
        # Çeşitliliği sağlamak için değişen ayarlarla özeti oluşturur
        summary_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=80, min_length=5, do_sample=True, **setting, num_beams=4, num_return_sequences=1, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        #Tek cümle haline getirilir
        headline = summary.capitalize()
        if headline[-1] in ['.', ',', ';', ':']:
            headline = headline[:-1]


        first_period_pos = headline.find('.')
        if first_period_pos != -1:
            headline = headline[:first_period_pos + 1]
        
        headlines.append(headline)

    return headlines

def display_headlines(headlines):
    #Bir önceki çıktıyı sil
    for widget in output_frame.winfo_children():
        widget.destroy()

    #Her bir özeti ayrı ayrı göster
    for i, headline in enumerate(headlines, 1):
        text_widget = tk.Text(output_frame, height=4, width=100)
        text_widget.pack()

        text_widget.tag_configure("baslik_tag", foreground="blue", font=("Helvetica", 15))
        text_widget.insert("end", f"Özet {i}: ", "baslik_tag")

        text_widget.tag_configure("headline_tag", foreground="black", font=("Helvetica", 12, "bold"))
        text_widget.insert("end", headline, "headline_tag")

        text_widget.config(state="disabled")
        

def generate_and_display_headlines(modify_text=False):
    input_text = text_box.get("1.0", "end-1c") #Textboxtan yazıyı al
    
    if modify_text:
        input_text = rearrange_sentences(input_text) #Yeni özet oluştur
    
    headlines = generate_headlines(input_text) #Özetleri oluştur
    
    display_headlines(headlines) #Özetleri göster

#Tkinter windowunu oluştur
window = tk.Tk()
window.title("Özet ve Spot Oluşturucu")
window.geometry("800x700")

#Textbox oluştur
text_box = tk.Text(window, height=10, width=50)
text_box.pack()

#Özet oluşturmak için button oluştur
generate_button = tk.Button(window, text="Özet ve Spot Oluştur", command=lambda: generate_and_display_headlines(modify_text=False))
generate_button.pack()

#Yeniden özet oluşturmak için button oluştur
rearrange_button = tk.Button(window, text="Yeni Özet Oluştur", command=lambda: generate_and_display_headlines(modify_text=True))
rearrange_button.pack()

#Çıktıları tutacak bir window oluştur
output_frame = tk.Frame(window)
output_frame.pack()

#GUI döngüsünü başlat
window.mainloop()