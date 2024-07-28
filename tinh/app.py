import streamlit as st
from pathlib import Path
import os
from transformers import GPT2Tokenizer
import tensorflow as tf
import numpy as np
import gdown

BASE_DIR = Path(__file__).resolve(strict=True).parent

tokenizer = GPT2Tokenizer.from_pretrained('NlpHUST/gpt2-vietnamese')
tokenizer.pad_token = tokenizer.eos_token


def get_model(weights_path):
    if not os.path.exists(weights_path):
        file_id = '13ZG5HOVqf6QY7TyPDiEAyjzpSq1IwJc2'
        url = f'https://drive.google.com/uc?id={file_id}'

        gdown.download(url, weights_path, quiet=False)

    return tf.keras.models.load_model(weights_path)


weights_path = f'{BASE_DIR}/weights_sentiment_model.h5'
classifier = get_model(weights_path)

sentiments = ['đánh giá tốt về nhân viên',
              'đánh giá xấu về nhân viên',
              'đánh giá tốt về shop bán hàng',
              'đánh giá xấu về shop bán hàng',
              'đánh giá tốt về sản phẩm',
              'đánh giá xấu về sản phẩm']

# Streamlit app
st.title('Phân loại các đánh giá trên Shopee')

comment = st.text_area("Nhập đánh giá bạn muốn phân loại: ")

if st.button('Phân loại'):
    if comment:
        # Transform the input text using the loaded CountVectorizer
        feature = tokenizer.encode_plus(comment, add_special_tokens=True, padding='max_length',
                                        max_length=293, return_tensors='tf')['input_ids'].numpy()
        feature = feature[0][:, np.newaxis]
        feature = tf.expand_dims(feature, axis=0)
        pred = classifier.predict(feature, verbose=False)
        label = pred.argmax(axis=1)[0]
        # print(label)
        sentiment = sentiments[label]
        conf = np.max(pred) * 100

        st.write(f"Đánh giá là một {sentiment}")
        st.write(f"Độ chính xác: {conf:.2f}%")

    else:
        st.write('Please enter an SMS message to classify.')
