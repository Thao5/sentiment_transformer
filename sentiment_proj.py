import asyncio
import base64
import decimal
import json
from datetime import datetime, date
import os

import flask
# import pymysql
import requests
from flask import Flask, render_template, request
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
import openpyxl
import pandas as pd
import numpy as np
import preprocess_kgptalkie as ps
import re
from flask_cors import CORS
import glob
import os
from underthesea import sentiment, pos_tag, word_tokenize
from flask import g
from sklearn.pipeline import Pipeline
import underthesea
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from pandas.errors import ParserError
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import os.path


SIZE_DEMO = 20000

root = '/dataset/**/'
root2 = '/sentiment_transformer/other_dataset/'


# print(all_files)


def load_transformer_model():
    print(os.path.isdir("model"))
    if not os.path.isdir("model"):
        model = SentenceTransformer('keepitreal/vietnamese-sbert')
        model.save("model")
        return model
    return SentenceTransformer('model')


def get_clean(x):
    x = str(x).lower().replace('\\', '').replace('_', ' ')
    # x = str(x).lower()
    x = ps.cont_exp(x)
    x = ps.remove_emails(x)
    x = ps.remove_urls(x)
    x = ps.remove_html_tags(x)
    # x = ps.remove_accented_chars(x)
    # x = ps.remove_special_chars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x


def load_data_train_to_excel(root):
    data_train = pd.DataFrame()
    all_files_train = glob.glob(root + "train/**/*.txt", recursive=True)
    for f in all_files_train:
        df = pd.read_csv(f, header=None, sep=' ', names=['Review', 'Sentiment'], on_bad_lines='skip')
        with open(f, mode='r', encoding="utf8") as f:
            df['Review'] = get_clean(f.read())
            df['Sentiment'] = ""
        data_train = pd.concat([data_train, df], ignore_index=True)
        # print(data)
    data_train.to_csv('data_train.csv', index=False)


# i=-1
# all_files_test = glob.glob(f'{root}test/{i}/*.txt', recursive=True)
# for f in all_files_test:
#     df = pd.read_csv(f, header=None, sep = ' ', names = ['Review','Sentiment'], on_bad_lines='skip')
#     with open(f, mode='r', encoding="utf8") as f:
#         df['Review'] = get_clean(f.read())
#         df['Sentiment'] = "negative"
#     data_test = pd.concat([data_test, df], ignore_index=True)
#     print(data_test)


def load_data_test_to_excel(root):
    data_test = pd.DataFrame()
    sentiment_array = ["negative", "positive", "neutral"]
    all_files_test = []
    for i in [-1, 1, 2]:
        all_files_test = glob.glob(f'{root}test/{i}/*.txt', recursive=True)
        print(i)
        for f in all_files_test:
            df = pd.read_csv(f, header=None, sep=' ', names=['Review', 'Sentiment'], on_bad_lines='skip',
                             quoting=csv.QUOTE_NONE)
            with open(f, mode='r', encoding="utf8") as f:
                df['Review'] = get_clean(f.read())
                df['Sentiment'] = sentiment_array[(i + 1) if i == -1 else i]
            data_test = pd.concat([data_test, df], ignore_index=True)
            # print(data_test
    data_test.to_csv('data_test.csv', index=False)


def load_data_test_to_excel2(root):
    data_test = pd.DataFrame()
    sentiment_array = ["nv_tot", "nv_xau", "shop_tot", "shop_xau", "sp_tot", "sp_xau"]
    all_files_test = []
    for i in sentiment_array:
        all_files_test = glob.glob(f'{root}{i}/*.txt', recursive=True)
        print(i)
        for f in all_files_test:
            df = pd.read_csv(f, header=None, sep=' ', names=['Review', 'Sentiment'], on_bad_lines='skip',
                             quoting=csv.QUOTE_NONE)
            with open(f, mode='r', encoding="utf8") as f:
                df['Review'] = get_clean(f.read())
                df['Sentiment'] = i
            data_test = pd.concat([data_test, df], ignore_index=True)
            # print(data_test
    data_test.to_csv('data_test2.csv', index=False)


# load_data_train_to_excel(root)
# load_data_test_to_excel(root)
# load_data_test_to_excel2(root2)

df = pd.read_csv('data_test2.csv')

# df_train = pd.read_csv('data_train.csv')

# # existing_file = 'Book2.csv'
# # df = pd.read_csv(existing_file)
# # df['Review'] = df['Review'].apply(lambda x: get_clean(x))

# # total_files = len(df)
# # train_size = int(0.6 * total_files)
# # val_size = int(0.2 * total_files)
# # test_size = total_files - train_size - val_size

# # train_files = df[:train_size]
# # val_files = df[train_size:train_size + val_size]
# # test_files = df[train_size + val_size:]


train_texts, test_texts, train_labels, test_labels = train_test_split(df['Review'], df['Sentiment'], test_size=0.3,
                                                                      random_state=0)

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.25,
                                                                    random_state=0)

# print(type(train_texts))

# print(train_labels[0])

nv_tot = train_texts[train_labels[train_labels=="nv_tot"].index]
nv_xau = train_texts[train_labels[train_labels=="nv_xau"].index]
shop_tot = train_texts[train_labels[train_labels=="shop_tot"].index]
shop_xau = train_texts[train_labels[train_labels=="shop_xau"].index]
sp_tot = train_texts[train_labels[train_labels=="sp_tot"].index]
sp_xau = train_texts[train_labels[train_labels=="sp_xau"].index]
nv_tot = nv_tot[:2000]
nv_tot = pd.concat([nv_tot, nv_xau[:2000]])
nv_tot = pd.concat([nv_tot, shop_tot[:2000]])
nv_tot = pd.concat([nv_tot, shop_xau[:2000]])
nv_tot = pd.concat([nv_tot, sp_tot[:2000]])
nv_tot = pd.concat([nv_tot, sp_xau[:2000]])
docs = [underthesea.word_tokenize(sent.lower()) for sent in nv_tot]
model = load_transformer_model()
sentences = []
for sent in docs:
    k = []
    for w in sent:
        k.append(w.replace(" ", "_"))
    sentences.append(" ".join(k))
# print(len(sentences))
embeddings = model.encode(sentences, show_progress_bar=True)


def preprocess(docs):
    return [underthesea.word_tokenize(doc.lower()) for doc in docs]


def get_vocabularies(tokenized_docs):
    vocabs = set()
    for doc in tokenized_docs:
        vocabs.update(doc)
    return vocabs


def identity_tokenizer(text):
    return text


# def create_vector(tokenized_docs):


# tokenized_docs = preprocess(train_texts[:SIZE_DEMO].astype('U'))
# tokenized_docs_val = preprocess(val_texts[:SIZE_DEMO].astype('U'))
# tokenized_docs_test = preprocess(test_texts[:SIZE_DEMO].astype('U'))
# vocabs = get_vocabularies(tokenized_docs)
# print(tokenized_docs)
tokenized_docs = train_texts[:SIZE_DEMO].astype('U')
tokenized_docs_val = val_texts[:SIZE_DEMO].astype('U')
tokenized_docs_test = test_texts[:SIZE_DEMO].astype('U')
#
# tfidf = TfidfVectorizer(analyzer="word", tokenizer=identity_tokenizer, token_pattern=None, lowercase=False)
tfidf = TfidfVectorizer(analyzer="word", tokenizer=underthesea.word_tokenize, token_pattern=None, lowercase=False)
# print(tokenized_docs[:100])
X_train = tfidf.fit_transform(tokenized_docs)
# print(X_train)
X_train.shape
y_train = train_labels[:SIZE_DEMO]
# X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
#                                                                                 random_state=0)
clf = LinearSVC().fit(embeddings, y_train)


# clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
# clf = MultinomialNB().fit(X_train, y_train)
# clf = RandomForestClassifier().fit(X_train, y_train)


def val_process(tfidf, clf):
    X_val = tfidf.transform(tokenized_docs_val)
    val_predictions = clf.predict(X_val)
    val_accuracy = accuracy_score(val_labels[:SIZE_DEMO], val_predictions)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(classification_report(val_labels[:SIZE_DEMO], val_predictions))

    # docs_new = ['God is love', 'OpenGL on the GPU is fast']
    # X_new_tfidf = tfidf.transform(docs_new)

    # y_pred = clf.predict(X_new_tfidf)

    # for doc, category in zip(docs_new, y_pred):
    #     print('%r => %s' % (doc, category))

    # X_new_tfidf = tfidf.transform(test_texts)

    # predicted = clf.predict(X_new_tfidf)

    # np.mean(predicted == test_labels)

    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        # 'max_iter': [1000, 5000, 10000],
    }

    # tree_param = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}

    # parameters = {
    #     'alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)
    # }

    # parameters = {
    #     'max_depth':[3,5,10,None],
    #               'n_estimators':[10,100,200],
    #               'max_features':[1,3,5,7],
    #               'min_samples_leaf':[1,2,3],
    #               'min_samples_split':[1,2,3]
    # }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(LinearSVC(), param_grid, cv=3, refit=True, scoring='accuracy')

    # grid_search = GridSearchCV(tree.DecisionTreeClassifier(), tree_param, cv=3, refit = True, scoring='accuracy')

    # grid_search = GridSearchCV(MultinomialNB(), parameters, cv=3, refit = True, scoring='accuracy')

    # grid_search = GridSearchCV(RandomForestClassifier(), parameters, cv=3, refit = True, scoring='accuracy')

    # Fit the grid search
    grid_search.fit(X_train, train_labels[:SIZE_DEMO])

    # Best parameters
    best_params = grid_search.best_params_
    print(f"Best Parameters: {best_params}")

    # Evaluate best model on validation set
    best_model = grid_search.best_estimator_
    val_predictions = best_model.predict(X_val)
    val_accuracy = accuracy_score(val_labels[:SIZE_DEMO], val_predictions)
    print(f"Validation Accuracy with Best Model: {val_accuracy:.4f}")
    print(classification_report(val_labels[:SIZE_DEMO], val_predictions))


def val_process2(model, clf):
    nv_tot_val = val_texts[val_labels[val_labels == "nv_tot"].index]
    nv_xau_val = val_texts[val_labels[val_labels == "nv_xau"].index]
    shop_tot_val = val_texts[val_labels[val_labels == "shop_tot"].index]
    shop_xau_val = val_texts[val_labels[val_labels == "shop_xau"].index]
    sp_tot_val = val_texts[val_labels[val_labels == "sp_tot"].index]
    sp_xau_val = val_texts[val_labels[val_labels == "sp_xau"].index]
    nv_tot_val = nv_tot_val[:2000]
    nv_tot_val = pd.concat([nv_tot_val, nv_xau_val[:2000]])
    nv_tot_val = pd.concat([nv_tot_val, shop_tot_val[:2000]])
    nv_tot_val = pd.concat([nv_tot_val, shop_xau_val[:2000]])
    nv_tot_val= pd.concat([nv_tot_val, sp_tot_val[:2000]])
    nv_tot_val= pd.concat([nv_tot_val, sp_xau_val[:2000]])
    docs_val = [underthesea.word_tokenize(sent.lower()) for sent in nv_tot_val]
    sentences_val = []
    for sent in docs_val:
        k = []
        for w in sent:
            k.append(w.replace(" ", "_"))
        sentences_val.append(" ".join(k))
    # print(len(sentences))
    embeddings_val = model.encode(sentences_val, show_progress_bar=True)
    # X_val = tfidf.transform(tokenized_docs_val)
    val_predictions = clf.predict(embeddings_val)
    # val_predictions = clf.predict(X_val)
    val_accuracy = accuracy_score(val_labels[:SIZE_DEMO], val_predictions)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(classification_report(val_labels[:SIZE_DEMO], val_predictions))


# val_process(tfidf, clf)

val_process2(model, clf)


def train_demo(df_train, tfidf, clf):
    for index, row in df_train.iterrows():
        X_val = tfidf.transform([row['Review']])
        row['Sentiment'] = str(clf.predict(X_val)).strip('[]\'')
        print(f"{row['Review']}=>{row['Sentiment']}")


def draw_graph(df):
    df_tmp = df
    df_tmp.loc[df_tmp['Sentiment'] == 'positive', 'Sentiment_num'] = 1
    df_tmp.loc[df_tmp['Sentiment'] == 'negative', 'Sentiment_num'] = 0
    df_tmp.loc[df_tmp['Sentiment'] == 'neutral', 'Sentiment_num'] = 2
    # df_tmp2 = pd.DataFrame({
    #     'Sentiment': df_tmp['Sentiment'],
    #     'Sentiment_num': df_tmp['Sentiment_num']
    # })
    plt.figure(figsize=(5,4))
    sns.scatterplot(x='Sentiment', y='Sentiment_num', data=df_tmp)
    plt.show()


def df_transformer(embeddings):
    # nv_tot = df.loc[df['Sentiment'] == "nv_tot"]
    # nv_xau = df.loc[df['Sentiment'] == "nv_xau"]
    # shop_tot = df.loc[df['Sentiment'] == "shop_tot"]
    # shop_xau = df.loc[df['Sentiment'] == "shop_xau"]
    # sp_tot = df.loc[df['Sentiment'] == "sp_tot"]
    # sp_xau = df.loc[df['Sentiment'] == "sp_xau"]
    # nv_tot = nv_tot[:2000]
    # nv_tot = pd.concat([nv_tot, nv_xau.loc[:2000]])
    # nv_tot = pd.concat([nv_tot, shop_tot.loc[:2000]])
    # nv_tot = pd.concat([nv_tot, shop_xau.loc[:2000]])
    # nv_tot = pd.concat([nv_tot, sp_tot.loc[:2000]])
    # nv_tot = pd.concat([nv_tot, sp_xau.loc[:2000]])
    # docs = [underthesea.word_tokenize(sent.lower()) for sent in nv_tot['Review']]
    # model = load_transformer_model()
    # sentences = []
    # for sent in docs:
    #     k = []
    #     for w in sent:
    #         k.append(w.replace(" ", "_"))
    #     sentences.append(" ".join(k))
    # # print(sentences)
    # embeddings = model.encode(sentences, show_progress_bar=True)
    tsne = TSNE(n_components=3, perplexity=2, random_state=42)
    X_tsne = tsne.fit_transform(embeddings)
    ax = plt.figure(figsize=(12,12)).add_subplot(111, projection ="3d")
    # pallete = sns.color_palette('hsv', 3)
    # sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], z=X_tsne[:, 2], hue=pos['Sentiment'], palette=pallete)
    cmap = ListedColormap(sns.color_palette("hsv", 5))
    # groups = pd.DataFrame(X_tsne, columns=['x', 'y', 'z']).assign(category=nv_tot['Sentiment'])
    print(f'{X_tsne.shape}')
    # cdict = {'positive': 'red', 'negative': 'blue', 'neutral': 'green'}
    # for sen in pos['Sentiment'].unique():
    sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=X_tsne[:, 0], cmap=cmap, alpha=0.8)


    plt.title("truc quan hoa du lieu tuyen tinh")
    # plt.legend(*sc.legend_elements())
    # legend1 = ax.legend(*[sc.legend_elements()[0], pos['Sentiment']],
    #                     title="Legend")
    # ax.add_artist(legend1)
    # ax.legend()
    plt.show()


# def df_tfidf(df):
#     pos = df.loc[df['Sentiment'] == "positive"]
#     neg = df.loc[df['Sentiment'] == "negative"]
#     neu = df.loc[df['Sentiment'] == "neutral"]
#     pos = pd.concat([pos, pos.loc[:200]])
#     pos = pd.concat([pos, neg.loc[:200]])
#     pos = pd.concat([pos, neu.loc[:200]])
#     # docs = [underthesea.word_tokenize(sent.lower()) for sent in pos['Review']]
#     # model = None
#     # if not load_transformer_model():
#     #     model = SentenceTransformer('model')
#     # sentences = []
#     # for sent in docs:
#     #     k = []
#     #     for w in sent:
#     #         k.append(w.replace(" ", "_"))
#     #     sentences.append(" ".join(k))
#     # print(sentences)
#     # embeddings = model.encode(sentences, show_progress_bar=True)
#     X_tfidf = tfidf.fit_transform(pos['Review'])
#     pca = PCA(n_components=2).fit_transform(X_tfidf.todense())
#     # X_tsne = pca.transform(X_tfidf.toarray())
#     plt.figure(figsize=(6,4))
#     pallete = sns.color_palette('hsv', 3)
#     sns.scatterplot(x=pca[:, 0], y=pca[:, 1], hue=pos['Sentiment'], palette=pallete)
#
#     plt.title("truc quan hoa du lieu tuyen tinh")
#     plt.show()


# draw_graph(df)


# df_tfidf(df)

# df_transformer(embeddings)