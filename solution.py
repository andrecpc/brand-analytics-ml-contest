# Импорты

import sys
import pandas as pd
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import math
import jsonlines
import spacy
import random
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

_, TASK_TYPE, INPUT_PATH, OUTPUT_PATH = sys.argv
print('TASK_TYPE, INPUT_PATH, OUTPUT_PATH:', TASK_TYPE, INPUT_PATH, OUTPUT_PATH)

# Фиксирование сидов
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
set_seed()

# Функция, которая будет вырезать лишние вводные конструкции из ответов,
# чтобы приблизить ответы к эталонам организаторов
repl = pd.read_csv('src/replacements_long.csv')
repl.columns = ['r']
repl = list(repl.r.values)

def repls(x):

    if x[0]=='"' and x[-1]=='"':
        x = x[1:-1]

    very_pop_intro = 'Комментарии варьируются от положительных до отрицательных. '
    if very_pop_intro in x[0:len(very_pop_intro)] and len(x)*2 > len(very_pop_intro):
        x = x.replace(very_pop_intro, '')

    for _ in range(10):
        for el in repl:
            if el in x[0:len(el)]:
                x = x.replace(el, '')

    return x.capitalize()

# Сразу разбираемся с кудой и моделями, чтобы словить возможные ошибки

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
new_model_name = 'src/my_custom_model_4'
model = T5ForConditionalGeneration.from_pretrained(new_model_name, local_files_only=True).to(device).eval()
tokenizer = T5Tokenizer.from_pretrained(new_model_name, local_files_only=True)
nlp = spacy.load('ru_core_news_sm')

# Парсинг jsonl

def read_jsonl(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def process_data(data):
    posts = []
    comments = []
    for el in data:
        if 'root_id' not in el:
            posts.append(el)
        else:
            comments.append(el)
    return posts, comments

posts, comments = process_data(read_jsonl(INPUT_PATH))

# pd.DataFrame(posts).to_pickle('posts.pkl')
# pd.DataFrame(comments).to_pickle('comments.pkl')
posts = pd.DataFrame(posts)
comments = pd.DataFrame(comments)

comments['date'] = pd.to_datetime(comments['date'], unit='s')
comments.sort_values(by='date', inplace=True)



# posts = posts[0:5] # !!!!!!



# Получение суммаризаций текстов постов
# Чтобы тексты постов были короче, и косинусные меры считались быстрее

N = posts.shape[0]
SEQ_LEN = 1500
texts = [el[0:SEQ_LEN] for el in posts.text.values]

BATCH_SIZE = 8
steps = math.ceil(N / BATCH_SIZE)
result = []

for i in range(steps):
    if i != steps:
        batch = texts[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    else:
        batch = texts[i*BATCH_SIZE:]

    input_ids = tokenizer(batch, return_tensors='pt', truncation=True,
                          max_length=SEQ_LEN, padding=True).input_ids.to(device)
    outputs = model.generate(input_ids)
    for el in outputs:
        summary = tokenizer.decode(el, skip_special_tokens=True)
        summary = repls(summary)
        result.append(summary)

posts['text_short'] = result


# Подготовка текстов комментов в зависимости от подзадачи

N = posts.shape[0] # posts.shape[0]
SEQ_LEN = 1500

if TASK_TYPE == 'all_comments':

    posts_idx = []
    posts_hashes = []
    comments_hashes = []
    texts = []

    for index, row in posts[0:N].iterrows():
        id = row['id']
        parent_hash = row['hash']
        posts_idx.append(id)
        posts_hashes.append(parent_hash)

        sl = comments.loc[comments.parent_id==id]
        text = ' '.join(sl.text.values)[0:SEQ_LEN]
        childrens_hashes = list(sl.hash.values)
        comments_hashes.append(childrens_hashes)
        texts.append(text)


elif TASK_TYPE == 'post_comments':

    posts_idx = []
    posts_hashes = []
    comments_hashes = []
    texts = []

    for index, row in posts[0:N].iterrows():
        id = row['id']
        parent_hash = row['hash']
        posts_idx.append(id)
        posts_hashes.append(parent_hash)

        sl = comments.loc[comments.parent_id==id]
        sl['cos_sim'] = sl.text.apply(lambda x: nlp(row.text_short).similarity(nlp(x)))
        median = sl['cos_sim'].median()
        sl = sl[sl.cos_sim>=median]
        if sl.shape[0] == 0:
            sl = comments.loc[comments.parent_id==id]
        text = ' '.join(sl.text.values)[0:SEQ_LEN]
        childrens_hashes = list(sl.hash.values)
        comments_hashes.append(childrens_hashes)
        texts.append(text)


elif TASK_TYPE == 'topic_comments':

    posts_idx = []
    posts_hashes = []
    comments_hashes = []
    texts = []

    for index, row in posts[0:N].iterrows():
        id = row['id']
        parent_hash = row['hash']
        posts_idx.append(id)
        posts_hashes.append(parent_hash)

        sl = comments.loc[comments.parent_id==id]
        sl['cos_sim'] = sl.text.apply(lambda x: nlp(row.text_short).similarity(nlp(x)))
        median = sl['cos_sim'].median()
        quantile = sl['cos_sim'].quantile(0.1)
        sl = sl[(sl.cos_sim<median) & (sl.cos_sim>quantile)]
        if sl.shape[0] == 0:
            sl = comments.loc[comments.parent_id==id]
        text = ' '.join(sl.text.values)[0:SEQ_LEN]
        childrens_hashes = list(sl.hash.values)
        comments_hashes.append(childrens_hashes)
        texts.append(text)


# Получение суммаризаций

BATCH_SIZE = 8
steps = math.ceil(N / BATCH_SIZE)
result = []

for i in range(steps):
    if i != steps:
        batch = texts[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    else:
        batch = texts[i*BATCH_SIZE:]

    input_ids = tokenizer(batch, return_tensors='pt', truncation=True,
                          max_length=SEQ_LEN, padding=True).input_ids.to(device)
    outputs = model.generate(input_ids)
    for el in outputs:
        summary = tokenizer.decode(el, skip_special_tokens=True)
        summary = repls(summary)
        result.append(summary)


# Дополнительная обработка ответов
# Если в ответе один из популярных неполезных ответов, то делаем dummy summary

useless = [
    'Комментарии варьируются от положительных до отрицательных.',
    'Фотографии и ссылки на них.',
    'Пользователи выражают свое мнение о различных аспектах жизни.',
    'Пользователи выражают свою поддержку и поддержку автору поста.',
    'Участие в конкурсе.'
]
for i, res in enumerate(result):
    if res in useless:
        result[i] = result[i] + ' Например: ' + texts[i][0:100]


# Наполнение выходного файла

data = []

for i in range(N):
    data.append({"summary": result[i],
                 "post_hash": posts_hashes[i],
                 "comments_hash": comments_hashes[i]})
with jsonlines.open(OUTPUT_PATH, 'w') as writer:
    writer.write_all(data)
