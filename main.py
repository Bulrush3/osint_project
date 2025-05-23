import urllib.parse
from sberchat import refine_with_location
from embeddings import load_group_metadata, build_tfidf_matrix, load_embeddings, get_group_emb, parse_groups, recommend_users
from yandex_search import yandex_search_vk_groups
from vk_utils import collect_alive_users_from_groups
import os
import pandas as pd
from filtering import filter_by_query
import json
import datetime
import numpy as np

# 1) Находим группы
query = input("Введите запрос: ")

# фиксируем и выводим время старта работы кода
start = datetime.datetime.now()
refined = refine_with_location(query)
urls = yandex_search_vk_groups(query)
print("Найдено ссылок:", urls)
groups = [int(urllib.parse.urlparse(u).path.split("club")[-1]) for u in urls]

users = []

n_target = 200  # сколько живых пользователей нужно
users = collect_alive_users_from_groups(groups, n_target=n_target)
# 3) Сохраняем в JSON
filename = "users.json"
file_exists = os.path.isfile(filename)

# Если файл уже существует, подгружаем старые данные и добавляем новые
if file_exists:
    with open(filename, mode='r', encoding='utf-8') as f:
        try:
            old_data = json.load(f)
        except json.JSONDecodeError:
            old_data = []
else:
    old_data = []

all_users = old_data + users
with open(filename, mode='w', encoding='utf-8') as f:
    json.dump(all_users, f, ensure_ascii=False, indent=2)

print(f"Сохранено {len(users)} пользователей в {filename}\n")

# Чтение JSON-файла в DataFrame
# query = "Женщины"
df = pd.read_json(filename)
df['sex'] = df['sex'].map({1: 'Женский', 2: 'Мужской'})
users = filter_by_query(query, df)

print("\n Аудитория отфильтрована. Осталось пользователей: ", len(users))

# 1) Загрузка метаданных групп
groups_meta = load_group_metadata('groups_clean.csv')

# 2) Построение TF-IDF
vectorizer, tfidf_matrix = build_tfidf_matrix(groups_meta)

# 3) Загрузка эмбеддингов
emb_map = load_embeddings('groups.pkl')

# 4) Получение эмбеддинга пользователя
results = []
for idx, u in users.iterrows():
    groups = parse_groups(u.get('groups', []))
    embs = [get_group_emb(g, emb_map, groups_meta, vectorizer, tfidf_matrix) for g in groups]
    valid_embs = [e for e in embs if isinstance(e, np.ndarray)]
    if valid_embs:
        user_emb = np.mean(valid_embs, axis=0)
        results.append({
            'user_id': u['user_id'],
            'age': u.get('age'),
            'gender': u.get('sex'),
            'city': u.get('city'),
            'user_embedding': user_emb.tolist()
        })

df = pd.DataFrame(results)

top_users = recommend_users(df, query, top_k=5)

print(top_users[["user_id", "city", "age", "gender", "similarity"]])

#фиксируем и выводим время окончания работы кода
finish = datetime.datetime.now()

# вычитаем время старта из времени окончания
print('Время работы: ' + str(finish - start))