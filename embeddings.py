import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import json
from sentence_transformers import SentenceTransformer



def load_group_metadata(csv_path):
    """Загрузка метаданных групп из CSV."""
    groups_meta = pd.read_csv(csv_path)
    groups_meta['text'] = (groups_meta['name'].fillna('') + ' ' + groups_meta['status'].fillna('')).str.strip()
    return groups_meta

def build_tfidf_matrix(groups_meta, min_df=2):
    """Построение TF-IDF матрицы по текстам групп."""
    vectorizer = TfidfVectorizer(min_df=min_df)
    tfidf_matrix = vectorizer.fit_transform(groups_meta['text'])
    return vectorizer, tfidf_matrix

def load_embeddings(pkl_path):
    """Загрузка эмбеддингов групп из pickle."""
    with open(pkl_path, 'rb') as f:
        emb_map = pickle.load(f)
    return emb_map

def get_group_emb(g, emb_map, groups_meta, vectorizer, tfidf_matrix, sim_threshold=0.45):
    """
    Получение эмбеддинга группы:
    1. Если есть в emb_map — возвращаем.
    2. Иначе ищем ближайшую по TF-IDF.
    """
    gid = g['id']
    if gid in emb_map:
        return emb_map[gid]
    txt = ((g.get('name') or '') + ' ' + (g.get('status') or '')).strip()
    q_vec = vectorizer.transform([txt])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    best_idx = sims.argmax()
    max_sim = sims[best_idx]
    if max_sim < sim_threshold:
        return None
    best_gid = groups_meta.iloc[best_idx]['group_id']
    return emb_map.get(best_gid)

def parse_groups(groups_data):

    parsed_groups = []
    # Если данные переданы как строка, пытаемся преобразовать их в список
    if isinstance(groups_data, str):
        try:
            # Используем ast.literal_eval для безопасного преобразования строки
            groups_data = ast.literal_eval(groups_data)
        except (SyntaxError, ValueError):
            return parsed_groups  # Возвращаем пустой список в случае ошибки
    
    # Проверяем, что groups_data является итерируемым объектом (например, список)
    if not isinstance(groups_data, (list, tuple)):
        return parsed_groups
    
    # Фильтруем элементы, оставляя только словари
    for item in groups_data:
        if isinstance(item, dict):
            parsed_groups.append(item)
    
    return parsed_groups

def get_prompt_embedding(text: str) -> np.ndarray:
    """
    Кодируем строку в вектор той же размерности, что и user_embedding.
    """
    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
        cache_folder="./local_models",
        local_files_only=True
    )
    # convert_to_numpy=True вернёт np.ndarray
    emb = model.encode([text], convert_to_numpy=True)[0]
    return emb  # shape = (dim,)

def recommend_users(
    filtered_df: pd.DataFrame,
    prompt: str,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Для каждого user_embedding из filtered_df считает cosine-similarity
    с embedding промпта и возвращает топ-K пользователей.
    """
    # 3.1) Генерируем embedding промпта
    prompt_emb = get_prompt_embedding(prompt)

    # 3.2) Оставляем только валидные эмбеддинги нужной размерности
    mask = filtered_df['user_embedding'].apply(
        lambda x: isinstance(x, (list, np.ndarray))
                  and len(x) == prompt_emb.shape[0]
    )
    df_valid = filtered_df.loc[mask].copy()
    if df_valid.empty:
        raise ValueError("Нет валидных user_embedding для сравнения.")

    # 3.3) Матрица эмбеддингов shape=(N, dim)
    emb_matrix = np.vstack(df_valid['user_embedding'].values).astype(np.float32)

    # 3.4) Считаем cosine-similarity
    sims = cosine_similarity(
        prompt_emb.reshape(1, -1),
        emb_matrix,
    )[0]

    # 3.5) Добавляем колонку и сортируем
    df_valid['similarity'] = sims
    df_sorted = df_valid.sort_values('similarity', ascending=False)

    # 3.6) Берём топ-K
    return df_sorted.head(top_k)
