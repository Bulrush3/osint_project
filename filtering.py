import re
import pymorphy3
import pandas as pd
from datetime import datetime

morph = pymorphy3.MorphAnalyzer()

# 1) Демонимы (в нормальной форме) → канонический город
DEMONYMS_RAW = {
    'москвич':         'Москва',
    'питерец':         'Санкт-Петербург',
    'петербуржец':     'Санкт-Петербург',
    'свердловчанин':   'Екатеринбург',
    # ... 
}
DEMONYMS = {
    morph.parse(demon)[0].normal_form: city
    for demon, city in DEMONYMS_RAW.items()
}

# 2) Синонимы / варианты написания города → каноническое имя
CITY_SYNONYMS_RAW = {
    'Санкт-Петербург': ['санкт-петербург', 'санктпетербург', 'петербург', 'питер'],
    'Москва':           ['москва'],
    'Екатеринбург':     ['екатеринбург', 'екб', 'свердловск'],
    # ... и т.д.
}

CANONICAL = {
    syn: city
    for city, syns in CITY_SYNONYMS_RAW.items()
    for syn in syns
}

# предопределённые «категории»
CATEGORIES = {
'подросток': (13, 19),
'школьник': (7, 18),
'студент':  (17, 25),
'пенсионер': (60, 120),
}

def _tokenize(text: str) -> list[str]:
    """
    Разбиваем на «слова»: буквы, цифры и дефис.
    """
    return re.findall(r'[\w\-]+', text.lower())


def init_city_lemmas(city_list: list[str]) -> dict[str, set[str]]:
    """
    Для каждого города в city_list возвращает множество его лемм.
    Например, "Нижний Новгород" → {"нижний", "новгород"}.
    """
    city_lemmas: dict[str, set[str]] = {}
    for city in city_list:
        toks = _tokenize(city)
        city_lemmas[city] = {morph.parse(tok)[0].normal_form for tok in toks}
    return city_lemmas


def parse_city(
    query: str,
    city_list: list[str],
    city_lemmas: dict[str, set[str]]
) -> str | None:
    """
    Ищет в запросе упоминание города и возвращает точное имя из city_list,
    либо None. Логика:
      1) Демонимы: "москвичи", "питерцы" → сразу Москва/S-P
      2) Синонимы: "питер", "свердловск" → S-P/Екб
      3) Прямое лемматизированное вхождение: если лемма токена
         совпадает с леммой любого слова из названия города.
    """
    tokens = _tokenize(query)
    lemmas = [morph.parse(tok)[0].normal_form for tok in tokens]

    # 1) Демонимы
    for lem in lemmas:
        if lem in DEMONYMS:
            return DEMONYMS[lem]

    # 2) Синонимы
    for tok in tokens:
        if tok in CANONICAL:
            return CANONICAL[tok]

    # 3) Прямой лемматизированный match
    lemma_set = set(lemmas)
    for city, lem_set in city_lemmas.items():
        # если хоть одна лемма из названия города встречается в запросе
        if lemma_set & lem_set:
            return city

    # ничего не нашли
    return None

def parse_gender(query: str):
    q = query.lower()

    FEMALE_TERMS = ['женщины', 'школьницы', 'студентки' 'девушки', 'пенсионерки']
    MALE_TERMS   = ['мужчины', 'мужики', 'парни', 'парней', 'парни', 'мужской']

    if any(tok in q for tok in FEMALE_TERMS):
        return 'Женский'
    if any(tok in q for tok in MALE_TERMS):
        return 'Мужской'
    return None


def parse_age(query: str):
    q = query.lower()

    # 1) явный диапазон: "30-40 лет"
    m = re.search(r'(\d+)\s*-\s*(\d+)\s*лет', q)
    if m:
        return int(m.group(1)), int(m.group(2))

    # 2) открытые границы
    m = re.search(r'от\s*(\d+)\s*лет', q)
    if m:
        return int(m.group(1)), None

    m = re.search(r'до\s*(\d+)\s*лет', q)
    if m:
        return None, int(m.group(1))
    
    
    # 3) категории: "студенты", "пенсионеры" и т.д.
    for cat, (mn, mx) in CATEGORIES.items():
        if cat in q or (cat + 'ы') in q:
            return mn, mx

    # ничего не нашли
    return None, None

# Функция расчёта возраста
def calculate_age(bdate_str):
    try:
        parts = bdate_str.split('.')
        if len(parts) == 3:
            day, month, year = map(int, parts)
        elif len(parts) == 2:
            month, year = map(int, parts)
            day = 1
        else:
            return None
        birth = datetime(year, month, day)
        today = datetime.today()
        age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        return age
    except:
        return None

def filter_by_query(query: str, df: pd.DataFrame):
    # 0) предварительно рассчитываем возраст один раз
    if 'age' not in df.columns:
        df['age'] = df['bdate'].apply(calculate_age)

    # 1) возраст
    min_age, max_age = parse_age(query)
    print('Возрастной диапазон: от', min_age, 'до', max_age)
    # 2) пол
    gender = parse_gender(query)
    print('Пол запроса:', gender)
    # 3) город
    city_list   = df['city'].dropna().unique().tolist()
    city_lemmas = init_city_lemmas(city_list)
    city        = parse_city(query, city_list, city_lemmas)

    print('Город:', city)
    # 4) собственно фильтрация
    res = df.copy()
    if min_age is not None:
        res = res[res['age'] >= min_age]
    if max_age is not None:
        res = res[res['age'] <= max_age]
    if gender:
        res = res[res['sex'] == gender]
    if city:
        res = res[res['city'].str.contains(city, case=False, na=False)]
    return res