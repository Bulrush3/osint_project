from langchain_gigachat.chat_models import GigaChat
from langchain.schema import HumanMessage, SystemMessage

giga = GigaChat(
    credentials="MmYzNjJhMGEtYjdmOS00OTdiLWI0YzgtZDk2ZmU5NzhmYmM2OjM2ZTdjZjYyLThhOTctNDNmZi1iYjlkLTdlYjI4NjE2OGNiZQ==",
    model="GigaChat-preview",
    verify_ssl_certs=False
)


def parse_suggestions(text):
    """
    Из ответа LLM, вида:
      1) «вариант A»
      2) «вариант B»
      ...
    возвращает список строк-«вариантов» без нумерации.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    opts = []
    for line in lines:
        # предполагаем формат "N) «текст»"
        if ')' in line:
            _, rest = line.split(')', 1)
            opts.append(rest.strip().strip('«»"'))
    return opts

def refine_query(raw_query, max_attempts=3):
    # системный промпт
    msgs = [
        SystemMessage(content='''
        Ты — ассистент, задача которого — брать "сырой" пользовательский запрос 
        и выдавать 4–5 уточнённых чётко сформулированных вариантов для поиска целевой аудитории. 
        Фокусируйся на профессии, возрасте, поле, городе, интересах, образовании 
        и доходе, когда это уместно. Не задавай вопросов — сразу предлагай варианты.
        Пример:
        Вход: «айтишники питер»
        Выход:
        1) «разработчики ПО 25–40 лет в Санкт-Петербурге»
        2) «DevOps-инженеры и системные админы в СПб»
        3) «студенты IT-специальностей вузов Санкт-Петербурга»
        4) «frontend-разработчики с опытом 1–3 года в Питере»
        5) «женщины-программисты 30–45 лет в Ленобласти»
        ''')
    ]
    attempts = 0
    while attempts < max_attempts:
        msgs.append(HumanMessage(content=raw_query))
        answer = giga(msgs)
        options = parse_suggestions(answer.content)
        
        # Покажем варианты пользователю
        print("\nВозможные варианты:")
        for i, opt in enumerate(options, start=1):
            print(f"  {i}) {opt}")
        print("  0) Попробовать ещё раз\n")
        
        choice = input("Введите номер варианта (или 0 для новой генерации): ").strip()
        if choice == '0':
            attempts += 1
            print(f"Повторная генерация ({attempts}/{max_attempts})...\n")
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            selected = options[int(choice) - 1]
            print(f"\nВы выбрали: «{selected}»")
            return selected
        else:
            print("Неверный ввод, попробуйте снова.")
    
    # Если попытки исчерпаны — по умолчанию берём первый вариант
    print("Максимум попыток исчерпан — используем первый вариант.")
    return options[0]

# Функция-детектор «поблизости»
def needs_location(query):
    triggers = ["рядом", "поблизости", "возле", "недалеко", "вблизи"]
    q = query.lower()
    return any(term in q for term in triggers)

# Функция, которая спрашивает город у пользователя
def ask_user_location():
    return input("Похоже, вы ищете что-то поблизости. Укажите, пожалуйста, ваш город или район: ").strip()

# Обёртка, которая сначала проверяет запрос, потом вызывает refine_query
def refine_with_location(raw_query):
    if needs_location(raw_query):
        city = ask_user_location()
        # Подставляем город в запрос. Если в raw_query уже есть предлог,
        # просто добавляем в конец:
        raw_query = f"{raw_query} {city}"
    # Теперь точно передаём «сырый» (но уже с городом) запрос в LLM-рефайнер
    return refine_query(raw_query)

# Пример использования:
if __name__ == "__main__":
    # raw = "девушки казань"
    # refined = refine_with_location(user_input)
    
    # Далее «refined» отправляем в ваш следующий шаг пайплайна:
    # result = your_next_pipeline_step(refined)
    print("Результат обработки уточнённого запроса:", refined)