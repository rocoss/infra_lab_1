import pandas as pd
import requests
import time
import os
from datasets import Dataset

# Путь к файлу с данными CSV (если файл доступен)
csv_data_path = "/home/karfel/GitHub/Maga_1_kurs/infra/infra_lab_1/JEOPARDY_CSV.csv"

# Проверяем, существует ли файл с данными
if os.path.exists(csv_data_path):
    try:
        csv_data = pd.read_csv(csv_data_path)
        print(f"Успешно загружено {len(csv_data)} записей из CSV файла.")
        # Переименовываем колонки в ожидаемый формат
        expected_columns = ["Show Number", "Air Date", "Round", "Category", "Value", "Question", "Answer"]
        if len(csv_data.columns) == len(expected_columns):
            csv_data.columns = expected_columns
            print("Колонки переименованы в:", list(csv_data.columns))
        else:
            raise ValueError(f"Количество столбцов ({len(csv_data.columns)}) не соответствует ожидаемому ({len(expected_columns)}).")
    except Exception as e:
        print(f"Ошибка при чтении CSV файла: {e}")
        exit(1)
else:
    print(f"Файл {csv_data_path} не найден. Используется пример данных для демонстрации.")
    # Создаем пример данных, имитирующих структуру JEOPARDY_CSV.csv
    example_data = {
        "Show Number": [4680, 4680],
        "Air Date": ["2004-12-31", "2004-12-31"],
        "Round": ["Jeopardy!", "Jeopardy!"],
        "Category": ["HISTORY", "ESPN's TOP 10 ALL-TIME ATHLETES"],
        "Value": ["$200", "$200"],
        "Question": ["For the last 8 years of his life, Galileo was under house arrest for espousing this man's theory", "No. 2: 1912 Olympian; football star at Carlisle Indian School; 6 MLB seasons with the Reds, Giants & Braves"],
        "Answer": ["Copernicus", "Jim Thorpe"]
    }
    csv_data = pd.DataFrame(example_data)

# Функция для перевода текста через API Яндекс Переводчика
def translate_text_yandex(texts, api_key, folder_id, target_lang='ru'):
    translated_texts = []
    url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {api_key}"
    }
    for text in texts:
        if not text or pd.isna(text):
            translated_texts.append("")
            continue
        body = {
            "folderId": folder_id,
            "texts": [text],
            "targetLanguageCode": target_lang
        }
        try:
            response = requests.post(url, json=body, headers=headers)
            if response.status_code == 200:
                translated_text = response.json()["translations"][0]["text"]
                translated_texts.append(translated_text)
            else:
                print(f"Ошибка перевода для текста '{text}': {response.status_code}, {response.text}")
                translated_texts.append(text)  # Возвращаем исходный текст в случае ошибки
        except Exception as e:
            print(f"Исключение при переводе текста '{text}': {e}")
            translated_texts.append(text)  # Возвращаем исходный текст в случае ошибки
        time.sleep(0.1)  # Задержка, чтобы не превысить лимит запросов
    return translated_texts

# Укажите ваш API-ключ и folder_id для Yandex Translator
api_key = 'your_yandex_api_key'  # Замените на реальный ключ
folder_id = 'your_folder_id'      # Замените на реальный folder_id

# Переводим столбцы Question и Answer
questions = csv_data['Question'].tolist()
answers = csv_data['Answer'].tolist()

print("Перевод вопросов...")
translated_questions = translate_text_yandex(questions, api_key, folder_id, target_lang='ru')
print("Перевод ответов...")
translated_answers = translate_text_yandex(answers, api_key, folder_id, target_lang='ru')

# Добавляем переведенные данные в DataFrame
csv_data['Question_translated'] = translated_questions
csv_data['Answer_translated'] = translated_answers

# Выводим результат для проверки
print("Первые строки с переведенными данными:")
print(csv_data[['Question', 'Question_translated', 'Answer', 'Answer_translated']].head())

# Создаем датасет из DataFrame с переведенными столбцами
converted = []
for _, row in csv_data.iterrows():
    conversation = [
        {"from": "human", "value": row['Question_translated']},
        {"from": "gpt", "value": row['Answer_translated']}
    ]
    converted.append({
        "conversations": conversation,
        "source": "jeopardy_csv_translated",
        "score": 5.0
    })

new_data_df = pd.DataFrame(converted)

# Сохраняем в датасет
output_dir = "converted_finetome_dataset"

# Проверяем, существует ли папка
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

combined_dataset = Dataset.from_pandas(new_data_df)
combined_dataset.save_to_disk(output_dir)

# Проверяем содержимое папки
saved_files = os.listdir(output_dir)
print("Сохраненные файлы в папке:", saved_files)
