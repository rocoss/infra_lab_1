# Файл: translate_files.py
import pandas as pd
import os
import requests
import time

# --- НАСТРОЙКИ ---
# 1. Папка, где лежат очищенные файлы из Скрипта 1
input_dir = 'cleaned_parts'

# 2. Папка, куда будут сохранены переведенные файлы
output_dir = 'translated_parts'

# 3. Настройки API вашей локальной модели в LM-Studio
API_URL = 'http://localhost:1234/v1/chat/completions'
#MODEL_NAME = 'saiga_nemo_12b'
# модели чисто для перевода
#MODEL_NAME = 'towerbase-7b-v0.1'
#MODEL_NAME = 'wingpt-babel-2'
MODEL_NAME = 'salamandrata-7b-instruct'


# -----------------

def translate_text(text_to_translate):
    """Отправляет текст на локальный API и возвращает перевод."""
    if not isinstance(text_to_translate, str) or not text_to_translate.strip():
        return text_to_translate

    headers = {'Content-Type': 'application/json'}
    payload = {
        'model': MODEL_NAME,
        'messages': [
            {"role": "system", "content": "You are an assistant that translates text from English to Russian."},
            {'role': 'user', 'content': f"Translate the following English text to Russian: {text_to_translate}"}
        ],
        'temperature': 0.1, 'max_tokens': 1024
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        print(f"!! Ошибка сети: {e}")
        return text_to_translate  # При ошибке возвращаем исходный текст
    except (KeyError, IndexError):
        print("!! Неверный формат ответа от API.")
        return text_to_translate
    except Exception as e:
        print(f"!! Непредвиденная ошибка при переводе: {e}")
        return text_to_translate


# --- Основной скрипт перевода ---
try:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана директория для переведенных файлов: '{output_dir}'")

    files_to_process = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not files_to_process:
        print(f"Ошибка: В папке '{input_dir}' не найдено файлов .csv для обработки. "
              "Убедитесь, что вы сначала запустили скрипт prepare_data.py")
        exit()

    print(f"Начинаю перевод {len(files_to_process)} файлов из папки '{input_dir}'...")

    for filename in files_to_process:
        file_path = os.path.join(input_dir, filename)
        print(f"\n--- Обработка файла: {filename} ---")

        df = pd.read_csv(file_path)

        for index, row in df.iterrows():
            # Переводим вопрос
            df.at[index, 'Question'] = translate_text(row['Question'])

            # Переводим ответ
            df.at[index, 'Answer'] = translate_text(row['Answer'])

            print(f"  Строка {index + 1}/{len(df)}: Переведена.")
            time.sleep(0.1)  # Небольшая задержка, чтобы не перегружать API

        # Сохранение переведенного файла
        output_filename = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_translated.csv")
        df.to_csv(output_filename, index=False)
        print(f"-> Файл сохранен как: {output_filename}")

    print(f"\n✅ Все файлы успешно переведены и сохранены в папке '{output_dir}'.")

except FileNotFoundError:
    print(f"Ошибка: Папка с исходными файлами '{input_dir}' не найдена.")
except Exception as e:
    print(f"Произошла критическая ошибка: {e}")
