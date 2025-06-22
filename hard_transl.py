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

# Выберите одну из моделей (раскомментируйте нужную)
MODEL_NAME = 'saiga_nemo_12b'


# MODEL_NAME = 'towerbase-7b-v0.1'
# MODEL_NAME = 'wingpt-babel-2'
# MODEL_NAME = 'salamandrata-7b-instruct'

# -----------------

def get_optimized_prompt(model_name, text_to_translate):
    """Возвращает оптимизированный промпт для конкретной модели."""

    if model_name == 'saiga_nemo_12b':
        return {
            'messages': [
                {
                    "role": "system",
                    "content": """Вы — профессиональный переводчик с глубокими знаниями английского и русского языков. 
                    Ваша задача — создавать точные, естественные переводы, сохраняя стиль, тон и культурный контекст оригинала.

                    Правила перевода:
                    - Сохраняйте исходный смысл и эмоциональную окраску
                    - Используйте естественный русский язык
                    - Учитывайте культурные особенности и идиомы
                    - При переводе терминов выбирайте наиболее подходящий русский эквивалент"""
                },
                {
                    'role': 'user',
                    'content': f"""Переведите следующий английский текст на русский язык:

Текст для перевода: "{text_to_translate}"

Предоставьте только перевод без дополнительных комментариев."""
                }
            ]
        }

    elif model_name == 'towerbase-7b-v0.1':
        return {
            'messages': [
                {
                    "role": "system",
                    "content": """You are a professional translation specialist trained to translate between multiple languages with high accuracy. Focus on producing natural, contextually appropriate translations while preserving the original meaning and style."""
                },
                {
                    'role': 'user',
                    'content': f"""Task: Translate the following text from English to Russian.

Context: General text translation
Source Language: English
Target Language: Russian

Text to translate: "{text_to_translate}"

Instructions:
- Provide an accurate and fluent Russian translation
- Maintain the original tone and style
- Ensure grammatical correctness in Russian
- Output only the translation"""
                }
            ]
        }

    elif model_name == 'wingpt-babel-2':
        return {
            'messages': [
                {
                    "role": "system",
                    "content": "Translate this to Russian Language"
                },
                {
                    'role': 'user',
                    'content': text_to_translate
                }
            ]
        }

    elif model_name == 'salamandrata-7b-instruct':
        return {
            'messages': [
                {
                    "role": "system",
                    "content": """You are a specialized translation model. Translate the given text accurately while preserving meaning, style, and cultural context. Focus on producing natural, high-quality translations."""
                },
                {
                    'role': 'user',
                    'content': f"""Translate from English to Russian:

Source text: {text_to_translate}

Provide a natural and accurate Russian translation."""
                }
            ]
        }

    else:
        # Fallback для неизвестной модели
        return {
            'messages': [
                {"role": "system", "content": "You are an assistant that translates text from English to Russian."},
                {'role': 'user', 'content': f"Translate the following English text to Russian: {text_to_translate}"}
            ]
        }


def translate_text(text_to_translate):
    """Отправляет текст на локальный API и возвращает перевод."""
    if not isinstance(text_to_translate, str) or not text_to_translate.strip():
        return text_to_translate

    headers = {'Content-Type': 'application/json'}

    # Получаем оптимизированный промпт для текущей модели
    prompt_data = get_optimized_prompt(MODEL_NAME, text_to_translate)

    payload = {
        'model': MODEL_NAME,
        'messages': prompt_data['messages'],
        'temperature': 0.1,
        'max_tokens': 1024
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


def print_model_info():
    """Выводит информацию о текущей модели и используемом промпте."""
    model_info = {
        'saiga_nemo_12b': "Русскоязычная модель с контекстно-осведомленным переводом",
        'towerbase-7b-v0.1': "Специализированная переводческая модель для 10 языков",
        'wingpt-babel-2': "Многоязычная модель для 55 языков с минималистичным промптом",
        'salamandrata-7b-instruct': "Европейская переводческая модель для 35 языков"
    }

    print(f"\n📋 Информация о модели:")
    print(f"   Модель: {MODEL_NAME}")
    print(f"   Описание: {model_info.get(MODEL_NAME, 'Неизвестная модель')}")
    print(f"   Оптимизированный промпт: ✅ Активен")


# --- Основной скрипт перевода ---
try:
    print("🚀 Запуск скрипта перевода с оптимизированными промптами")
    print_model_info()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Создана директория для переведенных файлов: '{output_dir}'")

    files_to_process = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not files_to_process:
        print(f"❌ Ошибка: В папке '{input_dir}' не найдено файлов .csv для обработки.")
        print("   Убедитесь, что вы сначала запустили скрипт prepare_data.py")
        exit()

    print(f"\n🔄 Начинаю перевод {len(files_to_process)} файлов из папки '{input_dir}'...")

    total_translations = 0
    start_time = time.time()

    for file_idx, filename in enumerate(files_to_process, 1):
        file_path = os.path.join(input_dir, filename)
        print(f"\n--- 📄 Обработка файла {file_idx}/{len(files_to_process)}: {filename} ---")

        df = pd.read_csv(file_path)
        file_translations = 0

        for index, row in df.iterrows():
            # Переводим вопрос
            original_question = row['Question']
            translated_question = translate_text(original_question)
            df.at[index, 'Question'] = translated_question
            file_translations += 1

            # Переводим ответ
            original_answer = row['Answer']
            translated_answer = translate_text(original_answer)
            df.at[index, 'Answer'] = translated_answer
            file_translations += 1

            print(f"  ✅ Строка {index + 1}/{len(df)}: Переведена (Q+A)")
            time.sleep(0.1)  # Небольшая задержка, чтобы не перегружать API

        total_translations += file_translations

        # Сохранение переведенного файла
        output_filename = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_translated.csv")
        df.to_csv(output_filename, index=False, encoding='utf-8')
        print(f"💾 Файл сохранен как: {output_filename}")
        print(f"📊 Переведено в файле: {file_translations} текстов")

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"\n🎉 Все файлы успешно переведены!")
    print(f"📈 Статистика обработки:")
    print(f"   • Обработано файлов: {len(files_to_process)}")
    print(f"   • Всего переводов: {total_translations}")
    print(f"   • Время обработки: {processing_time:.2f} секунд")
    print(f"   • Средняя скорость: {total_translations / processing_time:.2f} переводов/сек")
    print(f"   • Результаты сохранены в: '{output_dir}'")

except FileNotFoundError:
    print(f"❌ Ошибка: Папка с исходными файлами '{input_dir}' не найдена.")
except Exception as e:
    print(f"💥 Произошла критическая ошибка: {e}")

