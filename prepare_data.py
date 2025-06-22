# Файл: prepare_data.py
import pandas as pd
import os
import re

# --- НАСТРОЙКИ ---
# 1. Путь к исходному большому файлу
source_file_path = '/home/karfel/GitHub/Maga_1_kurs/infra/infra_lab_1/JEOPARDY_CSV.csv'

# 2. Папка, куда будут сохранены очищенные и разделенные файлы
output_dir = 'cleaned_parts'

# 3. Количество частей для разделения
num_parts = 10


# -----------------

def clean_html(raw_html):
    """Удаляет все HTML-теги из строки с помощью регулярных выражений."""
    if not isinstance(raw_html, str):
        return raw_html
    # Шаблон <[^>]+> находит и удаляет любой HTML-тег
    clean_text = re.sub(r'<[^>]+>', '', raw_html)
    return clean_text.strip()


# --- Основной скрипт подготовки данных ---
try:
    # 1. Чтение исходного файла
    print(f"1. Чтение исходного файла: {source_file_path}")
    df = pd.read_csv(source_file_path)

    # 2. Очистка данных
    print("2. Очистка данных от HTML-тегов...")
    # Очищаем заголовки столбцов от лишних пробелов
    df.columns = df.columns.str.strip()

    # Выбираем только нужные столбцы
    df_selected = df[['Question', 'Answer']].copy()

    # Применяем функцию очистки ко всему столбцу. Это намного быстрее, чем в цикле.
    df_selected['Question'] = df_selected['Question'].apply(clean_html)
    df_selected['Answer'] = df_selected['Answer'].apply(clean_html)
    print("   Очистка завершена.")

    # 3. Создание директории для результатов
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"3. Создана директория для результатов: '{output_dir}'")
    else:
        print(f"3. Директория для результатов уже существует: '{output_dir}'")

    # 4. Разделение на части и сохранение
    total_rows = len(df_selected)
    part_size = total_rows // num_parts
    print(f"4. Начинаю разделение {total_rows} строк на {num_parts} файлов...")

    for i in range(num_parts):
        start_row = i * part_size
        end_row = (i + 1) * part_size if i != num_parts - 1 else total_rows

        df_chunk = df_selected.iloc[start_row:end_row]

        output_filename = os.path.join(output_dir, f'part_{i + 1}_clean.csv')
        df_chunk.to_csv(output_filename, index=False)
        print(f"   -> Сохранен файл: {output_filename} ({len(df_chunk)} строк)")

    print(f"\n✅ Задача успешно выполнена. Данные очищены и разделены на {num_parts} файлов.")

except FileNotFoundError:
    print(f"Ошибка: Исходный файл не найден по пути '{source_file_path}'")
except KeyError:
    print("Ошибка: В файле отсутствуют обязательные столбцы 'Question' и/или 'Answer'.")
except Exception as e:
    print(f"Произошла критическая ошибка: {e}")
