import pandas as pd
from datasets import load_from_disk, Dataset
import os

# Однозначный путь к файлу с данными CSV
csv_data_path = "./JEOPARDY_CSV.csv"

# Проверяем, существует ли файл с данными
if not os.path.exists(csv_data_path):
    print(f"Ошибка: Файл {csv_data_path} не найден. Убедитесь, что файл находится в правильной директории.")
    exit(1)

# Загружаем данные из CSV файла
def read_csv_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Успешно загружено {len(df)} записей из CSV файла.")
        # Выводим исходные названия столбцов для диагностики
        print("Исходные названия столбцов в CSV файле:", list(df.columns))
        # Переименовываем колонки в ожидаемый формат, предполагая, что они идут в правильном порядке
        expected_columns = ["Show Number", "Air Date", "Round", "Category", "Value", "Question", "Answer"]
        if len(df.columns) == len(expected_columns):
            df.columns = expected_columns
            print("Колонки переименованы в:", list(df.columns))
        else:
            print(f"Ошибка: Количество столбцов ({len(df.columns)}) не соответствует ожидаемому ({len(expected_columns)}).")
            exit(1)
        return df
    except Exception as e:
        print(f"Ошибка при чтении CSV файла: {e}")
        exit(1)

csv_data = read_csv_data(csv_data_path)

# Путь к уже сохраненному датасету
output_dir = "converted_finetome_dataset"

# Загружаем существующий датасет, если он есть
if os.path.exists(output_dir):
    print(f"Загрузка существующего датасета из {output_dir}...")
    existing_dataset = load_from_disk(output_dir)
    # Преобразуем в DataFrame для удобства обработки
    existing_df = pd.DataFrame(existing_dataset)
    # Проверяем количество записей до удаления дубликатов
    initial_count = len(existing_df)
    # Удаляем дубликаты из существующего датасета на основе строкового представления 'conversations'
    existing_df['conv_str'] = existing_df['conversations'].apply(lambda x: str(x))
    existing_df = existing_df.drop_duplicates(subset=['conv_str'], keep='first')
    existing_df = existing_df.drop(columns=['conv_str'])
    # Проверяем количество записей после удаления дубликатов
    final_count = len(existing_df)
    print(f"Удалено дубликатов из существующего датасета: {initial_count - final_count}")
    print(f"Количество уникальных записей в существующем датасете: {final_count}")
else:
    print("Существующий датасет не найден. Создается новый датасет.")
    existing_df = pd.DataFrame()

# Функция конвертации данных из CSV в формат FineTome-100k
def convert_csv_to_finetome_format(df):
    converted = []
    for _, row in df.iterrows():
        # Используем переименованные столбцы
        question_col = 'Question'
        answer_col = 'Answer'
        if question_col in row.index and answer_col in row.index:
            conversation = [
                {"from": "human", "value": str(row[question_col])},
                {"from": "gpt", "value": str(row[answer_col])}
            ]
            converted.append({
                "conversations": conversation,
                "source": "jeopardy_csv",
                "score": 5.0
            })
        else:
            print(f"Пропущена запись: отсутствуют столбцы {question_col} или {answer_col}")
    return converted

# Конвертируем данные из CSV
print("Конвертация данных из CSV файла...")
converted_csv_data = convert_csv_to_finetome_format(csv_data)

# Преобразуем в DataFrame для сравнения
new_data_df = pd.DataFrame(converted_csv_data)

# Проверяем, есть ли дубликаты среди новых данных
new_data_df['conv_str'] = new_data_df['conversations'].apply(lambda x: str(x))
initial_new_count = len(new_data_df)
new_data_df = new_data_df.drop_duplicates(subset=['conv_str'], keep='first')
final_new_count = len(new_data_df)
print(f"Удалено дубликатов среди новых данных: {initial_new_count - final_new_count}")

# Проверяем, есть ли дубликаты между новыми данными и существующим датасетом
if not existing_df.empty:
    print("Проверка на дубликаты между новым и существующим датасетом...")
    existing_conversations = existing_df['conversations'].apply(lambda x: str(x))
    new_conversations = new_data_df['conversations'].apply(lambda x: str(x))
    # Оставляем только те новые записи, которых нет в существующем датасете
    unique_new_data_df = new_data_df[~new_conversations.isin(existing_conversations)]
    print(f"Найдено {len(new_data_df) - len(unique_new_data_df)} дубликатов с существующим датасетом. Они будут исключены.")
else:
    unique_new_data_df = new_data_df

# Удаляем временный столбец из new_data_df, если он есть
if 'conv_str' in unique_new_data_df.columns:
    unique_new_data_df = unique_new_data_df.drop(columns=['conv_str'])

# Объединяем уникальные новые данные с существующими
if not existing_df.empty:
    combined_df = pd.concat([existing_df, unique_new_data_df], ignore_index=True)
else:
    combined_df = unique_new_data_df

# Преобразуем обратно в Dataset
combined_dataset = Dataset.from_pandas(combined_df)

# Сохраняем обновленный датасет
print(f"Сохранение обновленного датасета в {output_dir}...")
combined_dataset.save_to_disk(output_dir)

# Выводим информацию о количестве записей
print(f"Количество новых уникальных записей: {len(unique_new_data_df)}")
print(f"Общее количество записей после добавления: {len(combined_dataset)}")

# Проверяем содержимое папки
saved_files = os.listdir(output_dir)
print("Сохраненные файлы в папке:", saved_files)
