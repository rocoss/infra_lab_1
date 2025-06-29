# train.py

import pandas as pd
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt
from datasets import Dataset, load_from_disk
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig
import torch
import os

# --- Константы и глобальные переменные (для простоты доступа) ---
MODEL_NAME = "unsloth/gemma-3-1b-it"
MAX_SEQ_LENGTH = 2048  # Максимальная длина последовательности для обработки.
SEED = 3407  # Сид для воспроизводимости результатов.
OUTPUT_DIR = "gemma-3_1b_chat_lora_improved_v3"  # Директория для сохранения результатов обучения.
DATASET_DIR = "converted_finetome_dataset"


# --- Основная логика пайплайна обучения ---
def run_training_pipeline():
    """
    Запускает полный пайплайн обучения LoRA-адаптеров для языковой модели,
    включая загрузку данных, конфигурацию модели и тренера, обучение и сохранение.
    """
    print("Текущая рабочая директория:", os.getcwd())

    # Убедимся, что датасет существует
    if not os.path.exists(DATASET_DIR):
        # Используем ValueError вместо raise FileNotFoundError напрямую,
        # чтобы pytest мог легко перехватить эту ошибку
        raise ValueError(f"Датасет по пути {DATASET_DIR} не найден. Убедитесь, что он создан.")

    # ========== Загрузка базовой модели и токенизатора ==========
    # Эти строки будут замоканы в тестах.
    print(f"Загрузка модели {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,  # Включаем 4-битную квантовку для экономии памяти.
    )
    print("Модель и токенизатор загружены.")

    # ========== Применение LoRA-адаптеров ==========
    print("Применение LoRA-адаптеров...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # Ранг LoRA для компромисса между эффективностью и качеством.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],  # Добавляем lm_head для улучшения.
        lora_alpha=32,  # Коэффициент масштабирования LoRA.
        lora_dropout=0,  # Отключение dropout.
        bias="none",  # Без смещения в LoRA-адаптерах.
        use_gradient_checkpointing="unsloth",  # Градиентное чекпоинтирование для снижения потребления памяти.
        random_state=SEED,  # Сид для воспроизводимости.
        use_rslora=False,  # Не используем Rank-Stabilized LoRA.
        loftq_config=None,  # Без дополнительной квантовки LoRA.
    )
    print("LoRA-адаптеры применены.")

    # ========== Загрузка и форматирование датасета ==========
    # Загружаем локальный датасет для диалогов.
    print(f"Загрузка датасета из {DATASET_DIR}...")
    non_reasoning_dataset = load_from_disk(DATASET_DIR)  # Загружаем сохраненный датасет из локальной папки.
    print(f"Загружено {len(non_reasoning_dataset)} записей.")

    # Стандартизируем данные для диалогов и преобразуем их в формат чата.
    print("Стандартизация и форматирование датасета...")
    non_reasoning_standardized = standardize_sharegpt(non_reasoning_dataset)
    # Здесь используется метод токенизатора, который в тестах будет замокан.
    # Он преобразует список диалогов в список форматированных строк.
    non_reasoning_formatted = tokenizer.apply_chat_template(
        non_reasoning_standardized["conversations"],
        tokenize=False
    )
    print("Датасет отформатирован.")

    # ========== Подготовка датасета для SFTTrainer ==========
    # Поскольку используется только один датасет, балансировка не требуется.
    non_reasoning_series = pd.Series(non_reasoning_formatted)
    combined_series = non_reasoning_series  # Используем только диалоговые данные.
    combined_series.name = "text"
    combined_dataset = Dataset.from_pandas(pd.DataFrame(combined_series)).shuffle(seed=SEED)  # Перемешиваем датасет.

    # Разделяем датасет на обучающий и валидационный (90% на обучение, 10% на валидацию)
    split_dataset = combined_dataset.train_test_split(test_size=0.1, seed=SEED)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"Размер обучающего датасета: {len(train_dataset)} записей")
    print(f"Размер валидационного датасета: {len(eval_dataset)} записей")

    # ========== Конфигурация SFTTrainer ==========
    print("Конфигурирование SFTTrainer...")
    sft_config = SFTConfig(
        per_device_train_batch_size=4,  # Увеличиваем размер батча для лучшей стабильности.
        gradient_accumulation_steps=2,  # Уменьшаем накопление градиента, чтобы сохранить общий размер батча.
        #warmup_steps=20,  # Увеличиваем количество шагов разогрева для learning rate.
        max_steps=300,
        num_train_epochs=2,  # Устанавливаем 2 эпохи для полного прохода по данным.
        learning_rate=5e-5,  # Уменьшаем скорость обучения для большей стабильности.
        # Проверка поддержки BF16 перед использованием, иначе FP16.
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,  # Логирование каждые 10 шагов для отслеживания прогресса.
        optim="adamw_8bit",  # Оптимизатор с 8-битной точностью для экономии памяти.
        weight_decay=0.01,  # Регуляризация весов.
        lr_scheduler_type="linear",  # Линейный планировщик скорости обучения.
        seed=SEED,  # Сид для воспроизводимости.
        output_dir=OUTPUT_DIR,  # Директория для сохранения результатов.
        report_to="none",  # Отключаем отчеты (например, в W&B).
    )
    print("SFTTrainer сконфигурирован.")

    # ========== Инициализация и запуск обучения ==========
    print("Инициализация SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,  # Обучающий датасет.
        eval_dataset=eval_dataset,    # Валидационный датасет для оценки качества.
        dataset_text_field="text",    # Поле с текстом в датасете.
        max_seq_length=MAX_SEQ_LENGTH,  # Максимальная длина последовательности.
        args=sft_config  # Конфигурация обучения.
    )

    print("Starting training...")  # Сообщение о начале обучения.
    trainer.train()  # Запускаем процесс обучения. Эта строка будет замокана в тестах.
    print("Training completed.")  # Сообщение о завершении обучения.

    # ========== Сохранение обученной модели ==========
    print(f"Сохранение LoRA-адаптеров и токенизатора в '{OUTPUT_DIR}'...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA-адаптеры и токенизатор сохранены.")


    # ========== Утилита для инференса (пример использования обученной модели) ==========
    # Это вспомогательная функция, которая не является частью основного пайплайна обучения,
    # но полезна для демонстрации использования модели после обучения.
    def inference(message, detailed=False):
        """
        Генерирует ответ модели на заданный промпт.
        """
        formatted_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = tokenizer(formatted_input, return_tensors="pt").to("cuda")
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        print("\n--- {} Inference ---".format("Detailed" if detailed else "Brief"))
        print("Formatted Input:\n", formatted_input)
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=1.0,
            top_p=0.95,
            top_k=50,
            streamer=streamer,
            eos_token_id=tokenizer.eos_token_id
        )
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        cleaned_output = decoded_output.replace("<think>", "").replace("</think>", "").strip()
        print("\nCleaned Output:\n", cleaned_output)
        print("\n-----------------------------")

    # Пример вызова инференса после обучения (только при прямом запуске скрипта)
    if os.path.exists(OUTPUT_DIR): # Проверяем, что модель сохранена
        print("\n--- Пример инференса после обучения ---")
        inference("Как часто вопросы из категории LITERATURE появляются в раунде Final Jeopardy!, и какова их средняя стоимость?", detailed=False)
    else:
        print("\nПропуск примера инференса: модель не была сохранена.")


if __name__ == "__main__":
    try:
        run_training_pipeline()
    except ValueError as e:
        print(f"Ошибка при запуске пайплайна: {e}")
        # Можно добавить sys.exit(1) для выхода с кодом ошибки
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
        # Можно добавить sys.exit(1) для выхода с кодом ошибки
