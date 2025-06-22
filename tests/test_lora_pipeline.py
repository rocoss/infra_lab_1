# tests/test_lora_pipeline.py

"""
Модульные тесты для пайплайна обучения модели train.py.

Эти тесты используют моки (mocks) для имитации ресурсоемких операций,
таких как загрузка моделей с Hugging Face, обучение на GPU и работа с файловой системой.
Это позволяет быстро и надежно проверять корректность логики конфигурации,
обработки данных и последовательности вызовов, не запуская реальное обучение.

Для запуска тестов выполните в корневой директории проекта:
pytest -v
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from datasets import Dataset

# Импортируем основной модуль, который мы будем тестировать.
# Это безопасно, так как его код обернут в if __name__ == "__main__":
import train


# --- Фикстуры для настройки тестовой среды ---

@pytest.fixture
def mock_dataset_dir(tmp_path):
    """
    Создает временную директорию с фейковым датасетом для тестов.
    Это делает тесты независимыми от реальных данных и ускоряет их выполнение.
    `tmp_path` - это встроенная фикстура pytest для работы с временными файлами.
    """
    dataset_dir = tmp_path / "converted_finetome_dataset"
    dataset_dir.mkdir()

    # Создаем минимальный, но валидный датасет из 10 записей
    dummy_data = {
        "conversations": [[{"from": "human", "value": f"q{i}"}, {"from": "gpt", "value": f"a{i}"}]] for i in range(10)
    }
    dummy_dataset = Dataset.from_dict(dummy_data)
    dummy_dataset.save_to_disk(dataset_dir)
    return str(dataset_dir)


@pytest.fixture
def mock_dependencies(mocker):
    """
    Фикстура для "мока" всех внешних и тяжелых зависимостей.
    `mocker` - это фикстура из плагина pytest-mock.
    """
    # Создаем фейковые объекты модели и токенизатора
    mock_model = MagicMock(name="MockModel")
    mock_tokenizer = MagicMock(name="MockTokenizer")
    # Настраиваем метод, чтобы он возвращал предсказуемый результат
    mock_tokenizer.apply_chat_template.return_value = ["formatted_chat"] * 10

    # "Мокаем" функции из библиотеки unsloth
    mocker.patch('train.FastLanguageModel.from_pretrained', return_value=(mock_model, mock_tokenizer))
    mocker.patch('train.FastLanguageModel.get_peft_model', return_value=mock_model)

    # "Мокаем" класс SFTTrainer, чтобы он не создавался по-настоящему
    mock_sft_trainer_class = mocker.patch('train.SFTTrainer')
    # А когда его попытаются создать, вернем фейковый экземпляр
    mock_trainer_instance = MagicMock(name="MockTrainerInstance")
    mock_sft_trainer_class.return_value = mock_trainer_instance

    # "Мокаем" проверку поддержки bf16, чтобы избежать ошибок с CUDA на CPU
    mocker.patch('torch.cuda.is_bf16_supported', return_value=False)

    # Возвращаем словарь с моками для доступа к ним в тестах
    return {
        "model": mock_model,
        "tokenizer": mock_tokenizer,
        "SFTTrainer": mock_sft_trainer_class,
        "trainer_instance": mock_trainer_instance,
    }


# --- Тесты ---

def test_training_pipeline_end_to_end_logic(mock_dependencies, mock_dataset_dir):
    """
    Основной тест, который проверяет весь логический поток пайплайна:
    1. Правильность вызова загрузки модели.
    2. Корректность конфигурации SFTTrainer.
    3. Вызов методов обучения и сохранения.
    """
    TEST_OUTPUT_DIR = "test_output_dir"

    # Используем patch.dict для временной подмены глобальных констант в модуле 'train'.
    # Это гарантирует, что тест не будет зависеть от реальных путей и не создаст артефакты.
    with patch.dict(train.__dict__, {'DATASET_DIR': mock_dataset_dir, 'OUTPUT_DIR': TEST_OUTPUT_DIR}):
        # --- ACT: Запускаем основную функцию пайплайна ---
        train.run_training_pipeline()

        # --- ASSERT: Проверяем, что все было вызвано так, как мы ожидали ---

        # 1. Проверяем вызов загрузки модели
        train.FastLanguageModel.from_pretrained.assert_called_once_with(
            model_name=train.MODEL_NAME,
            max_seq_length=train.MAX_SEQ_LENGTH,
            load_in_4bit=True
        )

        # 2. Проверяем вызов PEFT-адаптации
        train.FastLanguageModel.get_peft_model.assert_called_once()

        # 3. Проверяем, что SFTTrainer был инициализирован
        SFTTrainer = mock_dependencies["SFTTrainer"]
        SFTTrainer.assert_called_once()

        # 4. Проверяем ключевые параметры конфигурации тренера
        call_args = SFTTrainer.call_args
        sft_config = call_args.kwargs['args']

        assert call_args.kwargs['max_seq_length'] == train.MAX_SEQ_LENGTH
        assert call_args.kwargs['dataset_text_field'] == "text"
        assert sft_config.learning_rate == 5e-5
        assert sft_config.per_device_train_batch_size == 4
        assert sft_config.output_dir == TEST_OUTPUT_DIR

        # Проверяем корректность разделения датасета (90/10)
        train_dataset = call_args.kwargs['train_dataset']
        eval_dataset = call_args.kwargs['eval_dataset']
        # На 10 записях разделение 90/10 дает 9 и 1
        assert len(train_dataset) == 9
        assert len(eval_dataset) == 1

        # 5. Проверяем, что были вызваны ключевые действия: обучение и сохранение
        trainer_instance = mock_dependencies["trainer_instance"]
        trainer_instance.train.assert_called_once()

        model_mock = mock_dependencies["model"]
        tokenizer_mock = mock_dependencies["tokenizer"]
        model_mock.save_pretrained.assert_called_once_with(TEST_OUTPUT_DIR)
        tokenizer_mock.save_pretrained.assert_called_once_with(TEST_OUTPUT_DIR)

