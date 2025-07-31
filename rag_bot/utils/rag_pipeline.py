import logging
from utils.hf_api import call_api

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_answer(context: str, question: str) -> str:
    """
    Генерирует ответ на вопрос на основе предоставленного контекста с использованием внешней языковой модели:
    1. Проверяет входные данные.
    2. Вызывает API для генерации ответа.
    3. Обрабатывает возможные ошибки и возвращает результат.

    Args:
        context (str): Контекст.
        question (str): Вопрос пользователя.

    Returns:
        str: Сгенерированный ответ или сообщение об ошибке.
    """

    logger.info("Генерация ответа")

    if not context or not question:
        logger.warning("Пустой контекст или вопрос")
        return "Пустой контекст или вопрос"

    try:
        answer = call_api(context=context, question=question)
        return answer
    except Exception as e:
        logger.error(f"Ошибка при обращении к модели: {e}")
        return "Ошибка при обращении к модели"