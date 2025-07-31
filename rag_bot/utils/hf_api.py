import os
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_API_TOKEN")
RAG_API_URL = os.getenv("RAG_API_URL")

# Заголовки для запроса к Hugging Face Inference API
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

logger.info("Вызов API")
def call_api(question: str, context: str) -> str:
    """
    Отправляет запрос к языковой модели через Hugging Face Inference API для генерации ответа.

    Args:
        question (str): Вопрос от пользователя
        context (str): Контекст из документации, найденный ретривером

    Returns:
        str: Сгенерированный ответ или сообщение об ошибке

    NB: Ожидается, что модель поддерживает chat-формат (например, Qwen, Llama-2-chat и т.п.)
    """

    prompt = f"""<|system|>
Ты — помощник по IBM SPSS. Старайся отвечать на основе контекста.

Если ответ найден в контексте — дай точный ответ.
Если точной информации нет, ответь на основе общих знаний "

<|user|>
Контекст:
{context}

Вопрос:
{question}"""

    # Тело запроса
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.3,
            "do_sample": True,
            "return_full_text": False
        }
    }

    logger.info("Отправляем запрос к модели...")

    try:
        response = requests.post(RAG_API_URL, headers=headers, json=payload)
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при обращении к API: {e}")
        return "Не удалось получить ответ от модели."

    try:
        generated_text = response.json()[0]["generated_text"].strip()
    except (KeyError, IndexError):
        logger.warning("Неверный формат ответа от модели")
        return "Ошибка получения ответа."

    if not generated_text or generated_text == "Информация не найдена.":
        return "Информация не найдена."

    return generated_text