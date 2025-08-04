import os
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_API_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")

base_url = "https://router.huggingface.co/v1"

client = OpenAI(
    base_url=base_url,
    api_key=HF_TOKEN
)

logger.info("Клиент OpenAI инициализирован")

def call_api(question: str, context: str) -> str:
    logger.info("Отправляем запрос к модели через OpenAI-совместимый API")
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Ты — помощник по IBM SPSS. Если вопрос касается функционала из контекста — отвечай на его основе. Если вопрос общий (например, 'Что такое SPSS?') — можешь ответить, опираясь на общие знания."
                },
                {
                    "role": "user",
                    "content": f"Контекст:\n{context}\n\nВопрос:\n{question}"
                }
            ],
            max_tokens=512,
            temperature=0.3
        )

        answer = completion.choices[0].message.content.strip()
        logger.info(f"Сгенерированный ответ: '{answer}'")
        
        if not answer:
            return "Не удалось сгенерировать ответ."
            
        return answer

    except Exception as e:
        logger.error(f"Ошибка при вызове API: {e}", exc_info=True)
        return f"Ошибка при генерации ответа: {str(e)}"