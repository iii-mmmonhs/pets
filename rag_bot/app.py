import gradio as gr
from time import sleep
from typing import List
import logging

from config import PDF_PATH, MODEL_NAME
from utils.pdf_parser import extract_text_from_pdf, split_text_into_chunks
from utils.vectorstore import build_vectorstore, load_vectorstore, retrieve_relevant_chunks
from utils.rag_pipeline import generate_answer

from sentence_transformers import SentenceTransformer

import os
os.makedirs("./data/embeddings", exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Количество  релевантных чанков в качестве контекста
CONTEXT_CHUNKS_NUM = 5

class RAGBot:
    """
    Класс RAGBot:    
    - Загрузка модели эмбеддингов
    - Парсинг и хранение текста из PDF
    - Построение векторного хранилища
    - Поиск релевантных фрагментов
    - Генерация ответов на вопросы
    
    """
    def __init__(self):
        self.model = None        # Модель для эмбеддингов
        self.chunks = []         # Список текстовых чанков из PDF
        self.index = None        # Векторный индекс
        self.is_initialized = False  # Флаг инициализации

    def setup(self):
        logger.info("Инициализация")
        self.model = SentenceTransformer(MODEL_NAME)
        raw_text = extract_text_from_pdf(PDF_PATH)
        self.chunks = split_text_into_chunks(raw_text)
    
        index, _, _ = build_vectorstore(self.model, self.chunks)
        
        self.index = index
        self.is_initialized = True

    def answer_question(self, question: str) -> str:
        """
        Получение запроса и возвращение ответа:
    
        1. Поиск релевантных чанков через векторный поиск
        2. Формирование контекста
        3. Генерация ответа с помощью LLM
        
        Args:
            question (str): Вопрос от пользователя
            
        Returns:
            str: Ответ на вопрос или сообщение об ошибке
        """
        logger.info(f"Получен вопрос: {question}")
        try:
            if self.model is None:
                logger.info("Модель не загружена, начинаем загрузку...")
                self.load_resources()
            
            logger.info("Получение релевантных чанков")
            relevant_chunks = retrieve_relevant_chunks(question, self.index, self.chunks, self.model)
            
            if not relevant_chunks:
                logger.warning("Релевантные чанки не найдены")
                return "Контекст не найден"
    
            context = "\n\n".join(relevant_chunks[:CONTEXT_CHUNKS_NUM])
            logger.info(f"Генерация ответа на основе контекста:\n{context[:200]}...")
    
            answer = generate_answer(context, question)
            return answer
    
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
            return f"Произошла ошибка: {str(e)}"


bot = RAGBot()

def respond(message, chat_history):
    """
    Функция для Gradio ChatInterface: показывает промежуточное сообщение "Ищу ответ", затем возвращает сгенерированный ответ.
    
    Args:
        message (str): Сообщение от пользователя
        chat_history (List): История чата (пока не используется)
        
    Yields:
        str: Промежуточное сообщение и финальный ответ
    """
    yield "Ищу ответ"
    sleep(1)
    result = bot.answer_question(message)
    yield f"{result}"


# Интерфейс
with gr.Blocks() as demo:
    gr.ChatInterface(
        respond,
        additional_inputs=[],
        chatbot=gr.Chatbot(label="RAG-бот по IBM SPSS", bubble_full_width=False),
        textbox=gr.Textbox(placeholder="Введите вопрос по IBM SPSS...", label="Ваш вопрос"),
        title="RAG-бот по IBM SPSS",
        description="Задайте вопрос — я найду ответ в официальной документации.",
        examples=[
            "Как запустить SPSS?",
            "Как создать таблицу в SPSS?"
        ]
    )

# Запуск приложения
if __name__ == "__main__":
    logger.info("Gradio готов к запуску")
    bot.setup()
    demo.launch(ssr_mode=False, share=True)