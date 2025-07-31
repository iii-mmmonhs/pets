import fitz
import re
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Извлекает весь текст из PDF-файла:
    открывает PDF по указанному пути и последовательно извлекает текст со всех страниц.

    Args:
        pdf_path (str): Путь к PDF-файлу на диске.

    Returns:
        str: Полный извлечённый текст.
    """
    logger.info(f"Чтение PDF: {pdf_path}")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Файл не найден: {pdf_path}")

    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        logger.error(f"Ошибка при чтении файла: {e}")
        raise

    logger.info(f"Извлечено символов: {len(text)}")
    return text


def clean_text(text: str) -> str:
    """
    Очищает текст от избыточных пробелов, табуляций и переносов строк, заменяет на пробел

    Args:
        text (str): Входной текст.

    Returns:
        str: Очищенный текст.
    """
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()


def split_text_into_chunks(text: str, chunk_size: int = 512, overlap: int = 32) -> list:
    """
    Разделяет текст на чанки заданного размера с перекрытием.

    Чанки создаются по словам, чтобы избежать разрыва слов, 
    перекрытие нужно чтобы сохранить контекст на границах чанков.

    Args:
        text (str): Текст для разбиения.
        chunk_size (int): Количество слов в одном чанке.
        overlap (int): Количество слов, дублируемых в следующем чанке.

    Returns:
        list: Список строк-чанков.
    """
    logger.info(f"Разбивка текста на чанки (размер={chunk_size}, перекрытие={overlap})")

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        end = i + chunk_size + overlap
        chunk = " ".join(words[i:end])
        chunks.append(chunk)

    logger.info(f"Получено {len(chunks)} чанков")
    return chunks