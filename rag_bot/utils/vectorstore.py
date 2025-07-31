import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SAVE_PATH = "./data/embeddings/index.faiss"

def build_vectorstore(model: SentenceTransformer, chunks: list, save_path=DEFAULT_SAVE_PATH):
    """
    Создаёт или загружает векторное хранилище на основе текстовых чанков:
    - Проверяет, существует ли уже сохранённый индекс.
    - Если да — загружает его.
    - Если нет — кодирует чанки в эмбеддинги, строит FAISS-индекс и сохраняет его на диск.

    Args:
        model (SentenceTransformer): Модель для кодирования текста в вектора.
        chunks (list of str): Список чанков, которые нужно добавить в хранилище.
        save_path (str): Путь для сохранения/загрузки FAISS-индекса.

    Returns:
        tuple:
            - index (faiss.Index): Векторный индекс FAISS.
            - chunks (list of str): Исходные текстовые чанки.
            - model (SentenceTransformer): Переданная модель.
    """

    logger.info("Создание векторного хранилища")

    if os.path.exists(save_path):
        logger.warning(f"Индекс уже существует: {save_path}. Загружаем его.")
        return load_vectorstore(model, save_path)

    logger.info("Кодировка чанков")
    try:
        embeddings = model.encode(chunks, convert_to_numpy=True)
    except Exception as e:
        logger.error(f"Ошибка кодирования: {e}")
        raise

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    logger.info(f"Сохраняем индекс в {save_path}")
    faiss.write_index(index, save_path)

    chunks_path = save_path + ".pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks, model


def load_vectorstore(model: SentenceTransformer, load_path=DEFAULT_SAVE_PATH):
    """
    Загружает сохранённый векторный индекс и соответствующие чанки.

    Args:
        model (SentenceTransformer): Модель, использованная для кодирования.
        load_path (str): Путь к сохранённому FAISS-индексу.

    Returns:
        tuple:
            - index (faiss.Index): Загруженный векторный индекс.
            - chunks (list of str): Сохранённые чанки.
            - model (SentenceTransformer): Переданная модель.
    """
    logger.info(f"Загружаем индекс из {load_path}")

    index = faiss.read_index(load_path)

    chunks_path = load_path + ".pkl"
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks, model


def retrieve_relevant_chunks(query, index, chunks, model, top_k=3):
    """
    Ищет наиболее релевантные чанки для заданного запроса:
    - Кодирует запрос в вектор.
    - Использует FAISS для поиска ближайших соседей.
    - Возвращает чанки, соответствующие найденным индексам.

    Args:
        query (str): Вопрос пользователя.
        index (faiss.Index): Векторный индекс, в котором ищем.
        chunks (list of str): Список исходных текстовых чанков.
        model (SentenceTransformer): Модель для кодирования запроса.
        top_k (int): Количество возвращаемых релевантных чанков.

    Returns:
        list of str: Список из `top_k` наиболее релевантных текстовых фрагментов.
    """
    logger.info("Поиск релевантных чанков")

    if not query or not isinstance(query, str):
        raise ValueError("Запрос должен быть непустой строкой")

    q_emb = model.encode([query])
    distances, indices = index.search(np.array(q_emb), top_k)
    results = [chunks[i] for i in indices[0]]
    return results