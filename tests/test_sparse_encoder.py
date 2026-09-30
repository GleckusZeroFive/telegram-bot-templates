"""Тесты для sparse encoder (BM25 keyword search)."""

from app.core.sparse_encoder import (
    VOCAB_SIZE,
    _token_to_index,
    encode_sparse,
    encode_sparse_query,
    tokenize,
)

# === tokenize ===

def test_tokenize_basic_russian():
    """Русские словоформы приводятся к лемме (pymorphy3 — runtime-зависимость)."""
    tokens = tokenize("Машинное обучение работает отлично")
    assert tokens == ["машинный", "обучение", "работать", "отлично"]


def test_tokenize_lemmatizes_word_forms_to_same_token():
    """Разные падежи одного слова дают один токен — BM25 находит «комиссию» по «комиссии»."""
    assert tokenize("комиссию") == tokenize("комиссии") == tokenize("комиссия")


def test_tokenize_removes_russian_stop_words():
    tokens = tokenize("это было хорошо для всех")
    # "это", "было", "хорошо", "для", "всех" — стоп-слова
    assert len(tokens) == 0


def test_tokenize_removes_english_stop_words():
    tokens = tokenize("the quick brown fox is a fast animal")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "quick" in tokens
    assert "brown" in tokens
    assert "animal" in tokens


def test_tokenize_removes_short_tokens():
    tokens = tokenize("a b cd ef ghi")
    assert "a" not in tokens
    assert "b" not in tokens
    assert "cd" in tokens
    assert "ef" in tokens
    assert "ghi" in tokens


def test_tokenize_lowercases():
    tokens = tokenize("Python JavaScript RUST")
    assert "python" in tokens
    assert "javascript" in tokens
    assert "rust" in tokens


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("   ") == []


def test_tokenize_unicode_numbers():
    tokens = tokenize("версия 3.12 и Python312")
    assert "версия" in tokens
    assert "12" in tokens or "python312" in tokens


# === _token_to_index ===

def test_token_to_index_deterministic():
    idx1 = _token_to_index("example")
    idx2 = _token_to_index("example")
    assert idx1 == idx2


def test_token_to_index_in_range():
    for word in ["cat", "собака", "Python", "документ", "12345"]:
        idx = _token_to_index(word)
        assert 0 <= idx < VOCAB_SIZE


def test_token_to_index_different_tokens():
    idx1 = _token_to_index("cat")
    idx2 = _token_to_index("dog")
    assert 0 <= idx1 < VOCAB_SIZE
    assert 0 <= idx2 < VOCAB_SIZE
    # Разные слова — скорее всего разные индексы (коллизии крайне редки)


def test_token_to_index_cyrillic_deterministic():
    idx1 = _token_to_index("документ")
    idx2 = _token_to_index("документ")
    assert idx1 == idx2


# === encode_sparse ===

def test_encode_sparse_basic():
    sv = encode_sparse("hello world hello")
    assert len(sv.indices) > 0
    assert len(sv.indices) == len(sv.values)


def test_encode_sparse_tf_counts():
    """Повторяющийся токен должен иметь TF > 1."""
    sv = encode_sparse("python python python java")
    # "python" встречается 3 раза
    assert max(sv.values) >= 3.0


def test_encode_sparse_empty():
    sv = encode_sparse("")
    # Sentinel для пустого текста
    assert len(sv.indices) == 1
    assert sv.values[0] == 0.0


def test_encode_sparse_only_stop_words():
    sv = encode_sparse("и в на по для")
    assert len(sv.indices) == 1
    assert sv.values[0] == 0.0


def test_encode_sparse_indices_sorted():
    sv = encode_sparse("некоторый случайный текст несколькими словами здесь проверка")
    assert sv.indices == sorted(sv.indices)


# === encode_sparse_query ===

def test_encode_sparse_query_binary_weights():
    """Query encoder даёт 1.0 на каждый уникальный терм."""
    sv = encode_sparse_query("hello world hello world")
    assert all(v == 1.0 for v in sv.values)


def test_encode_sparse_query_deduplicates():
    """Повторы в запросе не увеличивают количество индексов."""
    sv = encode_sparse_query("python python python")
    # Только один уникальный токен
    assert len(sv.indices) == 1
    assert sv.values[0] == 1.0


def test_encode_sparse_query_empty():
    sv = encode_sparse_query("")
    assert len(sv.indices) == 1
    assert sv.values[0] == 0.0


def test_encode_sparse_query_indices_sorted():
    sv = encode_sparse_query("машинное обучение нейронные сети трансформеры")
    assert sv.indices == sorted(sv.indices)


# === Совместимость document + query ===

def test_document_and_query_share_indices():
    """Одинаковые токены в документе и запросе → одинаковые индексы."""
    doc_sv = encode_sparse("машинное обучение глубокие нейронные сети")
    query_sv = encode_sparse_query("машинное обучение")

    doc_indices = set(doc_sv.indices)
    query_indices = set(query_sv.indices)
    # Все индексы запроса должны присутствовать в документе
    assert query_indices.issubset(doc_indices)
