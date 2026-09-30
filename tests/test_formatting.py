"""Экранирование пользовательского и модельного текста в Telegram HTML."""

from html.parser import HTMLParser

from app.bot.formatting import esc, llm_to_html
from app.bot.handlers.query import _build_final_response, _format_sources

_TELEGRAM_TAGS = {"b", "i", "code", "pre"}


class _TagCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)


def _tags(html_text: str) -> list[str]:
    parser = _TagCollector()
    parser.feed(html_text)
    return parser.tags


def test_esc_neutralizes_markup():
    assert esc("Отчёт <Q3> & итоги.pdf") == "Отчёт &lt;Q3&gt; &amp; итоги.pdf"
    assert esc(42) == "42"


def test_llm_to_html_keeps_telegram_tags_and_escapes_the_rest():
    out = llm_to_html("<b>Итог</b>: если a < b и c > d, <script>x</script>")
    assert out.startswith("<b>Итог</b>")
    assert "a &lt; b" in out
    assert "&lt;script&gt;" in out
    assert set(_tags(out)) <= _TELEGRAM_TAGS


def test_llm_to_html_converts_markdown():
    out = llm_to_html("**Важно**\n- пункт")
    assert "<b>Важно</b>" in out
    assert "• пункт" in out


def test_sources_with_markup_in_filename_are_escaped():
    """Раньше имя файла вставлялось в HTML как есть: «<» ломал сообщение целиком."""
    sources = [{"source_type": "user", "filename": "Договор <черновик> & правки.pdf", "page_number": 3}]
    out = _format_sources(sources)
    assert "Договор &lt;черновик&gt; &amp; правки.pdf" in out
    assert _tags(out) == []


def test_final_response_is_valid_telegram_html():
    sources = [
        {"source_type": "user", "filename": "a<b>.txt"},
        {"source_type": "law", "heading": "Закон <о чём-то>", "doc_type": "ФЗ", "pravo_nd": "1"},
    ]
    out = _build_final_response("Ответ: **да**, 2 < 3", sources, "model", law_search_failed=False)
    assert set(_tags(out)) <= _TELEGRAM_TAGS
    assert "a&lt;b&gt;.txt" in out
    assert "Закон &lt;о чём-то&gt;" in out
