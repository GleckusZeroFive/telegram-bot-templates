import logging
from collections.abc import AsyncGenerator
from typing import Any

from app.config import settings
from app.core.calibrator import get_model_profile
from app.llm.factory import get_llm_provider
from app.presets import get_preset

logger = logging.getLogger(__name__)



class ResponseGenerator:
    def __init__(self) -> None:
        self.provider = get_llm_provider()

    def _trim_history(
        self,
        conversation_history: list[dict[str, str]],
        system_chars: int,
        question_chars: int,
    ) -> list[dict[str, str]] | None:
        """Обрезать историю, если суммарный размер превышает бюджет."""
        max_chars = settings.conversation_max_context_chars
        budget = max_chars - system_chars - question_chars
        if budget <= 0:
            return None

        history_chars = sum(len(m["content"]) for m in conversation_history)
        if history_chars <= budget:
            return conversation_history

        # Убираем старые пары (по 2 сообщения: user + assistant), оставляя свежие
        trimmed: list[dict[str, str]] = []
        used = 0
        for msg in reversed(conversation_history):
            msg_len = len(msg["content"])
            if used + msg_len > budget:
                break
            trimmed.insert(0, msg)
            used += msg_len

        # История должна начинаться с user, не с assistant
        if trimmed and trimmed[0]["role"] == "assistant":
            trimmed = trimmed[1:]

        return trimmed if trimmed else None

    def _build_messages(
        self,
        question: str,
        context_chunks: list[dict],
        conversation_history: list[dict[str, str]] | None = None,
        user_state: str | None = None,
        mode: str = "rag",
    ) -> list[dict[str, Any]]:
        """Собрать messages для LLM: system + history + question."""
        preset = get_preset()
        from app.bot.commands import _get_commands_short, format_commands_for_prompt

        profile = get_model_profile()

        if mode == "followup":
            if not profile.follows_negative_instructions and preset.prompts.followup_strict:
                system_content = preset.prompts.followup_strict
            else:
                system_content = preset.prompts.followup
        elif mode == "chat":
            if conversation_history:
                system_content = preset.prompts.followup
            else:
                state = user_state or "Нет данных"
                commands_list = format_commands_for_prompt()
                system_content = preset.prompts.chat.format(
                    bot_name=preset.name,
                    user_state=state,
                    commands_list=commands_list,
                )
        elif not context_chunks:
            state = user_state or "Нет данных"
            commands_list = format_commands_for_prompt()
            system_content = preset.prompts.chat.format(
                bot_name=preset.name,
                user_state=state,
                commands_list=commands_list,
            )
        else:
            context_parts = []
            for chunk in context_chunks:
                if chunk.get("source_type") == "law":
                    parts = []
                    if chunk.get("doc_type"):
                        parts.append(chunk["doc_type"])
                    if chunk.get("heading"):
                        parts.append(chunk["heading"])
                    if chunk.get("doc_date"):
                        parts.append(f"от {chunk['doc_date']}")
                    if chunk.get("doc_number"):
                        parts.append(f"N{chunk['doc_number']}")
                    if chunk.get("status"):
                        parts.append(f"Статус: {chunk['status']}")
                    label = f"[law: {', '.join(parts)}]"
                else:
                    parts = [chunk["filename"]]
                    if chunk.get("page_number"):
                        parts.append(f"стр. {chunk['page_number']}")
                    if chunk.get("section_header"):
                        parts.append(chunk["section_header"])
                    parts.append(f"чанк {chunk['chunk_index']}")
                    label = f"[{', '.join(parts)}]"
                context_parts.append(f"{label}\n{chunk['text']}")

            context = "\n\n---\n\n".join(context_parts)
            if not profile.follows_negative_instructions and preset.prompts.system_strict:
                template = preset.prompts.system_strict
            else:
                template = preset.prompts.system
            system_content = template.format(
                context=context, commands_short=_get_commands_short(),
            )

        # Disable Qwen3 thinking mode to avoid wasting tokens on <think> blocks
        if "qwen" in (self.provider.model or "").lower():
            system_content += "\n/no_think"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
        ]

        if conversation_history:
            trimmed = self._trim_history(
                conversation_history,
                system_chars=len(system_content),
                question_chars=len(question),
            )
            if trimmed:
                messages.extend(trimmed)

        messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def extract_sources(context_chunks: list[dict]) -> list[dict]:
        """Дедупликация и извлечение источников из чанков."""
        sources = []
        seen: set[tuple] = set()
        for chunk in context_chunks:
            if chunk.get("source_type") == "law":
                key = ("law", chunk.get("pravo_nd", ""), chunk.get("chunk_index", 0))
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "source_type": "law",
                        "heading": chunk.get("heading"),
                        "doc_type": chunk.get("doc_type"),
                        "doc_date": chunk.get("doc_date"),
                        "doc_number": chunk.get("doc_number"),
                        "status": chunk.get("status"),
                        "pravo_nd": chunk.get("pravo_nd"),
                        "score": chunk.get("score", 0),
                    })
            else:
                key = ("user", chunk["filename"], chunk["chunk_index"])
                if key not in seen:
                    seen.add(key)
                    source = {
                        "source_type": "user",
                        "filename": chunk["filename"],
                        "chunk_index": chunk["chunk_index"],
                        "score": chunk["score"],
                    }
                    if chunk.get("page_number"):
                        source["page_number"] = chunk["page_number"]
                    if chunk.get("section_header"):
                        source["section_header"] = chunk["section_header"]
                    sources.append(source)
        return sources

    async def generate_hypothetical(self, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — ассистент. Напиши КОРОТКИЙ фрагмент документа (2-4 предложения), "
                    "который непосредственно отвечает на вопрос пользователя. "
                    "Пиши от третьего лица, как справочный текст. "
                    "Если не знаешь точного ответа — напиши правдоподобный фрагмент."
                    "\n/no_think"
                ),
            },
            {"role": "user", "content": question},
        ]
        return await self.provider.generate(messages, max_tokens=200, use_cache=True)

    async def rewrite_query(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Перефразировать запрос для улучшения поиска.

        Если есть история диалога — раскрывает эллиптические ссылки.
        Возвращает самодостаточный поисковый запрос на русском языке.
        """
        history_text = ""
        if conversation_history:
            recent = conversation_history[-4:]
            pairs = []
            for msg in recent:
                role = "Пользователь" if msg["role"] == "user" else "Ассистент"
                pairs.append(f"{role}: {msg['content'][:300]}")
            history_text = "\n".join(pairs)

        if history_text:
            prompt = (
                f"История диалога:\n{history_text}\n\n"
                f"Новый вопрос пользователя: {question}\n\n"
                "Перепиши вопрос как самодостаточный поисковый запрос (без местоимений, "
                "с полным контекстом). Верни ТОЛЬКО переформулированный запрос, без пояснений."
            )
        else:
            prompt = (
                f"Вопрос: {question}\n\n"
                "Перепиши как поисковый запрос для семантического поиска по документам. "
                "Верни ТОЛЬКО запрос, без пояснений."
            )

        messages = [
            {"role": "system", "content": "Ты — система переформулировки запросов.\n/no_think"},
            {"role": "user", "content": prompt},
        ]
        rewritten = await self.provider.generate(messages, max_tokens=100, use_cache=True)
        return rewritten.strip() or question

    async def generate_stream(
        self,
        question: str,
        context_chunks: list[dict],
        conversation_history: list[dict[str, str]] | None = None,
        user_state: str | None = None,
        mode: str = "rag",
    ) -> AsyncGenerator[str, None]:
        """
        Стриминг ответа. Yields текстовые дельты.

        Sources не зависят от LLM — вычисляются отдельно через extract_sources().
        """
        messages = self._build_messages(
            question, context_chunks, conversation_history,
            user_state=user_state, mode=mode,
        )
        async for delta in self.provider.generate_stream(messages):
            yield delta
