from app.config import settings
from app.llm.provider import FallbackProvider, OpenAICompatibleProvider, RoundRobinKeyManager

_provider: OpenAICompatibleProvider | FallbackProvider | None = None


def get_llm_provider() -> OpenAICompatibleProvider | FallbackProvider:
    """Фабрика LLM-провайдера (singleton).

    Основной провайдер — любой OpenAI-совместимый API (LLM_BASE_URL + LLM_MODEL).
    Если задан LLM_FALLBACK_MODEL, поверх него строится FallbackProvider.
    """
    global _provider
    if _provider is not None:
        return _provider

    if not settings.llm_model:
        raise RuntimeError(
            "LLM_MODEL не задан. Укажите модель и LLM_BASE_URL в .env (см. .env.example)."
        )

    key_manager = None
    keys = [k.strip() for k in settings.llm_api_keys.split(",") if k.strip()]
    if len(keys) > 1:
        key_manager = RoundRobinKeyManager(keys)

    primary_key = settings.llm_api_key or (keys[0] if keys else "")
    primary = OpenAICompatibleProvider(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=primary_key,
        key_manager=key_manager,
        extra_body=settings.llm_extra_body,
    )

    if not settings.llm_fallback_model:
        _provider = primary
        return _provider

    fallback = OpenAICompatibleProvider(
        base_url=settings.llm_fallback_base_url or settings.llm_base_url,
        model=settings.llm_fallback_model,
        api_key=settings.llm_fallback_api_key or primary_key,
        extra_body=settings.llm_fallback_extra_body or settings.llm_extra_body,
    )
    _provider = FallbackProvider(primary, fallback)
    return _provider
