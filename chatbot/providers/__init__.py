"""Provider factory: turns Settings into a concrete ChatProvider.

To add a new backend: implement ChatProvider, then register it in
``build_provider`` and add its name to config.SUPPORTED_PROVIDERS.
"""

from __future__ import annotations

from chatbot.config import Settings, SUPPORTED_PROVIDERS
from chatbot.errors import ConfigError
from chatbot.providers.base import ChatProvider
from chatbot.providers.offline import OfflineProvider
from chatbot.providers.openai_compatible import OpenAICompatibleProvider

__all__ = ["build_provider"]


def build_provider(settings: Settings) -> ChatProvider:
    """Create the provider selected by ``settings.provider``."""
    if settings.provider == "offline":
        return OfflineProvider()
    if settings.provider == "openai":
        return OpenAICompatibleProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            timeout=settings.timeout,
        )
    raise ConfigError(
        f"unknown provider {settings.provider!r}; supported: {SUPPORTED_PROVIDERS}"
    )
