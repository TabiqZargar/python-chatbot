"""Public package API for the Python chatbot."""

from chatbot.chatbot import Chatbot
from chatbot.config import Settings, load_settings
from chatbot.errors import ChatbotError, ConfigError, InvalidInputError, ProviderError
from chatbot.history import ConversationHistory, Message
from chatbot.providers import build_provider

__version__ = "0.1.0"

__all__ = [
    "Chatbot",
    "ChatbotError",
    "ConfigError",
    "ConversationHistory",
    "InvalidInputError",
    "Message",
    "ProviderError",
    "Settings",
    "build_provider",
    "load_settings",
    "__version__",
]
