"""Symmetric encryption for message text at rest."""

from __future__ import annotations

import logging
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from app.config import get_settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    key = get_settings().encryption_key
    if not key:
        log.warning("MC_ENCRYPTION_KEY not set; using an ephemeral key (stored messages unreadable after restart)")
        key = Fernet.generate_key().decode()
    return Fernet(key.encode())


def encrypt(text: str | None) -> str | None:
    return None if text is None else _fernet().encrypt(text.encode()).decode()


def decrypt(token: str | None) -> str | None:
    if token is None:
        return None
    try:
        return _fernet().decrypt(token.encode()).decode()
    except InvalidToken:
        return None
