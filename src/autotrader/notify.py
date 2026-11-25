"""알림: 텔레그램(선택) + 로그."""
from __future__ import annotations

import logging

import requests

from autotrader.config import env

log = logging.getLogger(__name__)


class Notifier:
    def __init__(self, telegram: bool = False):
        self.token = env("TELEGRAM_BOT_TOKEN") if telegram else ""
        self.chat_id = env("TELEGRAM_CHAT_ID") if telegram else ""

    def send(self, text: str) -> None:
        log.info("[알림] %s", text)
        if not (self.token and self.chat_id):
            return
        try:
            requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                          json={"chat_id": self.chat_id, "text": text[:4000]}, timeout=10)
        except requests.RequestException as e:  # 알림 실패가 매매를 막지 않게
            log.warning("텔레그램 전송 실패: %s", e)
