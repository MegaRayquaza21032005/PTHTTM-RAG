"""Centralized logging configuration for the chatbot application."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock


class ColorLevelFormatter(logging.Formatter):
    """Colorize the level name for console logs."""

    RESET = "\033[0m"
    LEVEL_COLORS = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[31;1m",  # bold red
    }

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        color = self.LEVEL_COLORS.get(record.levelno)

        if color:
            record.levelname = f"{color}{original_levelname:<8}{self.RESET}"

        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


class LoggerManager:
    """Configure and provide application loggers."""

    _handler_marker = "_chatbot_managed_handler"

    LOGGER_NAME = "chatbot"
    DEFAULT_FORMAT = (
        "%(asctime)s | %(levelname)-8s | %(name)s | "
        "%(filename)s:%(lineno)d | %(message)s"
    )
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    DEFAULT_MAX_BYTES = 10 * 1024 * 1024
    DEFAULT_BACKUP_COUNT = 5

    def __init__(self) -> None:
        self._configured = False

    def configure(
        self,
        *,
        level: str | int | None = None,
        log_file: str | Path | None = None,
        enable_file_logging: bool = True,
        max_bytes: int | None = None,
        backup_count: int | None = None,
    ) -> None:
        """Configure console and optional rotating-file handlers.

        Repeated calls replace only handlers created by this manager, which keeps
        configuration deterministic during FastAPI reloads and test runs.
        """
        log_level = self._resolve_level(level or os.getenv("LOG_LEVEL", "INFO"))
        file_formatter = logging.Formatter(
            self.DEFAULT_FORMAT,
            datefmt=self.DEFAULT_DATE_FORMAT,
        )
        console_formatter = ColorLevelFormatter(
            self.DEFAULT_FORMAT,
            datefmt=self.DEFAULT_DATE_FORMAT,
        )

        logger = logging.getLogger(self.LOGGER_NAME)
        logger.setLevel(log_level)
        logger.propagate = False
        self._remove_managed_handlers(logger)

        console_handler = logging.StreamHandler(sys.stdout)
        self._prepare_handler(console_handler, log_level, console_formatter)
        logger.addHandler(console_handler)

        if enable_file_logging:
            file_path = self._resolve_log_file(log_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=max_bytes or self.DEFAULT_MAX_BYTES,
                backupCount=(
                    self.DEFAULT_BACKUP_COUNT
                    if backup_count is None
                    else backup_count
                ),
                encoding="utf-8",
            )
            self._prepare_handler(file_handler, log_level, file_formatter)
            logger.addHandler(file_handler)

        self._configured = True

    def get_logger(self, name: str | None = None) -> logging.Logger:
        """Return a logger under the common ``chatbot`` namespace."""
        if not self._configured:
            self.configure()

        if not name or name == self.LOGGER_NAME:
            return logging.getLogger(self.LOGGER_NAME)

        normalized_name = name.removeprefix(f"{self.LOGGER_NAME}.")
        return logging.getLogger(f"{self.LOGGER_NAME}.{normalized_name}")

    @classmethod
    def _prepare_handler(
        cls,
        handler: logging.Handler,
        level: int,
        formatter: logging.Formatter,
    ) -> None:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        setattr(handler, cls._handler_marker, True)

    @classmethod
    def _remove_managed_handlers(cls, logger: logging.Logger) -> None:
        for handler in logger.handlers[:]:
            if getattr(handler, cls._handler_marker, False):
                logger.removeHandler(handler)
                handler.close()

    @staticmethod
    def _resolve_level(level: str | int) -> int:
        if isinstance(level, int):
            return level

        resolved_level = logging.getLevelName(level.upper())
        if not isinstance(resolved_level, int):
            raise ValueError(f"Invalid log level: {level}")
        return resolved_level

    @staticmethod
    def _resolve_log_file(log_file: str | Path | None) -> Path:
        configured_path = log_file or os.getenv("LOG_FILE")
        if configured_path:
            return Path(configured_path).expanduser()

        project_root = Path(__file__).resolve().parents[2]
        return project_root / "logs" / "chatbot.log"


_logger_manager: LoggerManager | None = None
_logger_manager_lock = Lock()


def get_logger_manager() -> LoggerManager:
    """Return the shared logger manager instance."""
    global _logger_manager

    if _logger_manager is None:
        with _logger_manager_lock:
            if _logger_manager is None:
                _logger_manager = LoggerManager()

    return _logger_manager


def get_logger(name: str | None = None) -> logging.Logger:
    """Return an application logger from the shared manager."""
    return get_logger_manager().get_logger(name)
