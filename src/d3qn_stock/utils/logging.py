from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass
class LogPaths:
    run_dir: Path
    metrics_csv: Path
    tensorboard_dir: Path


def setup_run_logger(
    name: str,
    run_dir: Optional[Path] = None,
    log_filename: str = "run.log",
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / log_filename
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class CSVLogger:
    def __init__(self, path: Path, fieldnames: Iterable[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=list(fieldnames))
        self._writer.writeheader()

    def log(self, row: Dict[str, object]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class TensorBoardLogger:
    def __init__(self, log_dir: Path) -> None:
        self._writer = None
        self._log_dir = log_dir
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(log_dir))
        except Exception:
            self._writer = None

    def log(self, metrics: Dict[str, float], step: int) -> None:
        if not self._writer:
            return
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            for key, value in metrics.items():
                self._writer.add_scalar(key, value, step)
        except Exception:
            try:
                self._writer.close()
            finally:
                self._writer = None

    def close(self) -> None:
        if self._writer:
            self._writer.close()


class MetricsLogger:
    def __init__(self, paths: LogPaths, fieldnames: Iterable[str]) -> None:
        self.csv_logger = CSVLogger(paths.metrics_csv, fieldnames)
        self.tb_logger = TensorBoardLogger(paths.tensorboard_dir)

    def log(self, row: Dict[str, object], step: int) -> None:
        self.csv_logger.log(row)
        float_metrics = {key: float(value) for key, value in row.items() if isinstance(value, (int, float))}
        self.tb_logger.log(float_metrics, step)

    def close(self) -> None:
        self.csv_logger.close()
        self.tb_logger.close()
