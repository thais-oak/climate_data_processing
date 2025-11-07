# logger.py

import sys
import logging

import pytz
import datetime as dt

# configurações de log
class TZFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.tz = tz or pytz.UTC

    def formatTime(self, record, datefmt=None):
        date_time = dt.datetime.fromtimestamp(record.created, self.tz)
        if datefmt:
            s = date_time.strftime(datefmt)
        else:
            s = date_time.isoformat()
        return s

# função de configuração do log
def configurar_logger(nome_logger, selected_tz):
    logger = logging.getLogger(nome_logger)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # evita envio ao root logger

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        formatter = TZFormatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S %z',
            tz=selected_tz
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger