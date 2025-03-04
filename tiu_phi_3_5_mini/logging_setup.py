import contextlib
import logging
import sys
from logging import Logger, StreamHandler
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

logs_dir = Path("./logs")
logs_dir.mkdir(exist_ok=True)

fmtr = logging.Formatter("%(asctime)s;%(name)s;%(levelname)s:%(message)s")
console_handler = StreamHandler(sys.stdout)
console_handler.setFormatter(fmtr)
console_handler.setLevel(logging.INFO)
file_handler = TimedRotatingFileHandler(logs_dir / "probe_train_logging.log", when="D", backupCount=14, encoding="utf-8")
file_handler.setFormatter(fmtr)
file_handler.setLevel(logging.DEBUG)

#credit to https://johnpaton.net/posts/redirect-logging/
class StdoutToLoggerRedirection:
    def __init__(self, logger_to_redirect_to: Logger):
        self.logger = logger_to_redirect_to
        self._redirector = contextlib.redirect_stdout(self)
    
    def write(self, msg):
        if msg and not msg.isspace():
            self.logger.log(logging.INFO, msg)
    
    def flush(self): pass
    
    def __enter__(self):
        self._redirector.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        # let contextlib do any exception handling here
        self._redirector.__exit__(exc_type, exc_value, traceback)




def create_logger(lgr_nm: str) -> Logger:
    new_logger = logging.getLogger(lgr_nm)
    new_logger.setLevel(logging.DEBUG)
    new_logger.addHandler(console_handler)
    new_logger.addHandler(file_handler)
    return new_logger

