import logging.config
import os.path as path


def pylog_config(outdir: str | None = None) -> dict:
    if outdir and not path.isdir(outdir):
        raise ValueError(f"Provided `outdir` not found: {outdir}")

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "standard",
                "filename": "",
                "mode": "a",
                "encoding": "utf-8",
            },
        },
        "root": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    }
    if outdir:
        LOGGING_CONFIG["handlers"]["file"]["filename"] = path.join(outdir, "run.log")
        LOGGING_CONFIG["root"]["handlers"].append("file")
    else:
        LOGGING_CONFIG["handlers"].pop("file")

    logging.config.dictConfig(LOGGING_CONFIG)
