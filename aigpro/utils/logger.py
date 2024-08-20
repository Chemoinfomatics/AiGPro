import logging
from rich.logging import RichHandler


def get_logger(level="FATAL") -> logging.Logger:
    """Returns logger with RichHandler.

    Args:
        name (str): Name of the logger
        level (int): Level of the logger

    Returns:
        logging.Logger: The logger
    """
    # Define log format
    FORMAT: str = "%(message)s"

    # Configure logging
    logging.basicConfig(
        level=level,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler()],
    )

    # Get logger
    log: logging.Logger = logging.getLogger("rich")
    return log
