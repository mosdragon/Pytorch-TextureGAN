import logging


# Pass in as many arguments as you want to these functions,
# they will simply concat them together.
def debug(*msg):
    logger.debug(''.join([str(m) for m in msg]))

def info(*msg):
    logger.info(''.join([str(m) for m in msg]))

def warn(*msg):
    logger.warning(''.join([str(m) for m in msg]))

def error(*msg):
    logger.error(''.join([str(m) for m in msg]))


log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "error": logging.ERROR,
    "warn": logging.WARN,
}

def add_log_file(filepath):
    """
    Add a file logger to the application logger. Appends to file if it already
    exists.
    """
    handler = logging.FileHandler(filepath, mode='a')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)  # Add the Stream Handler to the logger.
    debug("Logging all output to file {}".format(filepath))


logger = logging.getLogger("default")
# Create formatter and add it to a Stream Handler.
formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s",
                                datefmt='%m/%d/%Y %I:%M:%S %p')

handler = logging.StreamHandler()
handler.setFormatter(formatter)
# Add the Stream Handler to the logger.
logger.addHandler(handler)

# Set log level.
handler.setLevel(log_levels["debug"])
logger.setLevel(log_levels["debug"])

add_log_file("logs/textureGAN.log")