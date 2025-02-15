import logging

def setup_logging():
    logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] [%(thread)d] %(asctime)s-%(filename)s(line: %(lineno)d) : %(message)s',
    datefmt='%Y-%m-%d %H:%M'
)

setup_logging()
