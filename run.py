import torch
from loguru import logger


def main():
    logger.info(torch.cuda.is_available())
    logger.info(torch.cuda.current_device())
    logger.info(torch.cuda.device_count())
    logger.info(torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
