import torch
from loguru import logger
# import flash_attn_2_cuda as flash_attn_gpu


def main():
    logger.info(torch.__version__)
    logger.info(torch.cuda.is_available())
    logger.info(torch.cuda.current_device())
    logger.info(torch.cuda.device_count())
    logger.info(torch.cuda.get_device_name(0))
    # logger.info(flash_attn_gpu.__version__)


if __name__ == "__main__":
    main()
