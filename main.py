import logging

from equity_project.src.get_data import get_data
from equity_project.src.run_backtest import run_backtest
from equity_project.src.train import train


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Pipeline started")
    logger.info("Step 1/3: data preparation started")
    get_data()
    logger.info("Step 1/3: data preparation finished")

    logger.info("Step 2/3: model training started")
    train()
    logger.info("Step 2/3: model training finished")

    logger.info("Step 3/3: backtest started")
    run_backtest()
    logger.info("Step 3/3: backtest finished")
    logger.info("Pipeline finished")


if __name__ == "__main__":
    main()
