import hydra
from omegaconf import DictConfig

from apple_leaf_disease.train import train


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == '__main__':
    main()
