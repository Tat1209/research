from hydra import initialize, compose
from omegaconf import OmegaConf
import sys
import os

from exp import exp

# run.py があるディレクトリへの相対パス
# 同じ階層にあるなら "."
CONFIG_DIR = "." 
CONFIG_NAME = "config_single"

def main():
    with initialize(version_base=None, config_path=CONFIG_DIR):
        cfg = compose(config_name=CONFIG_NAME)
        # cfg = compose(config_name=CONFIG_NAME, overrides=["wd=1e-5"])
        
        print(OmegaConf.to_yaml(cfg))
        
        exp(cfg)

if __name__ == "__main__":
    main()