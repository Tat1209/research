import sys
import math
from pathlib import Path

import torch

from spectral_metric.estimator import CumulativeGradientEstimator
from spectral_metric.visualize import make_graph

work_path = Path(next((p for p in Path(__file__).resolve().parents if p.name == "Research"), None))
torchlib_path = str(work_path / Path("app/torch_libs"))
sys.path.append(torchlib_path)

from datasets import Datasets, dl
from run_manager import RunManager, RunsManager, RunViewer
from trainer import Model, MyMultiTrain
from trans import Trans
import utils

ds = Datasets(root=work_path / "assets/datasets/")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

X, y = dl(ds("cifar100_train"))
estimator = CumulativeGradientEstimator(M_sample=250, k_nearest=5)
estimator.fit(data=X, target=y)
csg = estimator.csg  # The actual complexity values.
estimator.evals, estimator.evecs  # The eigenvalues and vectors.

# You can plot the dataset with:
make_graph(estimator.difference, title="Your dataset", classes=["A", "B", "C"])

