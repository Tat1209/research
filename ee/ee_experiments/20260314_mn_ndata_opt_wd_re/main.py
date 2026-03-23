import sys
from pathlib import Path

this_path = Path(__file__) if "__file__" in globals() else Path("<undefined>.ipynb").resolve()
work_path = next((p for p in this_path.parents if p.name == "research"), None)
tools_path = work_path / Path("../torch-tools")
sys.path.append(str(tools_path))

ee_tools_path_p = work_path / Path("ee")
sys.path.append(str(ee_tools_path_p))

from gpu_scheduler import generate_tasks_grid, parallel_run
from pl_utils import filter_finished_tasks
from train import run


def main():
    this_path = Path(__file__).resolve()
    cfg = {}

    # cfg["exp_name"] = "exp_tmp"
    cfg["exp_name"] = "exp_opt_re"
    cfg["model_str"] = "resnet18"

    cfg["train_ds_str"] = "cifar100_train"
    cfg["val_ds_str"] = "cifar100_test"
    cfg["batch_size"] = 128

    cfg[("optim_str", "max_lr")] = list(zip(["sgd", "adamw"], [0.1, 5e-3]))

    cfg["wd"] = [1e-2, 5e-2, 1e-1, 3e-1, 1, 3, 10, 20, 30]
    cfg["ipc"] = [100, 50, 20, 10, 5]

    cfg["div"] = [1, 2, 4, 8, 16, 32]

    cfg["flex_ch"] = True

    cfg["num_threads"] = 4
    cfg["num_interop_threads"] = 4
    cfg["num_workers"] = 2
    cfg["compile"] = True

    tasks = generate_tasks_grid(run, cfg)
    parallel_run(tasks, avoid_used=True, gpu_ids=[1, 2, 3, 4, 5, 6, 7])


if __name__ == "__main__":
    main()
