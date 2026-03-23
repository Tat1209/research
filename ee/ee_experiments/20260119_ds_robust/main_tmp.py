from pathlib import Path

from gpu_scheduler import generate_tasks_grid, parallel_run
from pl_utils import filter_finished_tasks
from train import run


def main():
    this_path = Path(__file__).resolve()
    cfg = {}

    cfg["exp_name"] = "exp_robust"

    cfg["model_str"] = "resnet18"

    train_ds_str = ["cub200_train"]
    val_ds_str = ["cub200_val"]
    batch_size = [32]
    max_lr = [5e-3 * batch_size / 128 for batch_size in batch_size]

    cfg[("train_ds_str", "val_ds_str", "batch_size", "max_lr")] = list(
        zip(train_ds_str, val_ds_str, batch_size, max_lr)
    )

    cfg["optim_str"] = "adamw"

    cfg["wd"] = 5e-2
    cfg["ipc"] = ["all"]

    # ndata = [50000, 20000, 10000, 5000, 2000, 1000, 500, 100]
    # epochs = [200 if n >= 10000 else 10000 * 200 // n for n in (ipc * 100)]
    # cfg[("ipc", "epochs")] = list(zip(ipc, epochs))

    cfg["div"] = [16, 32]

    cfg["flex_ch"] = True

    cfg["num_threads"] = 4
    cfg["num_interop_threads"] = 4
    cfg["num_workers"] = 2
    cfg["compile"] = False

    tasks = generate_tasks_grid(run, cfg)
    key_map = {
        "model_str": "model_arc",  # run内で net.__name__ としてログ保存
        # "ndata": "train_ndata",      # run内で len(train_ds) としてログ保存
        "div": "div",  # そのままログ保存
        # "epochs": "epochs",          # そのままログ保存（ndataと連動するが念のため）
        "batch_size": "batch_size",  # そのままログ保存（model_strと連動するが念のため）
    }
    tasks = filter_finished_tasks(
        tasks,
        parquet_path=f"{this_path.parent}/{cfg['exp_name']}/_results.parquet",
        key_map=key_map,
    )
    # print(len(tasks))
    parallel_run(tasks, avoid_used=True)
    # parallel_run(tasks, avoid_used=True, gpu_ids=[0,1,2,3,4,5,6,7])


if __name__ == "__main__":
    main()
