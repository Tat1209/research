from pathlib import Path
from train import exp
from pl_utils import filter_finished_tasks
from gpu_scheduler import parallel_run, generate_tasks_grid

def main():
    this_path = Path(__file__).resolve()
    cfg = {}

    # cfg["exp_name"] = "exp_tmp"
    cfg["exp_name"] = "exp_model_bl_best"
    cfg["train_ds_str"] = "cifar100_train"
    cfg["val_ds_str"] = "cifar100_val"
    cfg["optim_str"] = "SGD"
    # cfg["optim_str"] = "adamw"

    model_str = ["resnet18"]
    batch_size = [128]
    max_lr = [1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1, 3e-1, 5e-1]

    # max_lr = [1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3]
    # max_lr = [base_lr * batch_size / 128 for batch_size in batch_size]
    # cfg[("model_str", "batch_size", "max_lr")] = list(zip(model_str, batch_size, max_lr))
    cfg["max_lr"] = max_lr
    cfg[("model_str", "batch_size")] = list(zip(model_str, batch_size))

    # cfg["wd"] = [1e-1, 3e-1, 5e-1, 7e-1, 1, 3, 5, 7, 10, 12, 15, 18, 20, 25, 30]
    cfg["wd"] = [1e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2, 7e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1]

    ndata = [1000]

    epochs = [200 if n >= 10000 else 10000 * 200 // n for n in ndata]
    cfg[("ndata", "epochs")] = list(zip(ndata, epochs))

    cfg["div"] = [1]

    cfg["cifar_style"] = True
    cfg["flex_ch"] = False
    
    cfg["num_threads"] = 4
    cfg["num_interop_threads"] = 4
    cfg["num_workers"] = 2
    cfg["compile"] = True
    
    tasks = generate_tasks_grid(exp, cfg)

    key_map = {
        "model_str": "model_arc",    # run内で net.__name__ としてログ保存
        "ndata": "train_ndata",      # run内で len(train_ds) としてログ保存
        "wd": "wd",                # そのままログ保存
        "epochs": "epochs",          # そのままログ保存（ndataと連動するが念のため）
        "optim_str": "optim_str",
        "batch_size": "batch_size"   # そのままログ保存（model_strと連動するが念のため）
    }
    tasks = filter_finished_tasks(tasks, parquet_path=f"{this_path.parent}/{cfg['exp_name']}/_results.parquet", key_map=key_map)

    parallel_run(tasks, avoid_used=True, gpu_ids=[0,1,2,3,4,5,6,7])

if __name__ == "__main__":
    main()