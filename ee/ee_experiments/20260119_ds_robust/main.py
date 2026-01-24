from pathlib import Path
from train import run
from pl_utils import filter_finished_tasks
from gpu_scheduler import parallel_run, generate_tasks_grid

def main():
    this_path = Path(__file__).resolve()
    cfg = {}

    cfg["exp_name"] = "exp_robust"
    
    cfg["model_str"] = "resnet18"
    
    train_ds_str = ["stl10_train", "tiny-imagenet_train", "caltech101_trainval", "cub200_train", "flowers102_train", "oxford-pet_trainval", "mnist_train", "fashion-mnist_train", "svhn_train", "cifar100_train", "cifar10_train"]
    val_ds_str = ["stl10_val", "tiny-imagenet_val", "caltech101_trainval", "cub200_val", "flowers102_val", "oxford-pet_val", "mnist_val", "fashion-mnist_val", "svhn_val", "cifar100_val", "cifar10_val"]
    batch_size = [128, 128, 32, 32, 32, 32, 128, 128, 128, 128, 128]
    max_lr = [5e-3 * batch_size / 128 for batch_size in batch_size]
    
    cfg[("train_ds_str", "val_ds_str", "batch_size", "max_lr")] = list(zip(train_ds_str, val_ds_str, batch_size, max_lr))

    cfg["optim_str"] = "adamw"

    cfg["wd"] = 5e-2
    cfg["ipc"] = ["all", "max", 5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5, 1]

    # ndata = [50000, 20000, 10000, 5000, 2000, 1000, 500, 100]
    # epochs = [200 if n >= 10000 else 10000 * 200 // n for n in (ipc * 100)]
    # cfg[("ipc", "epochs")] = list(zip(ipc, epochs))

    cfg["div"] = [1, 2, 4, 8, 16, 32]

    cfg["flex_ch"] = True
    
    cfg["num_threads"] = 4
    cfg["num_interop_threads"] = 4
    cfg["num_workers"] = 2
    cfg["compile"] = True
    

    tasks = generate_tasks_grid(run, cfg)
    key_map = {
        "model_str": "model_arc",
        "div": "div",
        "batch_size": "batch_size",
        "ipc": "ipc",             # 追加
        "train_ds_str": "train_dataset" # ログ保存名に合わせて追加 (run関数内で log_paramしている名前)
    }
    tasks = filter_finished_tasks(tasks, parquet_path=f"{this_path.parent}/{cfg['exp_name']}/_results.parquet", key_map=key_map)
    # print(len(tasks))
    parallel_run(tasks, avoid_used=True)
    # parallel_run(tasks, avoid_used=True, gpu_ids=[0,1,2,3,4,5,6,7])

if __name__ == "__main__":
    main()
