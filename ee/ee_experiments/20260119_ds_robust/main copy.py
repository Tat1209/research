from pathlib import Path
from train import run
from pl_utils import filter_finished_tasks
from gpu_scheduler import parallel_run, generate_tasks_grid

def main():
    this_path = Path(__file__).resolve()
    cfg = {}

    # cfg["exp_name"] = "exp_tmp"
    cfg["exp_name"] = "exp_robust"
    cfg["model_str"] = "resnet18"
    
    train_ds_str = ["tiny-imagenet_train"]
    val_ds_str = ["tiny-imagenet_test"]
    batch_size = [128]
    # train_ds_str = ["stl10_train", "tiny-imagenet_train", "caltech101_labeled", "cub200_train", "flowers102_train", "oxford-pet_train", "mnist_train", "fashion-mnist_train", "svhn_train", "cifar100_train", "cifar10_train"]
    # val_ds_str = ["stl10_test", "tiny-imagenet_test", "caltech101_labeled", "cub200_test", "flowers102_test", "oxford-pet_test", "mnist_test", "fashion-mnist_test", "svhn_test", "cifar100_test", "cifar10_test"]
    # batch_size = [128, 128, 32, 32, 32, 32, 128, 128, 128, 128, 128]
    max_lr = [5e-3 * batch_size / 128 for batch_size in batch_size]
    
    cfg[("train_ds_str", "val_ds_str", "batch_size", "max_lr")] = list(zip(train_ds_str, val_ds_str, batch_size, max_lr))

    cfg["optim_str"] = "adamw"


    cfg["wd"] = 5e-2
    cfg["ipc"] = [1]

    # ndata = [50000, 20000, 10000, 5000, 2000, 1000, 500, 100]
    # epochs = [200 if n >= 10000 else 10000 * 200 // n for n in (ipc * 100)]
    # cfg[("ipc", "epochs")] = list(zip(ipc, epochs))

    cfg["div"] = [4]

    cfg["flex_ch"] = True
    
    cfg["num_threads"] = 4
    cfg["num_interop_threads"] = 4
    cfg["num_workers"] = 2
    cfg["compile"] = True
    

    tasks = generate_tasks_grid(run, cfg)
    # key_map = {
    #     "model_str": "model_arc",
    #     "div": "div",
    #     "batch_size": "batch_size",
    #     "ipc": "ipc",             # 追加
    #     "train_ds_str": "train_dataset" # ログ保存名に合わせて追加 (run関数内で log_paramしている名前)
    # }
    # tasks = filter_finished_tasks(tasks, parquet_path=f"{this_path.parent}/{cfg['exp_name']}/_results.parquet", key_map=key_map)
    # print(len(tasks))
    # parallel_run(tasks, avoid_used=True)
    parallel_run(tasks, avoid_used=True, gpu_ids=[0])
    # parallel_run(tasks, avoid_used=True, free_mem_th=49000)

if __name__ == "__main__":
    main()
