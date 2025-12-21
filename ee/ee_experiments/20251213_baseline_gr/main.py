from train import exp
from gpu_scheduler import parallel_run, generate_tasks_grid

def main():
    cfg = {}

    cfg["exp_name"] = "exp_model"
    cfg["train_ds_str"] = "cifar100_train"
    cfg["val_ds_str"] = "cifar100_val"
    cfg["optim_str"] = "adamw"

    model_str = ["resnet18"]
    batch_size = [128]
    max_lr = [5e-3 * batch_size / 128 for batch_size in batch_size]
    cfg[("model_str", "batch_size", "max_lr")] = list(zip(model_str, batch_size, max_lr))

    cfg["wd"] = [3, 5, 7, 10]
    # cfg["wd"] = [1e-4, 5e-4, 1e-3, 3e-3, 1e-2, 3e-2, 7e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1]

    ndata = [50000, 10000, 1000, 500, 100]

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
    parallel_run(tasks, avoid_used=True, gpu_ids=[0,1,2,3,4,5,6,7])

if __name__ == "__main__":
    main()