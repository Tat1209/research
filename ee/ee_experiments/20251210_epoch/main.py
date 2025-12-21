from exp import exp
from gpu_scheduler import parallel_run, generate_tasks_grid

def main():
    cfg = {}

    cfg["exp_name"] = "exp_epoch_re"
    cfg["train_ds_str"] = "cifar100_train"
    cfg["val_ds_str"] = "cifar100_val"

    cfg["wd"] = [5e-1, 5e-2, 5e-4, 0]
    cfg["optim_str"] = "adamw"
    cfg["max_lr"] = 0.005

    model_str = ["resnet18"]
    batch_size = [128]
    cfg[("model_str", "batch_size")] = list(zip(model_str, batch_size))


    ndata = [50000, 5000, 500, 100]
    epochs = [200 if n >= 10000 else 10000 * 200 // n for n in ndata] + [200 if n >= 20000 else 20000 * 200 // n for n in ndata]
    base_epochs = [10000] * 4 + [20000] * 4
    cfg[("ndata", "epochs", "base_epochs")] = list(zip(ndata * 2, epochs, base_epochs))

    cfg["div"] = [1, 16]
    
    
    cfg["num_threads"] = 4
    cfg["num_interop_threads"] = 4
    cfg["num_workers"] = 2
    cfg["compile"] = True
    
    tasks = generate_tasks_grid(exp, cfg)
    parallel_run(tasks, avoid_used=True, gpu_ids=[0,1,2,3,4,5,6,7])

    cfg["wd"] = [5e-2, 5e-3, 5e-4, 0]
    cfg["optim_str"] = "sgd"
    cfg["max_lr"] = 0.1
    tasks = generate_tasks_grid(exp, cfg)
    parallel_run(tasks, avoid_used=True, gpu_ids=[0,1,2,3,4,5,6,7])

if __name__ == "__main__":
    main()