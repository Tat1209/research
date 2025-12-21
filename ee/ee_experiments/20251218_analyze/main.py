from train import exp
from gpu_scheduler import parallel_run, generate_tasks_grid

def main():
    cfg = {}

    cfg["exp_name"] = "exp_model"
    cfg["train_ds_str"] = "cifar100_train"
    cfg["val_ds_str"] = "cifar100_val"

    cfg["model_str"] = ["resnet18"]
    
    cfg["batch_size"] = 128
    cfg[("optim_str", "max_lr", "wd")] = list(zip(["sgd"], [0.1], [5e-4]))
    # cfg[("optim_str", "max_lr", "wd")] = list(zip(["adamw", "sgd"], [0.005, 0.1], [5e-2, 5e-4]))

    ndata = [50000, 1000]

    epochs = [200 if n >= 10000 else 10000 * 200 // n for n in ndata]
    cfg[("ndata", "epochs")] = list(zip(ndata, epochs))

    cfg["div"] = [1, 32]

    cfg["cifar_style"] = True
    cfg["flex_ch"] = False
    
    cfg["num_threads"] = 4
    cfg["num_interop_threads"] = 4
    cfg["num_workers"] = 2
    cfg["compile"] = False
    
    tasks = generate_tasks_grid(exp, cfg)
    parallel_run(tasks, avoid_used=True, gpu_ids=[0,1,2,3,4,5,6,7])

if __name__ == "__main__":
    main()