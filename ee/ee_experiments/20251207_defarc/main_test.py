from exp import exp
from gpu_scheduler import parallel_run

def main():
    cfg = {}

    cfg["exp_name"] = "exp_tmp"

    cfg["model_name"] = "resnet18"
    cfg["train_ds_str"] = "cifar100_train"
    cfg["val_ds_str"] = "cifar100_val"

    cfg["base_epochs"] = 2
    cfg["batch_size"] = 128
    cfg["base_ndata"] = 10000
    
    cfg["wd"] = [1e-4]
    cfg["div"] = [1, 4]
    cfg["ndata"] = [10000]
    cfg["optim_lr"] = [("sgd", 0.1)]
    
    parallel_run(
        task_func=exp,
        config=cfg,
        check_interval=10.0,
        avoid_used=True,
        )

if __name__ == "__main__":
    main()