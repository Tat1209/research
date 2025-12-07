from exp import exp
from gpu_scheduler import parallel_run
from torchvision import models

def main():
    cfg = {}

    cfg["exp_name"] = "exp_tmp"

    # cfg["model_name"] = ["resnet18", "mobilenet_v2", "efficientnet_b0"]
    cfg["model_name"] = ["resnet50", "convnext_tiny", "wide_resnet50_2", ]
    cfg["train_ds_str"] = "cifar100_train"
    cfg["val_ds_str"] = "cifar100_val"

    cfg["base_epochs"] = 200
    cfg["batch_size"] = 32
    cfg["base_ndata"] = 10000

    cfg["wd"] = [1e-3, 0]
    cfg["div"] = [1, 2, 4, 8, 16, 32]
    cfg["ndata"] = [50000, 20000, 10000, 5000, 2000, 1000, 500, 100]
    cfg["optim_lr"] = [("sgd", 0.1), ("adamw", 0.005)]
    
    parallel_run(
        task_func=exp,
        config=cfg,
        check_interval=10.0,
        avoid_used=True,
        )

if __name__ == "__main__":
    main()