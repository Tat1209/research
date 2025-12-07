from exp import exp
from gpu_scheduler import parallel_run

def main():
    cfg = {}

    cfg["exp_name"] = "exp_model"

    model = ["resnet18", "mobilenet_v2", "efficientnet_b0", "convnext_tiny", "resnet50", "wide_resnet50_2"]
    batch_size = [128, 128, 128, 32, 32, 32]
    cfg["model_batch"] = list(zip(model, batch_size))
    cfg["train_ds_str"] = "cifar100_train"
    cfg["val_ds_str"] = "cifar100_val"

    cfg["base_epochs"] = 200
    cfg["base_ndata"] = 10000

    cfg["wd"] = [1e-3, 0]
    cfg["div"] = [1, 2, 4, 8, 16, 32]
    cfg["ndata"] = [50000, 20000, 10000, 5000, 2000, 1000, 500, 100]
    cfg["optim_lr"] = [("sgd", 0.1), ("adamw", 0.005)]
    
    cfg["num_threads"] = 8
    cfg["num_interop_threads"] = 4
    cfg["compile"] = True
    
    parallel_run(task_func=exp, config=cfg, check_interval=10.0, avoid_used=True)

if __name__ == "__main__":
    main()