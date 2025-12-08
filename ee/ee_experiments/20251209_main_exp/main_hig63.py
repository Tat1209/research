from exp import exp
from gpu_scheduler import parallel_run

def main():
    cfg = {}

    cfg["exp_name"] = "exp_model"
    cfg["train_ds_str"] = "cifar100_train"
    cfg["val_ds_str"] = "cifar100_val"

    cfg["wd"] = [1e-3, 0]
    cfg[("optim_str", "max_lr")] = list(zip(["adamw", "sgd"], [0.005, 0.1]))

    model_str = ["resnet18", "mobilenet_v2", "resnet50"]
    batch_size = [128, 128, 48]
    cfg[("model_str", "batch_size")] = list(zip(model_str, batch_size))


    ndata = [50000, 20000, 10000, 5000, 2000, 1000, 500, 100]
    epochs = [200 if n >= 20000 else 10000 * 200 // n for n in ndata]
    cfg[("ndata", "epochs")] = list(zip(ndata, epochs))

    cfg["div"] = [1, 2, 4, 8, 16, 32]
    
    
    cfg["num_threads"] = 4
    cfg["num_interop_threads"] = 4
    cfg["num_workers"] = 2
    cfg["compile"] = True
    
    parallel_run(task_func=exp, config=cfg, check_interval=10.0, avoid_used=True, util_th=10, free_mem_th=20000)

if __name__ == "__main__":
    main()