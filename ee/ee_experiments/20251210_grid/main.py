from exp import exp
from gpu_scheduler import parallel_run

def main():
    cfg = {}

    cfg["exp_name"] = "exp_bl"
    cfg["train_ds_str"] = "cifar100_train"
    cfg["val_ds_str"] = "cifar100_val"

    cfg["wd"] = [1e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1, 3e-1]
    cfg[("optim_str", "max_lr")] = list(zip(["adamw", "sgd"], [0.005, 0.1]))

    model_str = ["resnet18"]
    batch_size = [128]
    cfg[("model_str", "batch_size")] = list(zip(model_str, batch_size))


    ndata = [50000, 20000, 10000, 5000, 2000, 1000, 500, 100]
    epochs = [200 if n >= 20000 else 20000 * 200 // n for n in ndata]
    cfg[("ndata", "epochs")] = list(zip(ndata, epochs))

    cfg["div"] = [1]
    
    
    cfg["num_threads"] = 4
    cfg["num_interop_threads"] = 4
    cfg["num_workers"] = 2
    cfg["compile"] = True
    
    parallel_run(task_func=exp, config=cfg, avoid_used=True, gpu_ids=[0,1,2,3])

if __name__ == "__main__":
    main()