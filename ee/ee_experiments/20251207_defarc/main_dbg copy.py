from exp import exp
from gpu_scheduler import parallel_run
from utils import get_resource_config

def main():
    cfg = {}

    cfg["exp_name"] = "exp_test"
    # cfg["exp_name"] = "exp_model"

    # --- Linked Parameters (Zip) ---
    # 文字列キー "model_batch" ではなく、タプルキー ("model_str", "batch_size") を使用します
    model_str = ["resnet50"]
    batch_size = [48]
    cfg[("model_str", "batch_size")] = list(zip(model_str, batch_size))

    # --- Fixed Parameters ---
    cfg["train_ds_str"] = "cifar100_train"
    cfg["val_ds_str"] = "cifar100_val"
    cfg["ndata"] = [1000]

    # --- Grid Parameters (Product) ---
    cfg["wd"] = [1e-3]
    cfg["epochs"] = [50]
    cfg["div"] = [1, 2, 4, 8, 16, 32]

    # --- Linked Parameters (Zip) ---
    # ここもタプルキーにして、("sgd", 0.1) がそれぞれ optim_str, max_lr に展開されるようにします
    cfg[("optim_str", "max_lr")] = [("sgd", 0.1)]
    
    cfg["num_threads"] = 4
    cfg["num_interop_threads"] = 4
    cfg["num_workers"] = 2
    
    # リストで渡すとGrid扱いになります（要素数1なので実質固定ですが、記法として正しいです）
    cfg["compile"] = [False]
    
    parallel_run(task_func=exp, config=cfg, check_interval=10.0, avoid_used=True, )

if __name__ == "__main__":
    main()