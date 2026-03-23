import sys
from pathlib import Path

import torch
from torchvision import models

this_path = Path(__file__) if "__file__" in globals() else Path("<undefined>.ipynb").resolve()
work_path = next((p for p in this_path.parents if p.name == "research"), None)
tools_path = work_path / Path("../torch-tools")
sys.path.append(str(tools_path))
ee_tools_path_p = work_path / Path("ee")
sys.path.append(str(ee_tools_path_p))

from ee_tools.ee_refiner import EERefiner
from exp_manager import ExpManager
from model_stats import profile_model_stats
from network import Network


def main():
    exp_name = "exp_calc_cost"
    exp = ExpManager(exp_path=this_path.parent / exp_name, exp_tpl="exp_tpl_ee")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 100
    batch_size = 128

    for div in [1, 2, 4, 8, 16, 32]:
        run_mgr = exp.create_run()
        stats = {}
        stats["div"] = div
        stats["num_classes"] = num_classes
        stats["batch_size"] = batch_size

        net = getattr(models, "resnet18")
        refiner = (
            EERefiner(net(num_classes=num_classes))
            .adapt_structure(target="low_res")  # CIFAR用の設定
            .multi_narrow(div=div, agg="mean", flex_ch=True)
            .init_weights()
        )

        network = Network(refiner.build().to(device))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            network.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
            fused=True,
        )
        # optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=5e-2, fused=True)

        stats |= profile_model_stats(
            model=network,
            num_classes=num_classes,
            input_shape=(3, 32, 32),
            batch_size=batch_size,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp_dtype=torch.bfloat16,
        )

        run_mgr.log_params(stats)

        run_mgr.sync()
        exp.ref_results()


if __name__ == "__main__":
    main()
