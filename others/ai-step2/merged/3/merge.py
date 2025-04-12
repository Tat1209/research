import torch
import polars as pl


ens_output = None
ens_label = None


def add_output(outputs, labels):
    global ens_output
    global ens_label
    if ens_output is None:
        ens_output = outputs
        ens_label = labels
    else:
        ens_output += outputs


for i in [0, 26, 27, 28, 5]:
    outputs, labels = torch.load(f"/home/haselab/Documents/tat/Research/app/ai-step2/exp_submit_single/{i}/output_t.pt")
    add_output(outputs, labels)


_, pred = torch.max(ens_output, dim=1)
df_out = pl.DataFrame({"fname": labels, "pred": pred.tolist()})

df_out.write_csv("/home/haselab/Documents/tat/Research/app/ai-step2/merged/3/merged.csv", include_header=False)

# 60.347
