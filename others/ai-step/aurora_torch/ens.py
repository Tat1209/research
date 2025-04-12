import numpy as np
import time

from model import Model
import post



class Ens:
    def __init__(self):
        pass

    def cross(self, model_list, categorize=True):
        mod_res = {"vAcc":[], "outputs_l":[], "cross_merge":None, "val_feat":None, "val_label":None}
        for model in model_list:
            hist = model.fit()

            summary = model.pred(categorize=False)
            val_summary = model.pred(categorize=False, val=True)

            val_outputs = val_summary["outputs"]
            val_labels = val_summary["labels"]
            
            mod_res["vAcc"].append(hist["vAcc"][-1])
            mod_res["outputs_l"].append(summary["outputs"])

            if mod_res["val_feat"] is None: mod_res["val_feat"] = val_outputs
            else: mod_res["val_feat"] = np.concatenate((mod_res["val_feat"], val_outputs), axis=0)

            if mod_res["val_label"] is None: mod_res["val_label"] = val_labels
            else: mod_res["val_label"] = np.concatenate((mod_res["val_label"], val_labels), axis=0)
            
            post.postprocess(summary["outputs"], hist, model)
        vAcc_sum = (np.array(mod_res["outputs_l"]) * np.array(mod_res["vAcc"])[:, np.newaxis, np.newaxis]).sum(axis=0)
        cross_merge =  vAcc_sum / np.array(mod_res["vAcc"]).sum()

# ほんとはこれ消す
        post.postprocess(cross_merge, None, model)
        time.sleep(1)

        if categorize: cross_merge = np.argmax(cross_merge, axis=1)
        mod_res["cross_merge"] = cross_merge

        post.postprocess(mod_res["cross_merge"], None, model)
        
        return mod_res