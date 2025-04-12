import numpy as np

from model import Model
import post


def ens(model_list, pr, categorize=True, fit_aug_ratio=None, tta_times=None, tta_aug_ratio=None, mixup_alpha=None):
    mod_res = {"vAcc":[], "outputs_list":[], "outputs_val":[]}
    num_models = len(model_list)
    for i, model in enumerate(model_list):
        pr.val_range = (i/num_models, (i+1)/num_models)
        hist = model.fit(pr, fit_aug_ratio=fit_aug_ratio, mixup_alpha=mixup_alpha)

        result = model.pred(pr, categorize=False, tta_times=tta_times, tta_aug_ratio=tta_aug_ratio)
        
        mod_res["vAcc"].append(hist["vAcc"][-1])
        mod_res["outputs_list"].append(result["outputs"])
        
        post.postprocess(pr, result, hist, model)
    vAcc_sum = (np.array(mod_res["results"]) * np.array(mod_res["vAcc"])[:, np.newaxis, np.newaxis]).sum(axis=0)
    ens_res =  vAcc_sum / np.array(mod_res["vAcc"]).sum()
    if categorize: ens_res = np.argmax(ens_res, axis=1)
    post.postprocess(pr, ens_res, None, None)
        