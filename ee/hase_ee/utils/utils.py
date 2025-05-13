import yaml
import wandb
import pandas as pd

def load_config(config_file):
    """
    YAML設定ファイルを読み込む関数
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def runs2df(entity, project):
    wandb.login()
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    result = pd.DataFrame()
    for run in runs:
        r = run.config
        r['id'] = run.id
        r['group'] = run.group
        r['name'] = run.name
        r['status'] = run.state
        r.update(run.summary)
        result = pd.concat([result, pd.DataFrame(pd.Series(r)).T], axis=0)
    result = result.reset_index(drop=True)
    return result
