import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
import streamlit as st
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer

this_path = Path(__file__) if '__file__' in globals() else Path("<unknown>.ipynb").resolve()
work_path = next((p for p in this_path.parents if p.name == "research"), None)
tools_path = work_path / Path("../torch-tools")
sys.path.append(str(tools_path))

from run_manager import RunViewer

rv = RunViewer(exp_path=this_path.parent)
df_base = rv.fetch_results(met_listed=False)

nested_columns = [name for name, dtype in zip(df_base.columns, df_base.dtypes) if dtype.is_nested()]
df_base = df_base.with_columns([pl.col(name).list.last().alias(f"{name}") for name in nested_columns])

st.set_page_config(
    page_title="Use Pygwalker In Streamlit",
    layout="wide"
)
pyg_app = StreamlitRenderer(df_base)
 
pyg_app.explorer()