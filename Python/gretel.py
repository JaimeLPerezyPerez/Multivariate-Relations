# %%
import pandas as pd
import numpy as np

import dateutil.parser
import matplotlib.pyplot as plt
import matplotlib.dates as md

import torch

from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType


# %%

data = pd.read_csv("data_test.csv")


# %%
new_time = []
for x in data["time"]:
    new_time.append(dateutil.parser.parse(x))

data["time"] = new_time

print("CUDA:", torch.cuda.is_available())

# %%

model = DGAN(DGANConfig(

    max_sequence_len= 15093,
    sample_len= 1161,
    batch_size= 100,
    apply_feature_scaling=True,
    # apply_example_scaling=False,
    apply_example_scaling=True,
    use_attribute_discriminator=False,
    generator_learning_rate=1e-4,
    discriminator_learning_rate=1e-4,
    epochs= 10000,
    # For quick experiments, reduce epochs, but this does impact quality
    # epochs=1000,
))

model.train_dataframe(
    df = data,
    example_id_column= "id",
    time_column = "time",
    df_style= "long"
)

# Generate synthetic data
synthetic_data = model.generate_dataframe(25)
synthetic_data.to_csv("Gretel.csv",index=False)


