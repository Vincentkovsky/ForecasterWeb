import argparse
import os
import torch
import sys

sys.path.insert(0, "Autoformer-main/")
from exp.exp_main import Exp_Main
import random
import numpy as np
import pandas as pd


def run(df: pd.DataFrame):
    args = argparse.Namespace()

    args.root_path = "./"
    args.data_path = "data.csv"

    args.is_training = 1

    args.use_gpu = False
    args.use_multi_gpu = False
    args.model_id = "weather_96_96"
    args.model = "Autoformer"
    args.data = "custom"
    args.features = "M"
    args.freq = "h"
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 96
    args.e_layers = 2
    args.d_layers = 1
    args.factor = 3
    args.enc_in = 31
    args.dec_in = 31
    args.c_out = 31
    args.des = "Exp"
    args.itr = 1
    args.target = "temp"
    args.train_epochs = 10

    args.d_model = 512
    args.n_heads = 8
    args.d_ff = 2048
    args.dropout = 0.05
    args.activation = "gelu"
    args.distil = True
    args.batch_size = 32
    args.patience = 3
    args.learning_rate = 0.0001
    args.lradj = "type1"
    args.use_amp = False
    args.loss = "mse"
    args.moving_avg = 25
    args.embed = "timeF"
    args.output_attention = False
    args.num_workers = 10
    args.checkpoints = "Autoformer-main/checkpoints/"

    print("Args in experiment:")
    print(args)

    print("-------")

    Exp = Exp_Main

    setting = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        0,
    )
    exp = Exp(args, df)
    print(">>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
    return exp.predict(setting, True)


if __name__ == "__main__":
    run()
