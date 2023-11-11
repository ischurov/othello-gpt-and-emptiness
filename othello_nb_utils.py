# This file is heavily based on 
# https://colab.research.google.com/github/likenneth/othello_world/blob/master/Othello_GPT_Circuits.ipynb
# Modified by Ilia Shchurov (Ilya Schurov)
# Available under MIT License

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from typing import List, Union, Optional
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)

torch.set_grad_enabled(False)

from neel_plotly import line, scatter, imshow, histogram

import transformer_lens.utils as utils

cfg = HookedTransformerConfig(
    n_layers=8,
    d_model=512,
    d_head=64,
    n_heads=8,
    d_mlp=2048,
    d_vocab=61,
    n_ctx=59,
    act_fn="gelu",
    normalization_type="LNPre",
)


def get_model():
    model = HookedTransformer(cfg)

    sd = utils.download_file_from_hf(
        "NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth"
    )
    # champion_ship_sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "championship_model.pth")
    model.load_state_dict(sd)
    return model


OTHELLO_ROOT = Path("./othello_world/")
import sys

sys.path.append(str(OTHELLO_ROOT / "mechanistic_interpretability"))
from mech_interp_othello_utils import (
    plot_single_board,
    to_string,
    to_int,
    int_to_label,
    string_to_label,
    OthelloBoardState,
)

board_seqs_int = torch.tensor(
    np.load(OTHELLO_ROOT / "mechanistic_interpretability/board_seqs_int_small.npy"),
    dtype=torch.long,
)
board_seqs_string = torch.tensor(
    np.load(OTHELLO_ROOT / "mechanistic_interpretability/board_seqs_string_small.npy"),
    dtype=torch.long,
)

num_games, length_of_game = board_seqs_int.shape
# print(
#     "Number of games:",
#     num_games,
# )
# print("Length of game:", length_of_game)

# fmt: off
stoi_indices = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
    10, 11, 12, 13, 14, 15, 16, 17, 
    18, 19, 20, 21, 22, 23, 24, 25, 
    26, 29, 30, 31, 32, 33, 34, 37, 
    38, 39, 40, 41, 42, 43, 44, 45, 
    46, 47, 48, 49, 50, 51, 52, 53, 
    54, 55, 56, 57, 58, 59, 60, 61, 
    62, 63,
]
alpha = "ABCDEFGH"
# fmt: on


def to_board_label(i):
    return f"{alpha[i//8]}{i%8}"


board_labels = list(map(to_board_label, stoi_indices))


def plot_square_as_board(state, diverging_scale=True, **kwargs):
    """Takes a square input (8 by 8) and plot it as a board. Can do a stack of boards via facet_col=0"""
    if diverging_scale:
        imshow(
            state,
            y=[i for i in alpha],
            x=[str(i) for i in range(8)],
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0.0,
            aspect="equal",
            **kwargs,
        )
    else:
        imshow(
            state,
            y=[i for i in alpha],
            x=[str(i) for i in range(8)],
            color_continuous_scale="Blues",
            color_continuous_midpoint=None,
            aspect="equal",
            **kwargs,
        )


def one_hot(list_of_ints, num_classes=64):
    out = torch.zeros((num_classes,), dtype=torch.float32)
    out[list_of_ints] = 1.0
    return out


def get_focus_games(num_games=50):
    """
    Returns
    =======

    focus_games_int: torch.Tensor of shape (num_games, 60)
        The integer representation of the moves in the focus games.
    focus_games_string: torch.Tensor of shape (num_games, 60)
        The string representation of the moves in the focus games.
    focus_states: torch.Tensor of shape (num_games, 60, 8, 8)
        The board states of the focus games.
    focus_valid_moves: torch.Tensor of shape (num_games, 60, 64)
        The valid moves of the focus games.

    """
    focus_games_int = board_seqs_int[:num_games]
    focus_games_string = board_seqs_string[:num_games]

    focus_states = np.zeros((num_games, 60, 8, 8), dtype=np.float32)
    focus_valid_moves = torch.zeros((num_games, 60, 64), dtype=torch.float32)
    for i in range(num_games):
        board = OthelloBoardState()
        for j in range(60):
            board.umpire(focus_games_string[i, j].item())
            focus_states[i, j] = board.state
            focus_valid_moves[i, j] = one_hot(board.get_valid_moves())
    return focus_games_int, focus_games_string, focus_states, focus_valid_moves


blank_index = 0
their_index = 1
my_index = 2


def get_linear_probe():
    full_linear_probe = torch.load(
        OTHELLO_ROOT / "mechanistic_interpretability/main_linear_probe.pth",
        map_location=torch.device("cpu"),
    )

    rows = 8
    cols = 8
    options = 3
    black_to_play_index = 0
    white_to_play_index = 1

    linear_probe = torch.zeros(cfg.d_model, rows, cols, options)
    linear_probe[..., blank_index] = 0.5 * (
        full_linear_probe[black_to_play_index, ..., 0]
        + full_linear_probe[white_to_play_index, ..., 0]
    )
    linear_probe[..., their_index] = 0.5 * (
        full_linear_probe[black_to_play_index, ..., 1]
        + full_linear_probe[white_to_play_index, ..., 2]
    )
    linear_probe[..., my_index] = 0.5 * (
        full_linear_probe[black_to_play_index, ..., 2]
        + full_linear_probe[white_to_play_index, ..., 1]
    )
    return linear_probe
