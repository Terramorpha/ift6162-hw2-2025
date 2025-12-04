import copy
import pdb
import random
import sys
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import (
    AffineTransform,
    Distribution,
    Normal,
    TanhTransform,
    TransformedDistribution,
)

import hw2
from calciner import CalcinerEnv, ConstantTemperatureController, evaluate_baseline
from hw2 import (
    MLP,
    Controller,
    Trajectory,
    importance_sampling_policy_gradients_loss,
    policy_gradients_loss,
    sample_trajectory,
)
