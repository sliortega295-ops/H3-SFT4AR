"""Deterministic startup hook used by the H3 A/B experiments."""

import os
import random

seed_value = os.environ.get("H3_EXPERIMENT_SEED")
if seed_value is not None:
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    seed = int(seed_value) + rank
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
