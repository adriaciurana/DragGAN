from os.path import exists
from urllib.parse import urlparse

import torch


def on_change_single_global_state(keys, value, global_state, map_transform=None):
    if map_transform is not None:
        value = map_transform(value)

    curr_state = global_state
    if isinstance(keys, str):
        last_key = keys

    else:
        for k in keys[:-1]:
            curr_state = curr_state[k]

        last_key = keys[-1]

    curr_state[last_key] = value
    return global_state


def is_local(url: str) -> bool:
    url_parsed = urlparse(url)
    if url_parsed.scheme in ("file", ""):  # Possibly a local file
        return exists(url_parsed.path)
    return False


class TrainableLatent:
    # TsODO Create this concept for each ARCH
    def __init__(self, latent: torch.Tensor, trainable_w_dims: int = 6, dim: int = 1):
        self._trainable_w_dims = trainable_w_dims
        self._dim = dim

        self._latent = latent
        self._trainable_latent = None
        self._fix_latent = None

        self._compute_latent()

    def _compute_latent(self):
        self._trainable_latent, self._fix_latent = torch.split(
            self._latent, [self._trainable_w_dims, self._latent.shape[self._dim] - self._trainable_w_dims], dim=self.dim
        )
        self._trainable_latent.requires_grad = True
        self._fix_latent.requires_grad = False

    @property
    def latent(self):
        return torch.cat((self._trainable_latent, self._fix_latent), dim=self._dim)

    @property
    def trainable_latent(self):
        if self._trainable_latent is None:
            self._compute_latent()
        return self._trainable_latent

    @property
    def fix_latent(self):
        if self._fix_latent is None:
            self._compute_latent()
        return self._fix_latent

    @property
    def trainable_w_dims(self):
        return self._trainable_w_dims

    @trainable_w_dims.setter
    def set_trainable_w_dims(self, value: int):
        self._trainable_w_dims = value
        self._compute_latent()

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def set_dim(self, value: int):
        self._dim = value
        self._compute_latent()
