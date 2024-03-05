import numpy as np

import jax
import jax.numpy as jnp

from jax import random
from jax.lib import xla_bridge
from jax.sharding import PartitionSpec as PS
from jax.sharding import Mesh
from jax.experimental.shard_map import shard_map
from jax.experimental.mesh_utils import create_device_mesh

from ring_attention_jax import ring_attention

from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union


def get_jax_mesh(axis_dims, names):
    if axis_dims.startswith("!"):
        mesh_axis_splitting = True
        axis_dims = axis_dims[1:]
    else:
        mesh_axis_splitting = False

    if ":" in axis_dims:
        dims = []
        dim_names = []
        for axis in axis_dims.split(","):
            name, dim = axis.split(":")
            assert name in names
            dims.append(int(dim))
            dim_names.append(name)
        assert set(dim_names) == set(names)
    else:
        dims = [int(x) for x in axis_dims.split(",")]
        dim_names = names
    assert len(dims) == len(names)
    mesh_shape = np.arange(jax.device_count()).reshape(dims).shape
    if mesh_axis_splitting:
        physical_mesh = np.array(jax.devices()).reshape(mesh_shape)
    else:
        physical_mesh = create_device_mesh(mesh_shape)
    return Mesh(physical_mesh, dim_names)


class LMConfig:
    @staticmethod
    def get_jax_mesh(axis_dims):
        return get_jax_mesh(axis_dims, ("dp", "fsdp", "tp", "sp"))


rng_key = random.PRNGKey(42)

q = random.normal(rng_key, (1, 2, 131072, 512))  # (batch, heads, seq, dim)
print("Q Shape :", q.shape)  # (1, 2, 131072, 512
k = random.normal(rng_key, (1, 2, 131072, 512))
print("K Shape :", k.shape)  # (1, 2, 131072, 512
v = random.normal(rng_key, (1, 2, 131072, 512))
print("V Shape :", v.shape)  # (1, 2, 131072, 512
mask = random.randint(
    rng_key,
    (
        1,
        131072,
    ),
    0,
    2,
)  # (batch, seq)
print("Mask Shape :", mask.shape)  # (1, 131072)

platform = xla_bridge.get_backend().platform

print("Platform :", platform)
if platform == "tpu":
    ring_attention_fn = ring_flash_attention_tpu
else:
    ring_attention_fn = ring_attention


deterministic = True
dropout_rng = None


def get_gradient_checkpoint_policy(name):
    return {
        "everything_saveable": jax.checkpoint_policies.everything_saveable,
        "nothing_saveable": jax.checkpoint_policies.nothing_saveable,
        "checkpoint_dots": jax.checkpoint_policies.checkpoint_dots,
        "checkpoint_dots_with_no_batch_dims": jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    }[name]


blockwise_kwargs = dict(
    deterministic=deterministic,
    dropout_rng=dropout_rng,
    attn_pdrop=0.0,
    causal=True,
    query_chunk_size=1024,
    key_chunk_size=1024,
    dtype=jnp.float32,
    policy=get_gradient_checkpoint_policy("nothing_saveable"),
    precision=None,
    prevent_cse=not True,
)
mesh_dims = "1,-1,1,1"
mesh = LMConfig.get_jax_mesh(mesh_dims)

ring_attention_sharded = shard_map(
    partial(
        ring_attention_fn, axis_name="sp", float32_logits=True, blockwise_kwargs=blockwise_kwargs
    ),
    mesh=mesh,
    in_specs=(
        PS(("dp", "fsdp"), "sp", "tp", None),
        PS(("dp", "fsdp"), "sp", "tp", None),
        PS(("dp", "fsdp"), "sp", "tp", None),
        PS(("dp", "fsdp"), None, None, None),
        PS(("dp", "fsdp"), None, None, None),
    ),
    out_specs=PS(("dp", "fsdp"), "sp", "tp", None),
    check_rep=False,
)

attn_output = ring_attention_sharded(q, k, v, mask)
print("attn output shape :", attn_output.shape)  # (1, 2, 131072, 512)
