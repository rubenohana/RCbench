from jacho.models.generic import GenericEchoState
from jacho.layers.output import Residual

from jax import random
import numpy as np
import jax.numpy as jnp

key = random.PRNGKey(42)

import time
from jax.interpreters import xla


def res_update_time(input_data, reservoir_type, n_reservoirs, reservoir_args):

  n_out = input_data.shape[-1]
  times = np.zeros([len(n_reservoirs)])

  for num in range(len(n_reservoirs)):
    n_reservoir = n_reservoirs[num]
    norm_factor = 1.1 * jnp.sqrt(n_out / n_reservoir)
    output_layer_args = (norm_factor, )

    model = GenericEchoState(n_reservoir, reservoir_type, reservoir_args,
                            n_out, Residual, output_layer_args)
    state = model.initialize_state(key, n_reservoir)
    params = model.init(key, state, input_data[0]) # initializing the parameters and state
    start = time.time()
    model.apply(params, state, input_data, method=model.run_reservoir)
    times[num] = time.time() - start
    xla._xla_callable.cache_clear()
    del state
    del params
    print(n_reservoir)
  return times

if __name__ == "__main__":
    res_update_time()