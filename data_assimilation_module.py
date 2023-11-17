import copy
import tensorflow as tf


def params_data_assimilation_module(parser):
    pass


def initialize_data_assimilation_module(params, state):

    state.ensemble_size = 2
    state.ensemble_iter = 0
    state.ensemble_t = None
    state.ensemble_x = []
    pass


def update_data_assimilation_module(params, state):


    # reset time
    if state.ensemble_iter >= state.ensemble_size -1 or state.ensemble_t is None:
        state.ensemble_t = state.t.numpy()
        state.ensemble_iter = 0

    else:
        state.t.assign(state.ensemble_t)
        state.ensemble_iter += 1
    pass
    print(state.t.numpy())

def finalize_data_assimilation(params, state):
    pass
