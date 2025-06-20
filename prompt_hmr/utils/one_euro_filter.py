import math
import numpy as np
import torch


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
    
    
def smooth_one_euro(vals, min_cutoff=0.004, beta=0.7, is_rot=False):
    '''
    vals: (N, D)
    return: (N, D)
    '''
    # min_cutoff: Decreasing the minimum cutoff frequency decreases slow speed jitter
    # beta: Increasing the speed coefficient(beta) decreases speed lag.
    
    if is_rot:
        from .rotation_conversions import matrix_to_quaternion, quaternion_to_matrix
        vals = matrix_to_quaternion(torch.from_numpy(vals)).numpy()

    one_euro_filter = OneEuroFilter(
        np.zeros_like(vals[0]),
        vals[0],
        min_cutoff=min_cutoff,
        beta=beta,
    )

    vals_hat = np.zeros_like(vals)

    # initialize
    vals_hat[0] = vals[0]

    for idx, val in enumerate(vals[1:]):
        idx += 1

        t = np.ones_like(val) * idx
        val = one_euro_filter(t, val)
        vals_hat[idx] = val

    if is_rot:
        # convert back to rotation matrix
        vals_hat = quaternion_to_matrix(torch.from_numpy(vals_hat)).numpy()
        
    return vals_hat
