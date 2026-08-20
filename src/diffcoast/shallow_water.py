import jax.numpy as jnp
from jax import jit

def step(h, u, v, tau_x, tau_y, dx, dy, dt, g, H, rho, r):
    # ... same as your notebook step function ...
    return h_new, u_new, v_new

def run_simulation(Nx=50, Ny=50, Lx=1000.0, Ly=1000.0, nt=1000, ...):
    # ... initialize arrays and loop over time ...
    return h, u, v
