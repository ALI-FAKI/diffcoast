"""
2D depth-integrated shallow water model for coastal applications.

The model uses the Arakawa C-grid and an explicit time-stepping scheme.
All arrays are JAX arrays, enabling automatic differentiation and GPU execution.

Parameters:
- h: water surface elevation (m)   [Ny, Nx]
- u: depth-averaged velocity in x (m/s)  [Ny, Nx+1]
- v: depth-averaged velocity in y (m/s)  [Ny+1, Nx]
"""

import jax.numpy as jnp
from jax import jit

def step(h, u, v, tau_x, tau_y, dx, dy, dt, g, H, rho, r):
    """
    One time step of the shallow water equations.
    
    Args:
        h: water surface elevation at cell centers, shape (Ny, Nx)
        u: x‑velocity at east‑west faces, shape (Ny, Nx+1)
        v: y‑velocity at north‑south faces, shape (Ny+1, Nx)
        tau_x, tau_y: wind stress components (N/m^2)
        dx, dy: grid spacing (m)
        dt: time step (s)
        g: gravitational acceleration (m/s^2)
        H: mean water depth (m)
        rho: water density (kg/m^3)
        r: linear bottom friction coefficient (1/s)

    Returns:
        h_new, u_new, v_new
    """
    # Update water height: dh/dt = -H * (du/dx + dv/dy)
    du_dx = (u[:, 1:] - u[:, :-1]) / dx
    dv_dy = (v[1:, :] - v[:-1, :]) / dy
    h_new = h - dt * H * (du_dx + dv_dy)

    # Update u velocity: du/dt = -g * dh/dx + tau_x/(rho*H) - r*u
    # At interior points only (skip boundaries)
    dh_dx = (h[:, 1:] - h[:, :-1]) / dx
    u_int = u[:, 1:-1]  # interior points, shape (Ny, Nx-1)
    u_new = u.at[:, 1:-1].set(
        u_int + dt * (-g * dh_dx + tau_x / (rho * H) - r * u_int)
    )
    # Closed boundaries: u = 0 at western and eastern walls
    u_new = u_new.at[:, 0].set(0.0)
    u_new = u_new.at[:, -1].set(0.0)

    # Update v velocity: dv/dt = -g * dh/dy + tau_y/(rho*H) - r*v
    dh_dy = (h[1:, :] - h[:-1, :]) / dy
    v_int = v[1:-1, :]  # interior points, shape (Ny-1, Nx)
    v_new = v.at[1:-1, :].set(
        v_int + dt * (-g * dh_dy + tau_y / (rho * H) - r * v_int)
    )
    # Closed boundaries: v = 0 at southern and northern walls
    v_new = v_new.at[0, :].set(0.0)
    v_new = v_new.at[-1, :].set(0.0)

    return h_new, u_new, v_new


def run_simulation(
    Lx=1000.0,
    Ly=1000.0,
    Nx=50,
    Ny=50,
    nt=1000,
    tau_x=0.1,
    tau_y=0.0,
    g=9.81,
    H=10.0,
    rho=1025.0,
    r=0.001,
    dt=None,
    save_every=100,
):
    """
    Run a 2D wind‑driven simulation in a closed rectangular basin.

    Args:
        Lx, Ly: domain size in meters
        Nx, Ny: number of grid cells in each direction
        nt: number of time steps
        tau_x, tau_y: wind stress (N/m^2)
        g, H, rho, r: physical parameters
        dt: time step (s). If None, automatically computed from CFL condition.
        save_every: interval for saving history (set to 0 to save only final state)

    Returns:
        h, u, v: final fields after nt steps
        history (optional): dict with lists of saved states if save_every > 0
    """
    dx = Lx / Nx
    dy = Ly / Ny

    # Automatic time step from CFL (barotropic wave speed sqrt(g*H))
    if dt is None:
        c = jnp.sqrt(g * H)
        dt = 0.4 * min(dx, dy) / c
        print(f"Using dt = {dt:.4f} s")

    # Initialize fields
    h = jnp.zeros((Ny, Nx))
    u = jnp.zeros((Ny, Nx + 1))
    v = jnp.zeros((Ny + 1, Nx))

    # JIT compile the step function for speed
    step_jit = jit(step)

    # Time loop
    for n in range(nt):
        h, u, v = step_jit(h, u, v, tau_x, tau_y, dx, dy, dt, g, H, rho, r)

    return h, u, v
