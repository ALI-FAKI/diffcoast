import matplotlib.pyplot as plt
from diffcoast.shallow_water import run_simulation

# Run the simulation
h, u, v = run_simulation(nt=1000, save_every=0)

# Plot water surface and velocity
dx = 1000.0 / 50   # must match the values used in run_simulation
dy = 1000.0 / 50
x = (jnp.arange(50) + 0.5) * dx   # but we need jnp here; we'll import
import jax.numpy as jnp

x = (jnp.arange(50) + 0.5) * dx
y = (jnp.arange(50) + 0.5) * dy

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

c1 = ax1.contourf(x, y, h, levels=20, cmap='RdBu_r')
ax1.set_title('Water surface elevation (m)')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
fig.colorbar(c1, ax=ax1)

# Average velocities to cell centers for plotting
u_center = 0.5 * (u[:, :-1] + u[:, 1:])
v_center = 0.5 * (v[:-1, :] + v[1:, :])
skip = 3
ax2.quiver(x[::skip], y[::skip], u_center[::skip, ::skip], v_center[::skip, ::skip], scale=2)
ax2.set_title('Current velocity vectors')
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_aspect('equal')

plt.tight_layout()
plt.show()
