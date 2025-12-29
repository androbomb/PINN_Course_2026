import matplotlib.pyplot as plt
import numpy as np

def plot_contourplot(T: np.array, X: np.array, Z: np.array, diffusion_parameter: float, levels: int = 75):
    fig, ax = plt.subplots(1,1,dpi=300)
    im = ax.contourf(
        T, X, Z,
        levels=levels,
        origin = 'lower',
        cmap='jet',
        alpha = 1.0
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('u(x,t)')
    #optional countour lines:
    contours = ax.contour(
        T, X, Z, 
        #colors='green'
        cmap='viridis'
    )
    ax.clabel(contours, inline=True, fontsize=8)
    
    ax.set_title(f"Exact solution with D={diffusion_parameter:.1f}")
    ax.set_xlabel('t (s)')
    ax.set_ylabel('x (m)')
    return fig