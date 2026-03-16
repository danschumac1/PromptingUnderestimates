import os
from typing import Optional, List
from matplotlib import pyplot as plt
import numpy as np

# ORIGINAL VERSION
# def plot_time_series(
#         X: np.ndarray, 
#         method: str,
#         title: str,
#         xlabs: str, 
#         ylabs: str,
#         legends: Optional[List[str]],
#         save_path: str,
#         recreate: bool = False,
#     ) -> str:
#     """
#     Simple time series plotter.
#     - X can be (T,), (T,V), or (V,T)
#     - xlabs, ylabs are direct axis-label strings
#     - legends: list of names for each variable, or None for no legend
#     - method can be "line", "spectrogram", or "imu" (for 6-channel IMU data)
#     """
#     assert method in ["line", "spectrogram", "imu"], f"Unsupported method {method}"
#     # Skip if already exists
#     if os.path.exists(save_path) and not recreate:
#         return save_path
#     # Ensure parent folder exists
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     # Normalize shape
#     X = np.asarray(X).squeeze()

#     if X.ndim == 1:
#         X = X[:, None]              # (T,) → (T,1)
#     elif X.ndim == 2:
#         T, V = X.shape
#         if T < V:                   # if transposed (V,T), fix it
#             X = X.T
#     else:
#         raise ValueError(f"Unsupported shape {X.shape}")

#     T, V = X.shape
#     x = np.arange(T)

#     if method == "imu":
#         # Special handling for 6-channel IMU data (3 acc + 3 gyro)
#         # Create two subplots: accelerometer and gyroscope
#         fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=100, sharex=True)
        
#         # Colors for x, y, z axes
#         colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
        
#         # Accelerometer subplot (first 3 channels)
#         ax_acc = axes[0]
#         for i in range(min(3, V)):
#             label_i = legends[i] if (legends is not None and i < len(legends)) else f"Acc {['X', 'Y', 'Z'][i]}"
#             ax_acc.plot(x, X[:, i], linewidth=1, label=label_i, color=colors[i])
#         ax_acc.set_ylabel("Acceleration (g)")
#         ax_acc.set_title(title)
#         ax_acc.legend(fontsize=7, loc="upper right")
#         ax_acc.grid(True, alpha=0.3)
        
#         # Gyroscope subplot (last 3 channels)
#         ax_gyro = axes[1]
#         for i in range(3, min(6, V)):
#             label_i = legends[i] if (legends is not None and i < len(legends)) else f"Gyro {['X', 'Y', 'Z'][i-3]}"
#             ax_gyro.plot(x, X[:, i], linewidth=1, label=label_i, color=colors[i-3])
#         ax_gyro.set_ylabel("Angular Velocity (dps)")
#         ax_gyro.set_xlabel(xlabs)
#         ax_gyro.legend(fontsize=7, loc="upper right")
#         ax_gyro.grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig(save_path)
#         plt.close()
#         return save_path

#     plt.figure(figsize=(6, 4), dpi=100)

#     if method == "spectrogram":
#         for i in range(V):
#             label_i = legends[i] if (legends is not None and i < len(legends)) else f"Var {i}"
#             plt.subplot(V, 1, i + 1)
#             eps = 1e-10
#             plt.specgram(X[:, i] + eps, NFFT=64, Fs=1, noverlap=32)

#             plt.ylabel(label_i)
#             if i == 0:
#                 plt.title(title)
#         plt.xlabel(xlabs)
#     else:  # line plot
#         for i in range(V):
#             label_i = legends[i] if (legends is not None and i < len(legends)) else None
#             plt.plot(x, X[:, i], linewidth=1, label=label_i)

#         plt.xlabel(xlabs)
#         plt.ylabel(ylabs)
#         plt.title(title)

#     # Only show legend if user gave one
#     if legends is not None and any(legends):
#         plt.legend(fontsize=8, loc="upper right")

#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

#     return save_path

# MINIMALIST VERSION
def plot_time_series(
        X: np.ndarray, 
        method: str,
        title: str, # Kept in signature for compatibility, but ignored
        xlabs: str, 
        ylabs: str,
        legends: Optional[List[str]],
        save_path: str,
        recreate: bool = False,
    ) -> str:
    """
    Minimalist plotter: No titles, no grid, no top/right spines.
    """
    assert method in ["line", "spectrogram", "imu"], f"Unsupported method {method}"
    
    if os.path.exists(save_path) and not recreate:
        return save_path
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Normalize shape
    X = np.asarray(X).squeeze()
    if X.ndim == 1:
        X = X[:, None]
    elif X.ndim == 2:
        T, V = X.shape
        if T < V: X = X.T
    else:
        raise ValueError(f"Unsupported shape {X.shape}")

    T, V = X.shape
    x = np.arange(T)

    # Helper to clean axis
    def clean_axis(ax):
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        # We deliberately ignore the title argument here

    if method == "imu":
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=100, sharex=True)
        colors = ['#333333', '#777777', '#aaaaaa'] # Greyscale/Monochrome approach
        
        # Acc
        for i in range(min(3, V)):
            lbl = legends[i] if (legends and i < len(legends)) else None
            axes[0].plot(x, X[:, i], linewidth=1.2, label=lbl, color=colors[i])
        axes[0].set_ylabel("Acc (g)")
        clean_axis(axes[0])
        
        # Gyro
        for i in range(3, min(6, V)):
            lbl = legends[i] if (legends and i < len(legends)) else None
            axes[1].plot(x, X[:, i], linewidth=1.2, label=lbl, color=colors[i-3])
        axes[1].set_ylabel("Gyro (dps)")
        axes[1].set_xlabel(xlabs)
        clean_axis(axes[1])
        
        if legends: axes[0].legend(frameon=False, fontsize=8)

    elif method == "spectrogram":
        fig, axes = plt.subplots(V, 1, figsize=(6, 4 * V), dpi=100)
        if V == 1: axes = [axes] # Ensure iterable
        
        for i, ax in enumerate(axes):
            ax.specgram(X[:, i] + 1e-10, NFFT=64, Fs=1, noverlap=32, cmap='Greys')
            clean_axis(ax)
            ax.set_yticks([]) # Remove y ticks for cleaner look in spec
            if i < V - 1: ax.set_xticks([]) # Remove x ticks for all but bottom

        axes[-1].set_xlabel(xlabs)

    else: # Line
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        clean_axis(ax)
        
        for i in range(V):
            lbl = legends[i] if (legends and i < len(legends)) else None
            ax.plot(x, X[:, i], linewidth=1.5, label=lbl, alpha=0.9)

        ax.set_xlabel(xlabs)
        ax.set_ylabel(ylabs)
        if legends: ax.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight') # bbox_tight helps minimalist plots
    plt.close()
    return save_path

# DARK/HUD VERSION
# def plot_time_series(
#         X: np.ndarray, 
#         method: str,
#         title: str, 
#         xlabs: str, 
#         ylabs: str,
#         legends: Optional[List[str]],
#         save_path: str,
#         recreate: bool = False,
#     ) -> str:
#     """
#     Dark/HUD plotter: Black background, bright lines, no frames/spines.
#     """
#     assert method in ["line", "spectrogram", "imu"], f"Unsupported method {method}"
    
#     # Check existence
#     if os.path.exists(save_path) and not recreate:
#         return save_path
    
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     # Normalize input
#     X = np.asarray(X).squeeze()
#     if X.ndim == 1: X = X[:, None]
#     elif X.ndim == 2:
#         T, V = X.shape
#         if T < V: X = X.T
#     else: raise ValueError(f"Unsupported shape {X.shape}")

#     T, V = X.shape
#     x = np.arange(T)

#     # Style Configuration
#     text_color = '#aaaaaa'
#     bg_color = '#000000'
    
#     def apply_dark_theme(ax):
#         ax.set_facecolor(bg_color)
#         ax.grid(False)
#         ax.tick_params(colors=text_color, which='both')
#         for spine in ax.spines.values():
#             spine.set_visible(False) # Totally frameless
#         ax.xaxis.label.set_color(text_color)
#         ax.yaxis.label.set_color(text_color)

#     # --- IMU Plot ---
#     if method == "imu":
#         fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=100, sharex=True)
#         fig.patch.set_facecolor(bg_color)
        
#         # Neon colors hardcoded
#         colors = ['#00ffff', '#ff00ff', '#ffff00'] 
        
#         # Acc
#         for i in range(min(3, V)):
#             lbl = legends[i] if (legends and i < len(legends)) else None
#             axes[0].plot(x, X[:, i], linewidth=1, label=lbl, color=colors[i])
        
#         # Gyro
#         for i in range(3, min(6, V)):
#             lbl = legends[i] if (legends and i < len(legends)) else None
#             axes[1].plot(x, X[:, i], linewidth=1, label=lbl, color=colors[i-3])

#         apply_dark_theme(axes[0])
#         apply_dark_theme(axes[1])
        
#         if legends: 
#             axes[0].legend(frameon=False, labelcolor=text_color, fontsize=7)

#     # --- Spectrogram Plot ---
#     elif method == "spectrogram":
#         fig, axes = plt.subplots(V, 1, figsize=(6, 4 * V), dpi=100)
#         fig.patch.set_facecolor(bg_color)
#         if V == 1: axes = [axes] # type: ignore
        
#         for i, ax in enumerate(axes): # type: ignore
#             # 'inferno' looks good on dark backgrounds
#             ax.specgram(X[:, i] + 1e-10, NFFT=64, Fs=1, noverlap=32, cmap='inferno')
#             apply_dark_theme(ax)
#             ax.set_yticks([]) 
#             if i < V - 1: ax.set_xticks([])

#     # --- Line Plot ---
#     else:
#         fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
#         fig.patch.set_facecolor(bg_color)
#         apply_dark_theme(ax)
        
#         # FIX IS HERE: Get the colormap object explicitly
#         cmap = plt.get_cmap('cool') 
        
#         for i in range(V):
#             lbl = legends[i] if (legends and i < len(legends)) else None
#             # Calculate color
#             c = cmap(i / V) if V > 1 else '#ffffff'
#             ax.plot(x, X[:, i], linewidth=1.2, label=lbl, color=c)

#         ax.set_xlabel(xlabs)
        
#         if legends: 
#             ax.legend(frameon=False, labelcolor=text_color, fontsize=8)

#     plt.tight_layout()
#     plt.savefig(save_path, facecolor=bg_color)
#     plt.close()
#     return save_path

def save_ts_plot_as_pdf(
        X: np.ndarray, 
        method: str,
        title: str,
        xlabs: str, 
        ylabs: str,
        legends: Optional[List[str]],
        save_path: str,
        recreate: bool = False,
    ):
    """
    Simple time series plotter.
    - X can be (T,), (T,V), or (V,T)
    - xlabs, ylabs are direct axis-label strings
    - legends: list of names for each variable, or None for no legend
    - method can be "line", "spectrogram", or "imu" (for 6-channel IMU data)
    """
    assert method in ["line", "spectrogram", "imu"], f"Unsupported method {method}"
    assert ".pdf" in save_path
    # Skip if already exists
    if os.path.exists(save_path) and not recreate:
        return save_path
    # Ensure parent folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Normalize shape
    X = np.asarray(X).squeeze()

    if X.ndim == 1:
        X = X[:, None]              # (T,) → (T,1)
    elif X.ndim == 2:
        T, V = X.shape
        if T < V:                   # if transposed (V,T), fix it
            X = X.T
    else:
        raise ValueError(f"Unsupported shape {X.shape}")

    T, V = X.shape
    x = np.arange(T)

    if method == "imu":
        # Special handling for 6-channel IMU data (3 acc + 3 gyro)
        # Create two subplots: accelerometer and gyroscope
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=100, sharex=True)
        
        # Colors for x, y, z axes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
        
        # Accelerometer subplot (first 3 channels)
        ax_acc = axes[0]
        for i in range(min(3, V)):
            label_i = legends[i] if (legends is not None and i < len(legends)) else f"Acc {['X', 'Y', 'Z'][i]}"
            ax_acc.plot(x, X[:, i], linewidth=1, label=label_i, color=colors[i])
        ax_acc.set_ylabel("Acceleration (g)")
        ax_acc.set_title(title)
        ax_acc.legend(fontsize=7, loc="upper right")
        ax_acc.grid(True, alpha=0.3)
        
        # Gyroscope subplot (last 3 channels)
        ax_gyro = axes[1]
        for i in range(3, min(6, V)):
            label_i = legends[i] if (legends is not None and i < len(legends)) else f"Gyro {['X', 'Y', 'Z'][i-3]}"
            ax_gyro.plot(x, X[:, i], linewidth=1, label=label_i, color=colors[i-3])
        ax_gyro.set_ylabel("Angular Velocity (dps)")
        ax_gyro.set_xlabel(xlabs)
        ax_gyro.legend(fontsize=7, loc="upper right")
        ax_gyro.grid(True, alpha=0.3)
        
        plt.tight_layout()
        print(f"saved to {save_path}")

        plt.savefig(save_path)
        plt.close()
        return save_path

    plt.figure(figsize=(6, 4), dpi=100)

    if method == "spectrogram":
        for i in range(V):
            label_i = legends[i] if (legends is not None and i < len(legends)) else f"Var {i}"
            plt.subplot(V, 1, i + 1)
            eps = 1e-10
            plt.specgram(X[:, i] + eps, NFFT=64, Fs=1, noverlap=32)

            plt.ylabel(label_i)
            if i == 0:
                plt.title(title)
        plt.xlabel(xlabs)
    else:  # line plot
        for i in range(V):
            label_i = legends[i] if (legends is not None and i < len(legends)) else None
            plt.plot(x, X[:, i], linewidth=1, label=label_i)

        plt.xlabel(xlabs)
        plt.ylabel(ylabs)
        plt.title(title)

    # Only show legend if user gave one
    if legends is not None and any(legends):
        plt.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"saved to {save_path}")
    plt.close()


# def plot_time_series(
#         X: np.ndarray, 
#         method: str,
#         title: str,
#         xlabs: str, 
#         ylabs: str,
#         legends: Optional[List[str]],
#         save_path: str,
#         recreate: bool = False,
#     ) -> str:
#     """
#     Simple time series plotter.
#     - X can be (T,), (T,V), or (V,T)
#     - xlabs, ylabs are direct axis-label strings
#     - legends: list of names for each variable, or None for no legend
#     """
#     assert method in ["line", "spectrogram"], f"Unsupported method {method}"
#     # Skip if already exists
#     if os.path.exists(save_path) and not recreate:
#         return save_path
#     # Ensure parent folder exists
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     # Normalize shape
#     X = np.asarray(X).squeeze()

#     if X.ndim == 1:
#         X = X[:, None]              # (T,) → (T,1)
#     elif X.ndim == 2:
#         T, V = X.shape
#         if T < V:                   # if transposed (V,T), fix it
#             X = X.T
#     else:
#         raise ValueError(f"Unsupported shape {X.shape}")

#     T, V = X.shape
#     x = np.arange(T)

#     plt.figure(figsize=(6, 4), dpi=100)

#     if method == "spectrogram":
#         for i in range(V):
#             label_i = legends[i] if (legends is not None and i < len(legends)) else f"Var {i}"
#             plt.subplot(V, 1, i + 1)
#             eps = 1e-10
#             plt.specgram(X[:, i] + eps, NFFT=64, Fs=1, noverlap=32)

#             plt.ylabel(label_i)
#             if i == 0:
#                 plt.title(title)
#         plt.xlabel(xlabs)
#     else:  # line plot
#         for i in range(V):
#             label_i = legends[i] if (legends is not None and i < len(legends)) else None
#             plt.plot(x, X[:, i], linewidth=1, label=label_i)

#         plt.xlabel(xlabs)
#         plt.ylabel(ylabs)
#         plt.title(title)

#     # Only show legend if user gave one
#     if legends is not None and any(legends):
#         plt.legend(fontsize=8, loc="upper right")

#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

#     return save_path
