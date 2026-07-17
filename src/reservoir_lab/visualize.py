import numpy as np
import matplotlib.pyplot as plt


def plot_results(inputs, targets, pred, states, title="Experiment results",
                  window=200, n_neurons=5, save_path=None):
    """
    Standard 3-panel visualization for any reservoir computing experiment.

    Parameters
    ----------
    inputs : array (n_timesteps, n_inputs)   -- what the reservoir was fed
    targets : array (n_timesteps, n_outputs) -- ground-truth values to predict
    pred : array (n_timesteps, n_outputs)    -- the model's predictions
    states : array (n_timesteps, n_reservoir) -- internal reservoir activity,
             as returned by ESN.fit() or ESN.run()
    title : str          -- figure title, e.g. "spectral_radius=1.2"
    window : int         -- how many timesteps to show (data is often too
             dense to read if you plot all of it)
    n_neurons : int      -- how many individual reservoir neurons to sample
             in the bottom panel (plotting all of them is unreadable)
    save_path : str or None -- if given, saves the figure to this path;
             otherwise just displays it

    Returns
    -------
    mse : float -- mean squared error, printed and returned for convenience
    """
    w = slice(0, min(window, len(targets)))
    mse = np.mean((pred - targets) ** 2)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f"{title}  (MSE={mse:.6f})", y=0.98)

    axes[0].plot(targets[w], label="target", linewidth=2)
    axes[0].plot(pred[w], label="prediction", linestyle="--")
    axes[0].set_title("Prediction vs target")
    axes[0].legend()

    axes[1].plot(inputs[w], color="gray")
    axes[1].set_title("Input signal")

    n = min(n_neurons, states.shape[1])
    axes[2].plot(states[w, :n])
    axes[2].set_title(f"Sample reservoir neuron activity ({n} of {states.shape[1]} neurons)")
    axes[2].set_xlabel("timestep")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    print(f"MSE: {mse:.6f}")
    return mse
