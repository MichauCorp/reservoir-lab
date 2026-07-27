"""
Tests for visualization utilities (visualize.py).
"""

import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pytest

from reservoir_lab.visualize import plot_results

# Suppress the non-interactive backend warning for plt.show()
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")


class TestPlotResults:
    def test_basic_plot_returns_mse(self):
        """plot_results should return the mean squared error."""
        np.random.seed(42)
        targets = np.random.randn(100, 1)
        pred = targets + 0.1 * np.random.randn(100, 1)
        inputs = np.random.randn(100, 2)
        states = np.random.randn(100, 50)

        mse = plot_results(inputs, targets, pred, states, title="Test", save_path=None)
        
        expected_mse = np.mean((pred - targets) ** 2)
        assert np.isclose(mse, expected_mse)

    def test_plot_with_save_path(self, tmp_path):
        """When save_path is given, the figure should be saved."""
        targets = np.random.randn(50, 1)
        pred = targets + 0.1 * np.random.randn(50, 1)
        inputs = np.random.randn(50, 2)
        states = np.random.randn(50, 20)

        save_path = tmp_path / "test_plot.png"
        mse = plot_results(inputs, targets, pred, states, title="Save test", 
                           save_path=str(save_path))
        
        assert save_path.exists()
        assert np.isclose(mse, np.mean((pred - targets) ** 2))

    def test_window_parameter(self):
        """The window parameter should limit how many timesteps are plotted."""
        np.random.seed(42)
        targets = np.random.randn(500, 1)
        pred = np.random.randn(500, 1)
        inputs = np.random.randn(500, 2)
        states = np.random.randn(500, 30)

        # Should work with window=100
        mse = plot_results(inputs, targets, pred, states, title="Window test", 
                           window=100, save_path=None)
        assert np.isfinite(mse)

    def test_window_larger_than_data(self):
        """Window larger than data length should not fail."""
        np.random.seed(42)
        targets = np.random.randn(50, 1)
        pred = np.random.randn(50, 1)
        inputs = np.random.randn(50, 2)
        states = np.random.randn(50, 20)

        mse = plot_results(inputs, targets, pred, states, title="Large window", 
                           window=1000, save_path=None)
        assert np.isfinite(mse)

    def test_n_neurons_parameter(self):
        """n_neurons should control how many neurons are sampled in the plot."""
        np.random.seed(42)
        targets = np.random.randn(100, 1)
        pred = np.random.randn(100, 1)
        inputs = np.random.randn(100, 2)
        states = np.random.randn(100, 100)

        mse = plot_results(inputs, targets, pred, states, title="Neurons test", 
                           n_neurons=10, save_path=None)
        assert np.isfinite(mse)

    def test_multi_output_targets(self):
        """Should work with multi-output targets."""
        np.random.seed(42)
        targets = np.random.randn(100, 3)
        pred = np.random.randn(100, 3)
        inputs = np.random.randn(100, 2)
        states = np.random.randn(100, 50)

        mse = plot_results(inputs, targets, pred, states, title="Multi-output", 
                           save_path=None)
        expected_mse = np.mean((pred - targets) ** 2)
        assert np.isclose(mse, expected_mse)

    def test_mse_computation(self):
        """MSE should be computed correctly across all outputs."""
        np.random.seed(42)
        targets = np.random.randn(100, 2)
        pred = targets + 0.5  # constant offset
        inputs = np.random.randn(100, 3)
        states = np.random.randn(100, 40)

        mse = plot_results(inputs, targets, pred, states, title="MSE test", 
                           save_path=None)
        # pred - targets = 0.5, so (0.5)^2 = 0.25 per element
        # Mean across all elements should be 0.25
        assert np.isclose(mse, 0.25, atol=1e-6)

    def test_title_displayed(self):
        """The title should be included in the figure."""
        np.random.seed(42)
        targets = np.random.randn(50, 1)
        pred = np.random.randn(50, 1)
        inputs = np.random.randn(50, 2)
        states = np.random.randn(50, 20)

        fig = plt.figure()
        original_manager = plt.get_current_fig_manager()
        
        try:
            plot_results(inputs, targets, pred, states, title="CustomTitle", 
                        window=50, save_path=None)
            fig = plt.gcf()
            # Check that the suptitle contains our custom title
            suptitle = fig._suptitle.get_text()
            assert "CustomTitle" in suptitle
        finally:
            plt.close('all')

    def test_finite_output_with_extreme_values(self):
        """Even with extreme input values, output should be finite."""
        np.random.seed(42)
        targets = np.random.randn(100, 1)
        pred = np.random.randn(100, 1) * 100  # large but finite
        inputs = np.random.randn(100, 2) * 100
        states = np.random.randn(100, 50) * 100

        mse = plot_results(inputs, targets, pred, states, title="Extreme values", 
                           save_path=None)
        assert np.isfinite(mse)

    def test_single_timestep(self):
        """Should handle a single timestep gracefully."""
        targets = np.random.randn(1, 1)
        pred = np.random.randn(1, 1)
        inputs = np.random.randn(1, 2)
        states = np.random.randn(1, 10)

        mse = plot_results(inputs, targets, pred, states, title="Single step", 
                           save_path=None)
        assert np.isfinite(mse)

    def test_single_neuron_plot(self):
        """Should work when n_neurons=1."""
        np.random.seed(42)
        targets = np.random.randn(100, 1)
        pred = np.random.randn(100, 1)
        inputs = np.random.randn(100, 2)
        states = np.random.randn(100, 50)

        mse = plot_results(inputs, targets, pred, states, title="One neuron", 
                           n_neurons=1, save_path=None)
        assert np.isfinite(mse)

    def test_no_nans_in_output(self):
        """MSE should never be NaN if inputs are finite."""
        np.random.seed(42)
        targets = np.random.randn(100, 1)
        pred = np.random.randn(100, 1)
        inputs = np.random.randn(100, 2)
        states = np.random.randn(100, 50)

        mse = plot_results(inputs, targets, pred, states, title="No NaNs", 
                           save_path=None)
        assert not np.isnan(mse)

    def test_save_path_creates_file(self, tmp_path):
        """Saving should create a PNG file at the specified path."""
        np.random.seed(42)
        targets = np.random.randn(80, 1)
        pred = np.random.randn(80, 1)
        inputs = np.random.randn(80, 2)
        states = np.random.randn(80, 30)

        save_path = tmp_path / "output" / "plot.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        mse = plot_results(inputs, targets, pred, states, title="Save test", 
                           save_path=str(save_path))
        
        assert save_path.exists()
        assert save_path.stat().st_size > 0  # file is non-empty
