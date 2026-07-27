"""
pytest configuration to suppress matplotlib backend warnings.
"""

import warnings

warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")