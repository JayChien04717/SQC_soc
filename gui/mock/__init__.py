"""Simulation layer — replaces real QICK hardware in --debug mode."""
from .hardware import MockSoc, MockSoccfg
from .runner  import generate_mock_data
