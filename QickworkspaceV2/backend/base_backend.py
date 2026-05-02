"""
BaseBackend — abstract hardware abstraction layer.

Decouples experiment logic from QICK-specific objects (soc / soccfg).
Concrete implementations: QICKBackend (real hardware), SimulatedBackend (mock).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseBackend(ABC):
    """
    Abstract backend interface.

    Attributes
    ----------
    soc : object
        Hardware SoC proxy (QICK ``QickSoc`` or mock).
    soccfg : object
        Hardware configuration (QICK ``QickConfig`` or mock).
    name : str
        Human-readable backend name (e.g. ``"QICK@192.168.1.1"``).
    """

    def __init__(self, soc, soccfg, name: str = ""):
        self.soc = soc
        self.soccfg = soccfg
        self.name = name or self.__class__.__name__

    @abstractmethod
    def run_program(self, prog, rounds: int = 1) -> Any:
        """
        Execute a compiled QICK program and return raw IQ data.

        Parameters
        ----------
        prog : AveragerProgramV2
            Compiled QICK program.
        rounds : int
            Number of hardware averaging rounds.

        Returns
        -------
        iq_list : list
            Raw IQ data as returned by ``prog.acquire()``.
        """
        ...

    def is_simulated(self) -> bool:
        """Return ``True`` if this is a simulated (mock) backend."""
        return False

    def __repr__(self) -> str:
        sim = " [simulated]" if self.is_simulated() else ""
        return f"{self.__class__.__name__}({self.name!r}{sim})"
