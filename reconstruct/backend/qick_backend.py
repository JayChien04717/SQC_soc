"""
QICKBackend — real hardware backend wrapping a live QICK connection.

Usage
-----
::

    import Pyro4
    soc, soccfg = ... # your QICK connection setup
    backend = QICKBackend(soc, soccfg, name="QICK@192.168.1.1")

    # Inject into BaseExperiment
    expt = T1(cfg, backend=backend)
    result = expt.run(py_avg=50)

    # Or use legacy class-level setup (backward compat)
    from reconstruct.core import BaseExperiment
    BaseExperiment.setup(soc, soccfg, data_path)
"""

from __future__ import annotations

from typing import Any

from .base_backend import BaseBackend


class QICKBackend(BaseBackend):
    """
    Hardware backend for a live QICK SoC connection.

    Parameters
    ----------
    soc : QickSoc
        QICK SoC proxy (Pyro4 remote object or local).
    soccfg : QickConfig
        QICK hardware configuration.
    name : str, optional
        Human-readable label (e.g. IP address or board ID).
    data_path : str, optional
        Default data save directory.  Passed to ``BaseExperiment._data_path``
        when :meth:`activate` is called.
    """

    def __init__(self, soc, soccfg, name: str = "", data_path: str = ""):
        super().__init__(soc, soccfg, name)
        self.data_path = data_path

    def run_program(self, prog, rounds: int = 1) -> Any:
        """Execute program on real hardware."""
        return prog.acquire(self.soc, rounds=rounds, progress=False)

    def activate(self):
        """
        Set this backend as the global session for legacy ``BaseExperiment``
        API.  Equivalent to ``BaseExperiment.setup(soc, soccfg, data_path)``.
        """
        from reconstruct.core.base_experiment import BaseExperiment

        BaseExperiment.setup(self.soc, self.soccfg, self.data_path)
        print(
            f"[QICKBackend] Session activated: {self.name or 'QICK'}, "
            f"data_path={self.data_path!r}"
        )

    @classmethod
    def from_pyro4(
        cls,
        ns_host: str,
        soc_name: str = "myqick",
        data_path: str = "",
    ) -> "QICKBackend":
        """
        Connect to a remote QICK server via Pyro4 Name Server.

        Parameters
        ----------
        ns_host : str
            IP address of the Pyro4 Name Server (same as QICK board IP).
        soc_name : str, optional
            Registered object name.  Default is ``"myqick"``.
        data_path : str, optional
            Data save directory.
        """
        try:
            import Pyro4
            from qick import QickConfig
        except ImportError as exc:
            raise ImportError(
                "Pyro4 and qick must be installed to use QICKBackend.from_pyro4()"
            ) from exc

        Pyro4.config.SERIALIZER = "pickle"
        Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

        ns = Pyro4.locateNS(host=ns_host)
        soc = Pyro4.Proxy(ns.lookup(soc_name))
        soccfg = QickConfig(soc.get_cfg())

        return cls(soc, soccfg, name=f"QICK@{ns_host}", data_path=data_path)
