"""
Minimal mock objects that satisfy the QICK soc/soccfg interface.

Used exclusively in --debug mode.  Only the attributes and methods
actually called by the GUI are implemented; everything else is a no-op.
"""


class MockSoccfg:
    """
    Fake QickConfig — returns plausible board-info text and ignores all else.
    """

    def __str__(self) -> str:
        return (
            "QICK running on ZCU216 [SIMULATION MODE]\n"
            "  Global clocks (MHz): tProc 430.080, RF ref 245.760\n"
            "  16 signal generator channels\n"
            "  10 readout channels\n"
            "  tProc: qick_processor (v2) rev 21\n"
            "  DDR4 buffer: 1073741824 samples (3.495 sec)"
        )

    def __getattr__(self, name):
        return None


class MockSoc:
    """
    Fake QICK SoC proxy — all calls are silent no-ops.

    This is enough for ``BaseExperiment.setup(soc, soccfg, path)``
    which only stores the objects as class attributes.
    """

    def __getattr__(self, name):
        return lambda *a, **kw: None
