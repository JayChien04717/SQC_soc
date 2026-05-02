"""
BaseProgram: Base class for all QICK programs.

Centralizes generator declaration, readout, qubit pulse setup (prefix-aware),
standard gates, cooling, and measurement.
Subclasses only need to implement _initialize() and _body().
"""

from qick.asm_v2 import AveragerProgramV2


GATE_ALIAS = {
    "X":    "x180_{pfx}",
    "Y":    "y180_{pfx}",
    "X/2":  "x90_{pfx}",
    "-X/2": "x90m_{pfx}",
    "Y/2":  "y90_{pfx}",
    "-Y/2": "y90m_{pfx}",
    "x180": "x180_{pfx}",
    "y180": "y180_{pfx}",
    "x90":  "x90_{pfx}",
    "x90m": "x90m_{pfx}",
    "y90":  "y90_{pfx}",
    "y90m": "y90m_{pfx}",
}


def resolve_gate(name, prefix="ge"):
    """
    Resolve a shorthand gate name to its fully-qualified pulse name.

    "I", "-I", None, and "None" are returned as-is (identity gate).
    """
    if name in ("I", "-I", None, "None"):
        return name
    if name in GATE_ALIAS:
        return GATE_ALIAS[name].format(pfx=prefix)
    return name


class BaseProgram(AveragerProgramV2):
    """
    Base class for all QICK programs in this framework.

    Centralises generator declaration, readout configuration, qubit pulse
    setup (prefix-aware for ge/ef transitions), standard calibration gates,
    active cooling, and the measurement trigger sequence.  Subclasses only
    need to implement :meth:`_initialize` and :meth:`_body`.
    """

    def declare_gen_auto(self, ch, nqz, mixer_key=None, cfg=None):
        """Declare a generator, automatically injecting mixer_freq when needed."""
        if mixer_key and cfg and self.soccfg["gens"][ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(ch=ch, nqz=nqz, mixer_freq=cfg[mixer_key])
        else:
            self.declare_gen(ch=ch, nqz=nqz)

    def setup_qubit_gen(self, cfg, prefix="ge"):
        """Declare the qubit generator for a given transition prefix."""
        if prefix == "ge":
            ch, nqz_key, mixer_key = cfg["qb_ch"], "nqz_qb", "qb_mixer"
        else:
            ch = cfg[f"qb_ch_{prefix}"]
            nqz_key = f"nqz_qb_{prefix}"
            mixer_key = f"qb_mixer_{prefix}"
        self.declare_gen_auto(ch, cfg[nqz_key], mixer_key, cfg)

    def setup_resonator(self, cfg, prefix="ge"):
        """Configure the resonator readout channel and flat_top readout pulse."""
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_readoutconfig(ch=ro_ch, name="myro", freq=cfg[f"res_freq_{prefix}"], gen_ch=res_ch)
        self.add_gauss(ch=res_ch, name="readout", sigma=cfg["res_sigma"],
                       length=5 * cfg["res_sigma"], even_length=True)
        self.add_pulse(
            ch=res_ch, name="res_pulse", ro_ch=ro_ch, style="flat_top",
            envelope="readout", length=cfg["res_length"],
            freq=cfg[f"res_freq_{prefix}"], phase=cfg["res_phase"], gain=cfg[f"res_gain_{prefix}"],
        )

    def setup_qb_pulse(self, cfg, prefix="ge", pulse_type=None, shape="gauss", name="qb_pulse",
                       phase=None, gain_key=None, gain_override=None, ch=None, length_mult=5):
        """Configure a qubit pulse with automatic channel, gain, and phase resolution."""
        if ch is None:
            ch = cfg["qb_ch"] if prefix == "ge" else cfg[f"qb_ch_{prefix}"]
        if pulse_type is None:
            pulse_type = cfg.get("pulse_type", "arb")
        if pulse_type == "drag":
            shape = "drag"
        if phase is None:
            phase = cfg["qb_phase"]
        if gain_override is not None:
            gain = gain_override
        elif gain_key:
            gain = cfg[gain_key]
        else:
            gain = cfg[f"qb_gain_{prefix}"]
        freq = cfg[f"qb_freq_{prefix}"]

        env_name = None
        if pulse_type in ("arb", "flat_top", "drag"):
            env_name = f"env_{prefix}_{shape}"
            if not hasattr(self, "_added_envs"):
                self._added_envs = set()
            env_key = (ch, env_name)
            if env_key not in self._added_envs:
                sigma = cfg[f"sigma_{prefix}"]
                if shape in ("gauss", "gaussian"):
                    self.add_gauss(ch=ch, name=env_name, sigma=sigma, length=sigma * length_mult, even_length=True)
                elif shape in ("cos", "cosine"):
                    self.add_cosine(ch=ch, name=env_name, length=sigma, even_length=True)
                elif shape == "drag":
                    delta = cfg["qb_freq_ge"] - cfg["qb_freq_ef"]
                    if "drag_alpha" not in cfg:
                        raise KeyError("no parameter 'drag_alpha' found in cfg — calibrate DRAG first")
                    alpha = cfg["drag_alpha"]
                    self.add_DRAG(ch=ch, name=env_name, sigma=sigma, length=sigma * length_mult,
                                  delta=delta, alpha=alpha, even_length=True)
                else:
                    raise ValueError(f"Unknown pulse shape: {shape}")
                self._added_envs.add(env_key)

        if pulse_type == "const":
            self.add_pulse(ch=ch, name=name, style="const", length=cfg[f"qb_length_{prefix}"],
                           freq=freq, phase=phase, gain=gain)
        elif pulse_type in ("arb", "drag"):
            self.add_pulse(ch=ch, name=name, style="arb", envelope=env_name, freq=freq, phase=phase, gain=gain)
        elif pulse_type == "flat_top":
            self.add_pulse(ch=ch, name=name, style="flat_top", envelope=env_name, freq=freq, phase=phase, gain=gain,
                           length=cfg[f"qb_flat_top_length_{prefix}"])

    def setup_standard_gates(self, cfg, prefix="ge", pulse_type=None, shape="gauss"):
        """Register the six standard calibration gates for AllXY, RB, and tomography."""
        gates = [
            (f"x180_{prefix}", 0, f"pi_gain_{prefix}"),
            (f"y180_{prefix}", 90, f"pi_gain_{prefix}"),
            (f"x90_{prefix}", 0, f"pi2_gain_{prefix}"),
            (f"x90m_{prefix}", 180, f"pi2_gain_{prefix}"),
            (f"y90_{prefix}", 90, f"pi2_gain_{prefix}"),
            (f"y90m_{prefix}", -90, f"pi2_gain_{prefix}"),
        ]
        for gate_name, gate_phase, gain_key in gates:
            self.setup_qb_pulse(cfg, prefix=prefix, pulse_type=pulse_type, shape=shape,
                                name=gate_name, phase=gate_phase, gain_key=gain_key)

    def apply_cool(self, cfg, style="flat_top"):
        """Configure active-reset cooling channels and pulses."""
        for i in [1, 2]:
            ch_key = f"cool_ch{i}"
            if ch_key not in cfg:
                continue
            ch = cfg[ch_key]
            nqz = cfg.get(f"nqz_cool_ch{i}", 2)
            mixer_key = f"cool_mixer{i}"
            self.declare_gen_auto(ch, nqz, mixer_key, cfg)

            if style == "flat_top":
                env_name = f"cooling{i}"
                self.add_gauss(ch=ch, name=env_name, sigma=cfg["res_sigma"],
                               length=cfg["res_sigma"] * 5, even_length=True)
                self.add_pulse(ch=ch, name=f"cool_pulse{i}", envelope=env_name, style="flat_top",
                               length=cfg["cool_length"], freq=cfg[f"cool_freq_{i}"], phase=0, gain=cfg[f"cool_gain_{i}"])
            else:
                self.add_pulse(ch=ch, name=f"cool_pulse{i}", style="const",
                               length=cfg["cool_length"], freq=cfg[f"cool_freq_{i}"], phase=0, gain=cfg[f"cool_gain_{i}"])

    def cooling_body(self, cfg, ring_down=0.5):
        """Execute the active-reset cooling pulse sequence inside _body."""
        if not cfg.get("cooling", False):
            return False
        self.pulse(ch=cfg["cool_ch1"], name="cool_pulse1", t=0)
        self.pulse(ch=cfg["cool_ch2"], name="cool_pulse2", t=0)
        self.delay_auto(ring_down, tag="Ring down")
        return True

    def measure(self, cfg):
        """Execute the standard readout pulse and ADC trigger."""
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


__all__ = ["BaseProgram", "GATE_ALIAS", "resolve_gate"]
