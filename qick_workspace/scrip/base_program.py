"""
BaseProgram: Base class for all QICK programs.
================================================
Centralizes generator declaration, readout, qubit pulse setup (prefix-aware),
standard gates, cooling, and measurement.
Subclasses only need to implement _initialize() and _body().
"""

from qick.asm_v2 import AveragerProgramV2


# ── Gate Name Mapping ──
# Maps shorthand gate names to pulse name templates.
# {pfx} is replaced by the transition prefix (e.g. "ge", "ef").
GATE_ALIAS = {
    # Clifford shorthand (used in AllXY / Tomography)
    "X": "x180_{pfx}",
    "Y": "y180_{pfx}",
    "X/2": "x90_{pfx}",
    "-X/2": "x90m_{pfx}",
    "Y/2": "y90_{pfx}",
    "-Y/2": "y90m_{pfx}",
    # Bare names without prefix
    "x180": "x180_{pfx}",
    "y180": "y180_{pfx}",
    "x90": "x90_{pfx}",
    "x90m": "x90m_{pfx}",
    "y90": "y90_{pfx}",
    "y90m": "y90m_{pfx}",
}


def resolve_gate(name, prefix="ge"):
    """
    Resolve a shorthand gate name to its fully-qualified pulse name.

    Parameters
    ----------
    name : str or None
        Gate shorthand (e.g. ``"X"``, ``"-X/2"``, ``"x90"``) or a
        fully-qualified pulse name (passed through unchanged).
        ``"I"``, ``"-I"``, ``None``, and ``"None"`` are returned as-is
        (identity gate — no pulse issued).
    prefix : str, optional
        Transition prefix to substitute for ``{pfx}`` in the template.
        Default is ``"ge"``.

    Returns
    -------
    pulse_name : str or None
        Fully-qualified pulse name (e.g. ``"x180_ge"``), the original
        identity sentinel, or the input unchanged if already qualified.

    Examples
    --------
    >>> resolve_gate("X")
    'x180_ge'
    >>> resolve_gate("-X/2")
    'x90m_ge'
    >>> resolve_gate("x90")
    'x90_ge'
    >>> resolve_gate("x180_ge")
    'x180_ge'
    >>> resolve_gate("I")
    'I'
    """
    if name in ("I", "-I", None, "None"):
        return name
    if name in GATE_ALIAS:
        return GATE_ALIAS[name].format(pfx=prefix)
    return name  # already fully qualified


class BaseProgram(AveragerProgramV2):
    """
    Base class for all QICK programs in this framework.

    Centralises generator declaration, readout configuration, qubit pulse
    setup (prefix-aware for ge/ef transitions), standard calibration gates,
    active cooling, and the measurement trigger sequence.  Subclasses only
    need to implement :meth:`_initialize` and :meth:`_body`.

    Key helpers
    -----------
    - :meth:`declare_gen_auto` — auto-handle mixer_freq for axis_sg_int4_v2
    - :meth:`setup_resonator` — readout channel + flat_top pulse config
    - :meth:`setup_qubit_gen` — qubit generator declaration (prefix-aware)
    - :meth:`setup_qb_pulse` — flexible pulse (const/arb/flat_top/drag)
    - :meth:`setup_standard_gates` — 6 calibration gates for AllXY/RB/Tomography
    - :meth:`apply_cool` — cooling channel + pulse config
    - :meth:`cooling_body` — cooling pulse sequence for :meth:`_body`
    - :meth:`measure` — readout pulse + trigger
    """

    # ── Generator helpers ──

    def declare_gen_auto(self, ch, nqz, mixer_key=None, cfg=None):
        """
        Declare a generator, automatically injecting ``mixer_freq`` when needed.

        For ``axis_sg_int4_v2`` generator tiles the QICK firmware requires an
        explicit mixer frequency.  This helper checks the tile type and calls
        ``declare_gen`` with or without ``mixer_freq`` accordingly.

        Parameters
        ----------
        ch : int
            Generator channel index.
        nqz : int
            Nyquist zone (1 or 2).
        mixer_key : str, optional
            Config key whose value gives the mixer frequency (MHz).  Only
            used when the tile type is ``axis_sg_int4_v2``.
        cfg : dict, optional
            Experiment configuration dict used to look up *mixer_key*.
        """
        if mixer_key and cfg and self.soccfg["gens"][ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(ch=ch, nqz=nqz, mixer_freq=cfg[mixer_key])
        else:
            self.declare_gen(ch=ch, nqz=nqz)

    def setup_qubit_gen(self, cfg, prefix="ge"):
        """
        Declare the qubit generator for a given transition prefix.

        Selects channel, NQZ, and mixer keys based on *prefix*:

        - ``"ge"`` → ``cfg["qb_ch"]`` / ``"nqz_qb"`` / ``"qb_mixer"``
        - other   → ``cfg["qb_ch_{prefix}"]`` / ``"nqz_qb_{prefix}"`` /
          ``"qb_mixer_{prefix}"``

        Parameters
        ----------
        cfg : dict
            Experiment configuration dict.
        prefix : str, optional
            Transition prefix.  Default is ``"ge"``.
        """
        if prefix == "ge":
            ch, nqz_key, mixer_key = cfg["qb_ch"], "nqz_qb", "qb_mixer"
        else:
            ch = cfg[f"qb_ch_{prefix}"]
            nqz_key = f"nqz_qb_{prefix}"
            mixer_key = f"qb_mixer_{prefix}"
        self.declare_gen_auto(ch, cfg[nqz_key], mixer_key, cfg)

    # ── Resonator setup ──

    def setup_resonator(
        self,
        cfg,
        prefix="ge",
    ):
        """
        Configure the resonator readout channel and flat_top readout pulse.

        Declares the readout generator and ADC channel, adds a Gaussian
        envelope, and registers a ``flat_top`` pulse named ``"res_pulse"``.

        Parameters
        ----------
        cfg : dict
            Experiment configuration dict.  Required keys: ``"ro_ch"``,
            ``"res_ch"``, ``"nqz_res"``, ``"ro_length"``,
            ``f"res_freq_{prefix}"``, ``"res_sigma"``, ``"res_length"``,
            ``"res_phase"``, ``f"res_gain_{prefix}"``.
        prefix : str, optional
            Transition prefix used to select frequency and gain.
            Default is ``"ge"``.
        """
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        self.declare_gen(ch=res_ch, nqz=cfg["nqz_res"])
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        self.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg[f"res_freq_{prefix}"], gen_ch=res_ch
        )
        self.add_gauss(
            ch=res_ch,
            name="readout",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
        self.add_pulse(
            ch=res_ch,
            name="res_pulse",
            ro_ch=ro_ch,
            style="flat_top",
            envelope="readout",
            length=cfg["res_length"],
            freq=cfg[f"res_freq_{prefix}"],
            phase=cfg["res_phase"],
            gain=cfg[f"res_gain_{prefix}"],
        )

    # ── Qubit pulse setup (flexible, prefix-aware) ──

    def setup_qb_pulse(
        self,
        cfg,
        prefix="ge",
        pulse_type=None,
        shape="gauss",
        name="qb_pulse",
        phase=None,
        gain_key=None,
        gain_override=None,
        ch=None,
        length_mult=5,
    ):
        """
        Configure a qubit pulse with automatic channel, gain, and phase resolution.

        Envelopes are deduplicated: multiple pulses sharing the same
        ``(channel, prefix, shape)`` combination reuse a single registered
        envelope to avoid redundant ``add_*`` calls.

        Parameters
        ----------
        cfg : dict
            Experiment configuration dict.
        prefix : str, optional
            Transition prefix (``"ge"`` or ``"ef"``).  Selects channel,
            frequency, sigma, and gain keys.  Default is ``"ge"``.
        pulse_type : str, optional
            Pulse style: ``"const"``, ``"arb"``, ``"flat_top"``, or
            ``"drag"``.  When ``None``, falls back to
            ``cfg.get("pulse_type", "arb")``.  The ``"drag"`` type forces
            *shape* to ``"drag"`` and uses ``"arb"`` style internally.
        shape : str, optional
            Envelope shape for ``"arb"`` / ``"flat_top"`` pulses: ``"gauss"``,
            ``"cosine"``, or ``"drag"``.  Ignored when *pulse_type* is
            ``"drag"`` (which forces ``shape="drag"``).  Default is
            ``"gauss"``.
        name : str, optional
            Name to register the pulse under.  Default is ``"qb_pulse"``.
        phase : float, optional
            Pulse phase in degrees.  Defaults to ``cfg["qb_phase"]``.
        gain_key : str, optional
            Explicit config key for the gain value.  Defaults to
            ``f"qb_gain_{prefix}"``.
        gain_override : float or int, optional
            Explicit gain value.  Takes precedence over *gain_key* when
            provided.
        ch : int, optional
            Explicit channel override.  Defaults to ``cfg["qb_ch"]`` (ge) or
            ``cfg[f"qb_ch_{prefix}"]`` (other prefixes).
        length_mult : int, optional
            Gaussian / DRAG envelope length multiplier: length = sigma *
            length_mult.  Cosine envelopes always use length = sigma.
            Default is ``5``.

        Raises
        ------
        KeyError
            If ``"drag_alpha"`` is missing from *cfg* when *pulse_type* is
            ``"drag"``.
        ValueError
            If *shape* is not one of ``"gauss"``, ``"cosine"``, or
            ``"drag"``.
        """
        # Resolve channel
        if ch is None:
            ch = cfg["qb_ch"] if prefix == "ge" else cfg[f"qb_ch_{prefix}"]

        # Resolve pulse type from config if not specified
        if pulse_type is None:
            pulse_type = cfg.get("pulse_type", "arb")

        # 'drag' pulse_type forces DRAG envelope; treat it as 'arb' for the add_pulse stage
        if pulse_type == "drag":
            shape = "drag"

        # Resolve phase from config if not specified
        if phase is None:
            phase = cfg["qb_phase"]

        # Resolve gain: gain_override takes precedence over gain_key
        if gain_override is not None:
            gain = gain_override
        elif gain_key:
            gain = cfg[gain_key]
        else:
            gain = cfg[f"qb_gain_{prefix}"]

        # Resolve frequency
        freq = cfg[f"qb_freq_{prefix}"]

        # Add envelope if needed (deduplicated per channel+prefix+shape)
        env_name = None
        if pulse_type in ("arb", "flat_top", "drag"):
            env_name = f"env_{prefix}_{shape}"
            if not hasattr(self, "_added_envs"):
                self._added_envs = set()
            env_key = (ch, env_name)
            if env_key not in self._added_envs:
                sigma = cfg[f"sigma_{prefix}"]
                if shape in ("gauss", "gaussian"):
                    self.add_gauss(
                        ch=ch,
                        name=env_name,
                        sigma=sigma,
                        length=sigma * length_mult,
                        even_length=True,
                    )
                elif shape in ("cos", "cosine"):
                    self.add_cosine(
                        ch=ch,
                        name=env_name,
                        length=sigma,
                        even_length=True,
                    )
                elif shape == "drag":
                    delta = cfg["qb_freq_ge"] - cfg["qb_freq_ef"]
                    if "drag_alpha" not in cfg:
                        raise KeyError(
                            "no parameter 'drag_alpha' found in cfg — calibrate DRAG first (run s005a_drag.py)"
                        )
                    alpha = cfg["drag_alpha"]
                    self.add_DRAG(
                        ch=ch,
                        name=env_name,
                        sigma=sigma,
                        length=sigma * length_mult,
                        delta=delta,
                        alpha=alpha,
                        even_length=True,
                    )
                else:
                    raise ValueError(f"Unknown pulse shape: {shape}")
                self._added_envs.add(env_key)

        # Add pulse
        if pulse_type == "const":
            self.add_pulse(
                ch=ch,
                name=name,
                style="const",
                length=cfg[f"qb_length_{prefix}"],
                freq=freq,
                phase=phase,
                gain=gain,
            )
        elif pulse_type in ("arb", "drag"):  # drag envelope already built above
            self.add_pulse(
                ch=ch,
                name=name,
                style="arb",
                envelope=env_name,
                freq=freq,
                phase=phase,
                gain=gain,
            )
        elif pulse_type == "flat_top":
            self.add_pulse(
                ch=ch,
                name=name,
                style="flat_top",
                envelope=env_name,
                freq=freq,
                phase=phase,
                gain=gain,
                length=cfg[f"qb_flat_top_length_{prefix}"],
            )

    def setup_standard_gates(self, cfg, prefix="ge", pulse_type=None, shape="gauss"):
        """
        Register the six standard calibration gates for AllXY, RB, and tomography.

        Gates registered: X180, Y180, X90, X90m (−X90), Y90, Y90m (−Y90).
        Each gate is added via :meth:`setup_qb_pulse` with the appropriate
        phase and gain key.

        Parameters
        ----------
        cfg : dict
            Experiment configuration dict.
        prefix : str, optional
            Transition prefix.  Default is ``"ge"``.
        pulse_type : str, optional
            Forwarded to :meth:`setup_qb_pulse`.  ``None`` uses the config
            default.
        shape : str, optional
            Envelope shape forwarded to :meth:`setup_qb_pulse`.
            Default is ``"gauss"``.
        """
        gates = [
            (f"x180_{prefix}", 0, f"pi_gain_{prefix}"),
            (f"y180_{prefix}", 90, f"pi_gain_{prefix}"),
            (f"x90_{prefix}", 0, f"pi2_gain_{prefix}"),
            (f"x90m_{prefix}", 180, f"pi2_gain_{prefix}"),
            (f"y90_{prefix}", 90, f"pi2_gain_{prefix}"),
            (f"y90m_{prefix}", -90, f"pi2_gain_{prefix}"),
        ]
        for gate_name, gate_phase, gain_key in gates:
            self.setup_qb_pulse(
                cfg,
                prefix=prefix,
                pulse_type=pulse_type,
                shape=shape,
                name=gate_name,
                phase=gate_phase,
                gain_key=gain_key,
            )

    # ── Cooling ──

    def apply_cool(self, cfg, style="flat_top"):
        """
        Configure active-reset cooling channels and pulses.

        Iterates over ``cool_ch1`` and ``cool_ch2`` in *cfg* and declares
        each channel's generator and pulse.

        Parameters
        ----------
        cfg : dict
            Experiment configuration dict.  Required keys per channel:
            ``f"cool_ch{i}"``, ``f"nqz_cool_ch{i}"`` (optional, default 2),
            ``f"cool_mixer{i}"``, ``"res_sigma"``, ``"cool_length"``,
            ``f"cool_freq_{i}"``, ``f"cool_gain_{i}"``.
        style : str, optional
            Pulse style: ``"flat_top"`` (default, most experiments) or
            ``"const"`` (legacy).
        """
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
                self.add_gauss(
                    ch=ch,
                    name=env_name,
                    sigma=cfg["res_sigma"],
                    length=cfg["res_sigma"] * 5,
                    even_length=True,
                )
                self.add_pulse(
                    ch=ch,
                    name=f"cool_pulse{i}",
                    envelope=env_name,
                    style="flat_top",
                    length=cfg["cool_length"],
                    freq=cfg[f"cool_freq_{i}"],
                    phase=0,
                    gain=cfg[f"cool_gain_{i}"],
                )
            else:  # const
                self.add_pulse(
                    ch=ch,
                    name=f"cool_pulse{i}",
                    style="const",
                    length=cfg["cool_length"],
                    freq=cfg[f"cool_freq_{i}"],
                    phase=0,
                    gain=cfg[f"cool_gain_{i}"],
                )

    def cooling_body(self, cfg, ring_down=0.5):
        """
        Execute the active-reset cooling pulse sequence inside ``_body``.

        Parameters
        ----------
        cfg : dict
            Experiment configuration dict.  Must contain ``"cooling"`` (bool)
            and ``"cool_ch1"`` / ``"cool_ch2"`` keys.
        ring_down : float, optional
            Delay in microseconds after the cooling pulses to allow cavity
            ring-down.  Default is ``0.5`` µs.

        Returns
        -------
        ran : bool
            ``True`` if cooling was executed, ``False`` if
            ``cfg["cooling"]`` is falsy.
        """
        if not cfg.get("cooling", False):
            return False
        self.pulse(ch=cfg["cool_ch1"], name="cool_pulse1", t=0)
        self.pulse(ch=cfg["cool_ch2"], name="cool_pulse2", t=0)
        self.delay_auto(ring_down, tag="Ring down")
        return True

    # ── Measurement ──

    def measure(self, cfg):
        """
        Execute the standard readout pulse and ADC trigger.

        Parameters
        ----------
        cfg : dict
            Experiment configuration dict.  Required keys: ``"res_ch"``,
            ``"ro_ch"``, ``"trig_time"``.
        """
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])
