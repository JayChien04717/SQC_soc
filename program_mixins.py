from qick.asm_v2 import AveragerProgramV2

class QickProgramMixin:
    """
    mixin class providing common setup methods for Qick programs.
    """
    def setup_readout(self, cfg, prefix='ge'):
        """Standard setup for resonator readout."""
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]
        
        # Declare Generator for Resonator
        self.declare_gen(ch=res_ch, nqz=cfg.get("nqz_res", 2))
        
        # Declare Readout
        self.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        
        # Add Readout Config
        self.add_readoutconfig(
            ch=ro_ch, 
            name=f"myro", 
            freq=cfg[f"res_freq_{prefix}"], 
            gen_ch=res_ch
        )
        
        # Add standard Gaussian Readout Pulse
        self.add_gauss(
            ch=res_ch,
            name="readout",
            sigma=cfg["res_sigma"],
            length=5 * cfg["res_sigma"],
            even_length=True,
        )
    def measure(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.pulse(ch=cfg["res_ch"], name="readout", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])

    def pulse_gate(self, name, cfg, t=0):
        """Helper to pulse a gate by name on the qubit channel."""
        self.pulse(ch=cfg["qb_ch"], name=name, t=t)

    def setup_qubit_gen(self, cfg):
        """Standard setup for qubit generator."""
        qb_ch = cfg["qb_ch"]
        if self.soccfg["gens"][qb_ch]["type"] == "axis_sg_int4_v2":
            self.declare_gen(ch=qb_ch, nqz=cfg.get("nqz_qb", 2), mixer_freq=cfg.get("qb_mixer", 0))
        else:
            self.declare_gen(ch=qb_ch, nqz=cfg.get("nqz_qb", 2))
    
    def setup_qb_pulse(self, cfg, prefix='ge', pulse_type='const', shape='gauss', name='qb_pulse', phase=0, gain_key=None, ch=None):
        """Standard setup for a qubit pulse. gain_key can override default qb_gain_{prefix}. ch can override cfg['qb_ch']."""
        qb_ch = ch if ch is not None else cfg["qb_ch"]
        
        # 1. Add Envelope if needed (shared between pulses of same prefix/shape)
        env_name = None
        if pulse_type in ['arb', 'flat_top']:
            env_name = f"env_{prefix}_{shape}"
            if shape in ['gauss', 'gaussian']:
                self.add_gauss(ch=qb_ch, name=env_name, sigma=cfg[f"sigma_{prefix}"], length=cfg[f"sigma_{prefix}"] * 4, even_length=True)
            elif shape in ['cos', 'cosine']:
                self.add_cosine(ch=qb_ch, name=env_name, length=cfg[f"sigma_{prefix}"] * 4, even_length=True)
            else:
                raise ValueError(f"Unknown pulse shape: {shape}")

        # 2. Determine Gain
        actual_gain = cfg[gain_key] if gain_key else cfg[f"qb_gain_{prefix}"]

        # 3. Add Pulse
        if pulse_type == 'const':
            self.add_pulse(ch=qb_ch, name=name, style="const", length=cfg[f"qb_length_{prefix}"], freq=cfg[f"qb_freq_{prefix}"], phase=phase, gain=actual_gain)
        elif pulse_type == 'arb':
            self.add_pulse(ch=qb_ch, name=name, style="arb", envelope=env_name, freq=cfg[f"qb_freq_{prefix}"], phase=phase, gain=actual_gain)
        elif pulse_type == 'flat_top':
            self.add_pulse(ch=qb_ch, name=name, style="flat_top", envelope=env_name, freq=cfg[f"qb_freq_{prefix}"], phase=phase, gain=actual_gain, length=cfg[f"qb_flat_top_length_{prefix}"])

    def setup_standard_gates(self, cfg, prefix='ge', pulse_type='arb', shape='gauss'):
        """Setup standard calibration gates: X180, Y180, X90, Y90, Y90m."""
        self.setup_qb_pulse(cfg, prefix=prefix, pulse_type=pulse_type, shape=shape, name=f"x180_{prefix}", phase=0, gain_key=f"pi_gain_{prefix}")
        self.setup_qb_pulse(cfg, prefix=prefix, pulse_type=pulse_type, shape=shape, name=f"y180_{prefix}", phase=90, gain_key=f"pi_gain_{prefix}")
        self.setup_qb_pulse(cfg, prefix=prefix, pulse_type=pulse_type, shape=shape, name=f"x90_{prefix}", phase=0, gain_key=f"pi2_gain_{prefix}")
        self.setup_qb_pulse(cfg, prefix=prefix, pulse_type=pulse_type, shape=shape, name=f"x90m_{prefix}", phase=180, gain_key=f"pi2_gain_{prefix}")
        self.setup_qb_pulse(cfg, prefix=prefix, pulse_type=pulse_type, shape=shape, name=f"y90_{prefix}", phase=90, gain_key=f"pi2_gain_{prefix}")
        self.setup_qb_pulse(cfg, prefix=prefix, pulse_type=pulse_type, shape=shape, name=f"y90m_{prefix}", phase=-90, gain_key=f"pi2_gain_{prefix}")

    def apply_cooling_setup(self, cfg, cooling=True):
        """Setup cooling channels and pulses."""
        if not cooling:
            return

        for i in [1, 2]:
            ch_key = f"cool_ch{i}"
            if ch_key not in cfg: continue
            
            ch = cfg[ch_key]
            nqz = cfg.get(f"nqz_cool_ch{i}", 2)
            mixer = cfg.get(f"cool_mixer{i}", 0)
            
            if self.soccfg["gens"][ch]["type"] == "axis_sg_int4_v2":
                self.declare_gen(ch=ch, nqz=nqz, mixer_freq=mixer)
            else:
                self.declare_gen(ch=ch, nqz=nqz)
            
            # Setup pulse
            self.add_pulse(
                ch=ch,
                name=f"cool_pulse{i}",
                style="const",
                length=cfg["cool_length"],
                freq=cfg[f"cool_freq_{i}"],
                phase=0,
                gain=cfg[f"cool_gain_{i}"],
            )
