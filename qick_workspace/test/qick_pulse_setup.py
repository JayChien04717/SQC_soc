def declare_gen_ch(expt_func, cfg, ch, usage="res", suffix=""):
    """
    declare_gen_ch declares a generator channel properly depending on the configurations

    expt_func: is a variable for "self" (i.e. the experiment function) in the code
    cfg: The configuration file so that you can pull values for freqs and gains
    ch: The channel index that you want to declare
    usage: 'res' or 'qubit'
    pulse_style: type of pulse you want to send 'const', 'arb', 'flat_top'
    suffix: do the parameters have a suffix? ('', '_ge', or '_ef')

    returns:
        nothing...
    """

    if (
        expt_func.soccfg["gens"][ch]["type"] in "axis_signal_gen_v6"
    ):  # fast gen channels
        expt_func.declare_gen(ch=ch, nqz=cfg["nqz_" + usage])
    elif expt_func.soccfg["gens"][ch]["type"] in "axis_sg_mixmux8_v1":  # MUX channel
        expt_func.declare_gen(
            ch=ch,
            nqz=cfg["nqz_" + usage],
            ro_ch=cfg["mux_ro_chs"][0],
            mux_freqs=cfg[usage + "_freq" + suffix],
            mux_gains=cfg[usage + "_gain" + suffix],
            mux_phases=cfg[usage + "_phase"],
            mixer_freq=cfg["mixer_freq"],
        )
    elif (
        expt_func.soccfg["gens"][ch]["type"] in "axis_sg_int4_v2"
    ):  # interpolated channels
        expt_func.declare_gen(
            ch=ch,
            nqz=cfg["nqz_" + usage],
            ro_ch=cfg["ro_ch"],
            mixer_freq=cfg[usage + "_mixer_freq" + suffix],
        )


def declare_pulse(
    expt_func, cfg, ch, usage="res", pulse_style="const", pulse_name="pulse", suffix=""
):
    """
    declare_pulse declares a pulse to be sent properly depending on the configurations

    expt_func: is a variable for "self" (i.e. the experiment function) in the code
    cfg: The configuration file so that you can pull values for freqs and gains
    ch: The channel index that you want to declare
    usage: 'res' or 'qubit'
    pulse_style: type of pulse you want to send 'const', 'arb', 'flat_top' ('gauss' makes a gaussian pulse for us)
    pulse_name: name of the pulse we will send ('ramsey1' and 'ramsey2' are special)
    suffix: do the parameters have a suffix? ('', '_ge', or '_ef')

    returns:
        nothing...
    """
    if pulse_name == "ramsey1":
        expt_func.add_gauss(
            ch=ch,
            name="ramp" + suffix,
            sigma=cfg["sigma" + suffix],
            length=cfg["sigma" + suffix] * 5,
            even_length=True,
        )
        expt_func.add_pulse(
            ch=ch,
            name=pulse_name,
            ro_ch=cfg["ro_ch"],
            style="arb",
            envelope="ramp" + suffix,
            freq=cfg[usage + "_freq" + suffix],
            phase=cfg[usage + "_phase"],
            gain=cfg[usage + "_gain" + suffix] / 2,
        )
    elif pulse_name == "ramsey2":
        expt_func.add_gauss(
            ch=ch,
            name="ramp" + suffix,
            sigma=cfg["sigma" + suffix],
            length=cfg["sigma" + suffix] * 5,
            even_length=True,
        )
        expt_func.add_pulse(
            ch=ch,
            name=pulse_name,
            ro_ch=cfg["ro_ch"],
            style="arb",
            envelope="ramp" + suffix,
            freq=cfg[usage + "_freq" + suffix],
            phase=cfg[usage + "_phase"] + cfg["wait_time"] * 360 * cfg["ramsey_freq"],
            gain=cfg[usage + "_gain" + suffix] / 2,
        )
    elif pulse_name == "pi":
        expt_func.add_gauss(
            ch=ch,
            name="ramp" + suffix,
            sigma=cfg["sigma" + suffix],
            length=cfg["sigma" + suffix] * 5,
            even_length=True,
        )
        expt_func.add_pulse(
            ch=ch,
            name=pulse_name,
            ro_ch=cfg["ro_ch"],
            style="arb",
            envelope="ramp" + suffix,
            freq=cfg[usage + "_freq" + suffix],
            phase=cfg[usage + "_phase"],
            gain=cfg[usage + "_gain" + suffix],
        )
    else:
        if pulse_style == "gauss":  # for a gaussian pulse
            expt_func.add_gauss(
                ch=ch,
                name="ramp" + suffix,
                sigma=cfg["sigma" + suffix],
                length=cfg["sigma" + suffix] * 5,
                even_length=True,
            )
            expt_func.add_pulse(
                ch=ch,
                name=pulse_name,
                ro_ch=cfg["ro_ch"],
                style="arb",
                envelope="ramp" + suffix,
                freq=cfg[usage + "_freq" + suffix],
                phase=cfg[usage + "_phase"],
                gain=cfg[usage + "_gain" + suffix],
            )

        else:  # for other styles of pulses
            expt_func.add_pulse(
                ch=ch,
                name=pulse_name,
                ro_ch=cfg["ro_ch"],
                style=pulse_style,
                length=cfg[usage + "_length" + suffix],
                freq=cfg[usage + "_freq" + suffix],
                phase=cfg[usage + "_phase"],
                gain=cfg[usage + "_gain" + suffix],
            )


def declare_flux_pulse(
    expt_func,
    cfg,
    ch,
    usage="flux",
    pulse_style="const",
    pulse_name="flux_pulse",
    suffix="",
):
    """
    declare_pulse declares a pulse to be sent properly depending on the configurations

    expt_func: is a variable for "self" (i.e. the experiment function) in the code
    cfg: The configuration file so that you can pull values for freqs and gains
    ch: The channel index that you want to declare
    usage: 'flux'
    pulse_style: type of pulse you want to send 'const', 'arb', 'flat_top' ('gauss' makes a gaussian pulse for us)
    pulse_name: name of the pulse we will send ('ramsey1' and 'ramsey2' are special)
    suffix: do the parameters have a suffix? ('', '_q1', or '_qn')

    returns:
        nothing...
    """

    expt_func.add_pulse(
        ch=ch,
        name=pulse_name,
        style=pulse_style,
        length=cfg[usage + "_length" + suffix],
        freq=0,
        phase=0,
        gain=cfg[usage + "_gain" + suffix],
    )


# def declare_cooling_pulse(
#     expt_func,
#     cfg,
#     ch,
#     ch2,
#     pulse_style="const",
# ):
#     """
#     declare_pulse declares a pulse to be sent properly depending on the configurations

#     expt_func: is a variable for "self" (i.e. the experiment function) in the code
#     cfg: The configuration file so that you can pull values for freqs and gains
#     ch: The channel index that you want to declare
#     usage: 'res' or 'qubit'
#     pulse_style: type of pulse you want to send 'const', 'arb', 'flat_top' ('gauss' makes a gaussian pulse for us)
#     pulse_name: name of the pulse we will send ('ramsey1' and 'ramsey2' are special)
#     suffix: do the parameters have a suffix? ('', '_ge', or '_ef')

#     returns:
#         nothing...
#     """

#     expt_func.add_pulse(
#         ch=ch,
#         name="cool_pulse1",
#         style=pulse_style,
#         length=cfg["cool_length"],
#         freq=cfg["cool_freq_1"],
#         phase=0,
#         gain=cfg["cool_gain_1"],
#     )
#     expt_func.add_pulse(
#         ch=ch2,
#         name="cool_pulse2",
#         style=pulse_style,
#         length=cfg["cool_length"],
#         freq=cfg["cool_freq_2"],
#         phase=0,
#         gain=cfg["cool_gain_2"],
#     )


def initialize_ro_chs(expt_func, cfg, MUX: bool = False, suffix=""):
    """
    initialize_ro_chs initializes the readout signal generator and readout ADC channels properly depending on the configurations

    expt_func: is a variable for "self" (i.e. the experiment function) in the code
    cfg: The configuration file so that you can pull values for freqs and gains
    suffix: do the parameters have a suffix? ('', '_ge', or '_ef')

    returns:
        nothing...
    """

    if MUX == "True":  # for MUX readout
        ro_chs = cfg["mux_ro_chs"]
        res_ch = cfg["mux_ch"]

        # declare a MUX gen channel
        declare_gen_ch(expt_func, cfg, res_ch, usage="res", suffix=suffix)

        for ch, f, ph in zip(ro_chs, cfg["res_freq" + suffix], cfg["res_phase"]):
            expt_func.declare_readout(
                ch=ch, length=cfg["ro_length"], freq=f, phase=ph, gen_ch=res_ch
            )

        # declare a pulse for the MUX channel
        declare_pulse(expt_func, cfg, res_ch, pulse_name="res_pulse", suffix=suffix)

    else:  # for non-MUX readout
        ro_ch = cfg["ro_ch"]
        res_ch = cfg["res_ch"]

        # declare the proper readout channel using
        declare_gen_ch(expt_func, cfg, res_ch, usage="res", suffix=suffix)

        expt_func.declare_readout(ch=ro_ch, length=cfg["ro_length"])
        expt_func.add_readoutconfig(
            ch=ro_ch, name="myro", freq=cfg["res_freq" + suffix], gen_ch=res_ch
        )

        # declare a pulse for the non-MUX channel
        declare_pulse(expt_func, cfg, res_ch, pulse_name="res_pulse", suffix=suffix)


def readout(expt_func, cfg, MUX: bool = False):
    """
    readout function sends a readout pulse and triggers the adc depending on the configurations

    expt_func: is a variable for "self" (i.e. the experiment function) in the code
    cfg: The configuration file so that you can pull values for freqs and gains

    returns:
        nothing...

    Note: we may want to rewrite this to not depend on the 'MUX' setting but rather the readout channel
          that is input. For now this should work fine, but if firmware changes or other types of
          channels are used, then we want to adjust for this.
    """

    if MUX == "True":  # for MUX readout
        expt_func.pulse(ch=cfg["mux_ch"], name="res_pulse", t=0)
        expt_func.trigger(ros=cfg["mux_ro_chs"], pins=[0], t=cfg["trig_time"])

    else:  # for non-MUX readout
        # if non-MUX then send readout configurations - useful when freq sweeping
        expt_func.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        expt_func.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        expt_func.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])
