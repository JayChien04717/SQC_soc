# ----- Qick package ----- #
from qick import *
from qick.pyro import make_proxy
from qick.asm_v2 import AveragerProgramV2
from qick.asm_v2 import QickSpan, QickSweep1D

# ----- Library ----- #
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import qick_workspace.tools.system_cfg as system_cfg
from qick_workspace.tools.system_tool import select_config_idx

soc, soccfg = system_cfg.soc, system_cfg.soccfg

qubit_idx = 3
config = {
    **system_cfg.hw_cfg,
    **system_cfg.readout_cfg,
    **system_cfg.qubit_cfg,
    **system_cfg.expt_cfg,
}


def test_Loopback():
    from qubit_char import LoopbackProgram

    run_cfg = select_config_idx(config, qubit_idx)
    run_cfg.update(
        {"ro_length": 1.3, "res_length_ge": 0.2, "res_gain_ge": 1, "trig_time": 0}
    )

    prog = LoopbackProgram(
        soccfg, reps=1, final_delay=run_cfg["relax_delay"], cfg=run_cfg
    )
    iq_list = prog.acquire_decimated(soc, soft_avgs=run_cfg["soft_avgs"])
    t = prog.get_time_axis(ro_index=0)

    plt.plot(t, iq_list[0].T[0])
    plt.plot(t, iq_list[0].T[1])
    plt.plot(t, np.abs((iq_list[0]).dot([1, 1j])))
    plt.xlabel("Time (us)")
    plt.ylabel("a.u")


def test_SingleToneSpectroscopyProgram():
    from qubit_char import SingleToneSpectroscopyProgram

    run_cfg = select_config_idx(config, qubit_idx)
    run_cfg.update({"steps": 101, "res_freq_ge": QickSweep1D("freqloop", 5000, 5500)})

    prog = SingleToneSpectroscopyProgram(
        soccfg, reps=run_cfg["reps"], final_delay=run_cfg["relax_delay"], cfg=run_cfg
    )
    iq_list = prog.acquire(soc, soft_avgs=run_cfg["py_avg"])
    freqs = prog.prog.get_pulse_param("res_pulse", "freq", as_array=True)
    plt.plot(freqs, np.abs(iq_list[0][0].dot([1, 1j])))
