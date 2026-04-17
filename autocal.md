# Giving file
   * pre calibrate data: rougth resontor freq, qb, freq
# Calibrate procudure
    1. resonator spectrum -> update resonator frequency
    2. qubit ge specturm -> update qb frequency
    3. power rabi -> update pi gain
       1. if pi gain > 1 (over instruement limit)
       2. increase sigma_ge (increase how much?)
       3. measurement again
       4. if pi gain < 0.2
       5. decrease sigma_ge (decrease how much?)
       6. measurement again
    4. short time ramsey(0~2us, ramsey freq 2MHz)
       1. use rasmey freq and fitting detuen to fine tune qb freq
       2. after update, measurement again to check detune again
       3. if detune < limit go done
       4. if detune > limit, after fine tune, re measurement power rabi
    5. spin echo (how to determine measurement time? if qb life time is too long, should decrease ramsey frequency to prevent oscillation too fast)
    6. T1 (also how to determint measurement time?)
    7. single shot optimize
       1. freq_axis = np.linspace(run_cfg["res_freq_ge"] - 1, run_cfg["res_freq_ge"] + 8, 11)
        gain_axis = np.linspace(0.07, 0.1, 5)
        length_axis = np.linspace(2, 4, 4)
        sweep_para = {"freq":run_cfg["res_freq_ge"] , "gain": gain_axis, "length": length_axis}
