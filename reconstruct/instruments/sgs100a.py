import pyvisa
from typing import Literal, Union, Tuple

ON_OFF_MAP = {
    "on": "1", "1": "1", 1: "1", True: "1",
    "off": "0", "0": "0", 0: "0", False: "0",
}
ON_OFF_MAP_INV = {"1": "on", "0": "off"}

PULSE_SOURCE_VALS = {"INT", "EXT"}
REF_LO_SOURCE_VALS = {"INT", "EXT"}
REF_LO_OUT_VALS = {"REF", "LO", "OFF"}
REF_FREQ_VALS = {"10MHZ", "100MHZ", "1000MHZ"}
TRIG_MODE_VALS = {"SVAL", "SNVAL", "PVO", "PET", "PEMS"}
PULSE_MODE_VALS = {"SING", "DOUB", "SINGLE", "DOUBLE"}
POLARITY_VALS = {"NORM", "INV", "NORMAL", "INVERTED"}
IMPEDANCE_VALS = {"G50", "G10K"}
SLOPE_VALS = {"NEG", "POS", "NEGATIVE", "POSITIVE"}
TRIG_MODE_EXT_VALS = {"AUTO", "EXT", "EGAT", "EXTERNAL", "EGATE"}
OP_MODE_VALS = {"NORMAL", "BBBYPASS"}


class RohdeSchwarzSGS100A:
    """Pure PyVISA driver for the Rohde & Schwarz SGS100A signal generator."""

    def __init__(self, address: str) -> None:
        self.rm = pyvisa.ResourceManager()
        try:
            self.instrument = self.rm.open_resource(address)
        except pyvisa.Error as e:
            print(f"Could not connect to {address}. Error: {e}")
            raise
        self.instrument.read_termination = "\n"
        self.instrument.write_termination = "\n"
        self.connect_message()

    def connect_message(self) -> None:
        try:
            idn = self.instrument.query("*IDN?")
            print(f"Connected to: {idn.strip()}")
        except pyvisa.Error as e:
            print(f"Could not query IDN. Error: {e}")

    def close(self) -> None:
        print(f"Disconnecting from {self.instrument.resource_name}")
        self.instrument.close()
        self.rm.close()

    def write(self, cmd: str) -> None:
        self.instrument.write(cmd)

    def query(self, cmd: str) -> str:
        return self.instrument.query(cmd).strip()

    def reset(self) -> None:
        print("Resetting instrument...")
        self.write("*RST")

    def run_self_tests(self) -> str:
        print("Running self-tests...")
        result = self.query("*TST?")
        print(f"Self-test result: {result}")
        return result

    def check_error(self) -> str:
        err_msg = self.query("SYST:ERR?")
        print(f"Instrument Status: {err_msg}")
        return err_msg

    def get_limit(self, parameter: str) -> Tuple[float, float]:
        limits = {
            "frequency": (1e6, 20e9),
            "phase": (0, 360),
            "power": (-120, 25),
            "i_offset": (-10, 10),
            "q_offset": (-10, 10),
            "iq_gain_imbalance": (-1, 1),
            "iq_angle": (-8, 8),
            "pulsemod_delay": (0, 100),
        }
        param_lower = parameter.lower()
        if param_lower in limits:
            return limits[param_lower]
        raise ValueError(f"Limits not defined for parameter '{parameter}'.")

    def _validate_and_write(self, cmd_template: str, value: str, valid_set: set, name: str) -> None:
        val_upper = str(value).upper()
        if val_upper not in valid_set:
            raise ValueError(f"Invalid {name} value: {value}. Allowed: {valid_set}")
        self.write(cmd_template.format(val_upper))

    def _map_and_write(self, cmd_template: str, value: Union[str, int, bool], name: str) -> None:
        try:
            mapped_val = ON_OFF_MAP[str(value).lower()]
            self.write(cmd_template.format(mapped_val))
        except KeyError:
            raise ValueError(f"Invalid {name} value: {value}. Use 'on' or 'off'.")

    def _query_and_map(self, cmd: str) -> str:
        val = self.query(cmd)
        return ON_OFF_MAP_INV.get(val, f"unknown_val_{val}")

    @property
    def frequency(self) -> float:
        return float(self.query("SOUR:FREQ?"))

    @frequency.setter
    def frequency(self, value: float) -> None:
        min_v, max_v = self.get_limit("frequency")
        if not (min_v <= value <= max_v):
            print(f"Warning: Frequency {value} Hz is outside driver's expected range ({min_v}, {max_v})")
        self.write(f"SOUR:FREQ {value:.2f}")

    @property
    def phase(self) -> float:
        return float(self.query("SOUR:PHAS?"))

    @phase.setter
    def phase(self, value: float) -> None:
        min_v, max_v = self.get_limit("phase")
        if not (min_v <= value <= max_v):
            print(f"Warning: Phase {value} deg is outside driver's expected range ({min_v}, {max_v})")
        self.write(f"SOUR:PHAS {value:.2f}")

    @property
    def power(self) -> float:
        return float(self.query("SOUR:POW?"))

    @power.setter
    def power(self, value: float) -> None:
        min_v, max_v = self.get_limit("power")
        if not (min_v <= value <= max_v):
            print(f"Warning: Power {value} dBm is outside driver's expected range ({min_v}, {max_v})")
        self.write(f"SOUR:POW {value:.2f}")

    @property
    def status(self) -> str:
        return self._query_and_map(":OUTP:STAT?")

    @status.setter
    def status(self, value: Union[str, int, bool]) -> None:
        self._map_and_write(":OUTP:STAT {}", value, "status")

    def on(self) -> None:
        self.status = "on"

    def off(self) -> None:
        self.status = "off"

    @property
    def IQ_state(self) -> str:
        return self._query_and_map(":IQ:STAT?")

    @IQ_state.setter
    def IQ_state(self, value: Union[str, int, bool]) -> None:
        self._map_and_write(":IQ:STAT {}", value, "IQ_state")

    @property
    def pulsemod_state(self) -> str:
        return self._query_and_map(":SOUR:PULM:STAT?")

    @pulsemod_state.setter
    def pulsemod_state(self, value: Union[str, int, bool]) -> None:
        self._map_and_write(":SOUR:PULM:STAT {}", value, "pulsemod_state")

    @property
    def pulsemod_source(self) -> str:
        return self.query("SOUR:PULM:SOUR?")

    @pulsemod_source.setter
    def pulsemod_source(self, value: Literal["INT", "EXT", "int", "ext"]) -> None:
        self._validate_and_write("SOUR:PULM:SOUR {}", value, PULSE_SOURCE_VALS, "pulsemod_source")

    @property
    def ref_osc_source(self) -> str:
        return self.query("SOUR:ROSC:SOUR?")

    @ref_osc_source.setter
    def ref_osc_source(self, value: Literal["INT", "EXT", "int", "ext"]) -> None:
        self._validate_and_write("SOUR:ROSC:SOUR {}", value, REF_LO_SOURCE_VALS, "ref_osc_source")

    @property
    def LO_source(self) -> str:
        return self.query("SOUR:LOSC:SOUR?")

    @LO_source.setter
    def LO_source(self, value: Literal["INT", "EXT", "int", "ext"]) -> None:
        self._validate_and_write("SOUR:LOSC:SOUR {}", value, REF_LO_SOURCE_VALS, "LO_source")

    @property
    def ref_LO_out(self) -> str:
        return self.query("CONN:REFL:OUTP?")

    @ref_LO_out.setter
    def ref_LO_out(self, value: Literal["REF", "LO", "OFF", "ref", "lo", "off"]) -> None:
        self._validate_and_write("CONN:REFL:OUTP {}", value, REF_LO_OUT_VALS, "ref_LO_out")

    @property
    def ref_osc_output_freq(self) -> str:
        return self.query("SOUR:ROSC:OUTP:FREQ?")

    @ref_osc_output_freq.setter
    def ref_osc_output_freq(self, value: Literal["10MHz", "100MHz", "1000MHz"]) -> None:
        self._validate_and_write("SOUR:ROSC:OUTP:FREQ {}", value, REF_FREQ_VALS, "ref_osc_output_freq")

    @property
    def ref_osc_external_freq(self) -> str:
        return self.query("SOUR:ROSC:EXT:FREQ?")

    @ref_osc_external_freq.setter
    def ref_osc_external_freq(self, value: Literal["10MHz", "100MHz", "1000MHz"]) -> None:
        self._validate_and_write("SOUR:ROSC:EXT:FREQ {}", value, REF_FREQ_VALS, "ref_osc_external_freq")

    @property
    def IQ_impairments(self) -> str:
        return self._query_and_map(":SOUR:IQ:IMP:STAT?")

    @IQ_impairments.setter
    def IQ_impairments(self, value: Union[str, int, bool]) -> None:
        self._map_and_write(":SOUR:IQ:IMP:STAT {}", value, "IQ_impairments")

    @property
    def I_offset(self) -> float:
        return float(self.query("SOUR:IQ:IMP:LEAK:I?"))

    @I_offset.setter
    def I_offset(self, value: float) -> None:
        min_v, max_v = self.get_limit("i_offset")
        if not (min_v <= value <= max_v):
            print(f"Warning: I offset {value}% is outside expected range ({min_v}, {max_v})")
        self.write(f"SOUR:IQ:IMP:LEAK:I {value:.2f}")

    @property
    def Q_offset(self) -> float:
        return float(self.query("SOUR:IQ:IMP:LEAK:Q?"))

    @Q_offset.setter
    def Q_offset(self, value: float) -> None:
        min_v, max_v = self.get_limit("q_offset")
        if not (min_v <= value <= max_v):
            print(f"Warning: Q offset {value}% is outside expected range ({min_v}, {max_v})")
        self.write(f"SOUR:IQ:IMP:LEAK:Q {value:.2f}")

    @property
    def IQ_gain_imbalance(self) -> float:
        return float(self.query("SOUR:IQ:IMP:IQR?"))

    @IQ_gain_imbalance.setter
    def IQ_gain_imbalance(self, value: float) -> None:
        min_v, max_v = self.get_limit("iq_gain_imbalance")
        if not (min_v <= value <= max_v):
            print(f"Warning: IQ gain imbalance {value} dB is outside expected range ({min_v}, {max_v})")
        self.write(f"SOUR:IQ:IMP:IQR {value:.2f}")

    @property
    def IQ_angle(self) -> float:
        return float(self.query("SOUR:IQ:IMP:QUAD?"))

    @IQ_angle.setter
    def IQ_angle(self, value: float) -> None:
        min_v, max_v = self.get_limit("iq_angle")
        if not (min_v <= value <= max_v):
            print(f"Warning: IQ angle {value} deg is outside expected range ({min_v}, {max_v})")
        self.write(f"SOUR:IQ:IMP:QUAD {value:.2f}")

    @property
    def trigger_connector_mode(self) -> str:
        return self.query("CONN:TRIG:OMOD?")

    @trigger_connector_mode.setter
    def trigger_connector_mode(self, value: str) -> None:
        self._validate_and_write("CONN:TRIG:OMOD {}", value, TRIG_MODE_VALS, "trigger_connector_mode")

    @property
    def pulsemod_delay(self) -> float:
        return float(self.query("SOUR:PULM:DEL?"))

    @pulsemod_delay.setter
    def pulsemod_delay(self, value: float) -> None:
        min_v, max_v = self.get_limit("pulsemod_delay")
        if not (min_v <= value <= max_v):
            print(f"Warning: Pulse modulation delay {value} s is outside expected range ({min_v}, {max_v})")
        self.write(f"SOUR:PULM:DEL {value:g}")


class RohdeSchwarz_SGS100A(RohdeSchwarzSGS100A):
    """Alias for backwards compatibility."""
    pass
