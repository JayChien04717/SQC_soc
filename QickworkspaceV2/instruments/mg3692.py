import pyvisa
import time
from typing import Union, Tuple

ON_OFF_MAP = {
    "on": "1", "1": "1", "true": "1", 1: "1", True: "1",
    "off": "0", "0": "0", "false": "0", 0: "0", False: "0",
}
ON_OFF_MAP_INV = {"1": "on", "0": "off"}


class AnritsuMG3692:
    """
    Driver for the Anritsu MG3692 signal generator (property-based PyVISA API).

    Parameters
    ----------
    address : str
        VISA resource address or plain IP (auto-prefixed with TCPIP::).
    """

    def __init__(self, address: str) -> None:
        self.rm = pyvisa.ResourceManager()
        if "::" not in address:
            self.resource_name = f"TCPIP::{address}::INSTR"
        else:
            self.resource_name = address
        try:
            self.instrument = self.rm.open_resource(self.resource_name)
            self.instrument.timeout = 5000
        except pyvisa.Error as e:
            self.instrument = None
            raise ValueError(f"Could not connect to {self.resource_name}. Error: {e}")

        self.instrument.read_termination = "\n"
        self.instrument.write_termination = "\n"
        self.connect_message()

    def connect_message(self) -> None:
        if self.instrument:
            try:
                idn = self.instrument.query("*IDN?")
                print(f"Connected to: {idn.strip()}")
            except pyvisa.Error as e:
                print(f"Could not query IDN. Error: {e}")

    def close(self) -> None:
        if self.instrument:
            print(f"Disconnecting from {self.resource_name}")
            self.instrument.close()
            self.rm.close()

    def reconnect(self) -> None:
        print(f"[mg3692] Reconnecting to {self.resource_name} ...")
        try:
            try:
                self.instrument.close()
            except Exception:
                pass
            self.instrument = self.rm.open_resource(self.resource_name)
            self.instrument.timeout = 5000
            self.instrument.read_termination  = "\n"
            self.instrument.write_termination = "\n"
            print("[mg3692] Reconnected.")
        except pyvisa.Error as e:
            raise ConnectionError(f"[mg3692] Reconnect failed: {e}") from e

    def write(self, cmd: str) -> None:
        if self.instrument:
            try:
                self.instrument.write(cmd)
            except pyvisa.errors.VisaIOError as e:
                if e.error_code == pyvisa.constants.StatusCode.error_connection_lost:
                    self.reconnect()
                    self.instrument.write(cmd)
                else:
                    raise

    def query(self, cmd: str) -> str:
        if self.instrument:
            try:
                return self.instrument.query(cmd).strip()
            except pyvisa.errors.VisaIOError as e:
                if e.error_code == pyvisa.constants.StatusCode.error_connection_lost:
                    self.reconnect()
                    return self.instrument.query(cmd).strip()
                else:
                    raise
        return ""

    def check_error(self) -> str:
        err_msg = ""
        if self.instrument:
            err_msg = self.query("SYST:ERR?")
            print(f"Instrument Status: {err_msg}")
        return err_msg

    def get_limit(self, parameter: str) -> Tuple[float, float]:
        limits = {
            "frequency": (2e9, 20e9),
            "power": (-120, 27),
        }
        param_lower = parameter.lower()
        if param_lower in limits:
            return limits[param_lower]
        raise ValueError(f"Limits not defined for parameter '{parameter}'.")

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
        try:
            return float(self.query("FREQ?"))
        except ValueError:
            return 0.0

    @frequency.setter
    def frequency(self, value: float) -> None:
        min_f, max_f = self.get_limit("frequency")
        if not (min_f <= value <= max_f):
            print(f"Warning: Frequency {value} Hz is outside normal range ({min_f}, {max_f})")
        self.write(f"FREQ {value} HZ")

    @property
    def power(self) -> float:
        try:
            return float(self.query("POW?"))
        except ValueError:
            return -999.0

    @power.setter
    def power(self, value: float) -> None:
        min_p, max_p = self.get_limit("power")
        if not (min_p <= value <= max_p):
            print(f"Warning: Power {value} dBm is outside normal range ({min_p}, {max_p})")
        self.write(f"POW {value} DBM")

    @property
    def status(self) -> str:
        return self._query_and_map("OUTP?")

    @status.setter
    def status(self, value: Union[str, int, bool]) -> None:
        self._map_and_write("OUTP {}", value, "status")

    def output(self, state: Union[bool, None] = None) -> Union[bool, None]:
        if state is None:
            return self.status == "on"
        self.status = state

    def on(self) -> None:
        self.output(True)

    def off(self) -> None:
        self.output(False)


MG3692Driver = AnritsuMG3692
