import pyvisa as visa
import numpy as np
import time
from typing import Literal, Union


class YOKOGS200:
    """
    PyVISA driver for the Yokogawa GS200 DC Source (property-based API).

    Ramping is built into the 'voltage' and 'current' property setters
    for safe operation.
    """

    def __init__(self, VISAaddress: str, rm: visa.ResourceManager):
        self.VISAaddress = VISAaddress
        try:
            self.session = rm.open_resource(VISAaddress)
            self.session.read_termination = "\n"
            self.session.write_termination = "\n"
        except visa.Error as ex:
            raise ConnectionError(f"Couldn't connect to '{VISAaddress}'. Error: {ex}")

        self.voltage_ramp_step = 1e-4
        self.current_ramp_step = 1e-8
        self.ramp_interval = 0.01

        self._output_map = {
            "on": "1", "1": "1", 1: "1", True: "1",
            "off": "0", "0": "0", 0: "0", False: "0",
        }
        self._output_map_inv = {"1": "on", "0": "off"}
        self._mode_map = {"voltage": "VOLT", "current": "CURR"}
        self._mode_map_inv = {"VOLT": "voltage", "CURR": "current"}

        self.connect_message()

    def connect_message(self) -> None:
        try:
            idn = self.session.query("*IDN?")
            print(f"Connected to: {idn.strip()}")
        except visa.Error as e:
            print(f"Could not query IDN. Error: {e}")

    def close(self) -> None:
        print(f"Disconnecting from {self.VISAaddress}")
        self.session.close()

    @property
    def output(self) -> str:
        val = self.session.query("OUTPut?").strip()
        return self._output_map_inv.get(val, f"unknown_state_{val}")

    @output.setter
    def output(self, value: Union[str, int, bool]):
        val_str = str(value).lower()
        cmd_val = self._output_map.get(val_str)
        if cmd_val is None:
            raise ValueError(f"Invalid output value: {value}. Use 'on', 'off', 1, or 0.")
        self.session.write(f"OUTPut {cmd_val}")

    def on(self) -> None:
        self.output = "on"

    def off(self) -> None:
        self.output = "off"

    @property
    def mode(self) -> str:
        val = self.session.query("SOURce:FUNCtion?").strip()
        return self._mode_map_inv.get(val, f"unknown_mode_{val}")

    @mode.setter
    def mode(self, value: Literal["voltage", "current"]):
        val_str = str(value).lower()
        cmd_val = self._mode_map.get(val_str)
        if cmd_val is None:
            raise ValueError(f"Invalid mode: {value}. Use 'voltage' or 'current'.")
        self.session.write(f"SOURce:FUNCtion {cmd_val}")

    @property
    def level(self) -> float:
        result = self.session.query("SOURce:LEVel?")
        return float(result.strip())

    @level.setter
    def level(self, value: float):
        self.session.write(f":SOURce:LEVel:AUTO {value:.8f}")

    @property
    def voltage(self) -> float:
        self.mode = "voltage"
        return self.level

    @voltage.setter
    def voltage(self, new_voltage: float):
        self.mode = "voltage"
        start = self.level
        stop = new_voltage
        steps = max(1, round(abs(stop - start) / self.voltage_ramp_step))
        temp_volts = np.linspace(start, stop, num=steps + 1, endpoint=True)
        self.on()
        for v in temp_volts:
            self.level = v
            time.sleep(self.ramp_interval)

    @property
    def current(self) -> float:
        self.mode = "current"
        return self.level

    @current.setter
    def current(self, new_current: float):
        self.mode = "current"
        start = self.level
        stop = new_current
        steps = max(1, round(abs(stop - start) / self.current_ramp_step))
        temp_currents = np.linspace(start, stop, num=steps + 1, endpoint=True)
        self.on()
        for c in temp_currents:
            self.level = c
            time.sleep(self.ramp_interval)

    def GetValue(self) -> dict:
        current_mode = self.mode
        current_level = self.level
        if current_mode == "voltage":
            return dict(unit="V", value=current_level)
        else:
            return dict(unit="A", value=current_level)
