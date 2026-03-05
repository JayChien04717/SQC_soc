import sys
import pyvisa as visa
import numpy as np
import time
from typing import Literal, Union


class YOKOGS200:
    """
    This is a PyVISA driver for the Yokogawa GS200 DC Source.

    It has been refactored to use properties (@property) for a
    cleaner, more Pythonic user interface, replacing
    Set/Get methods.

    Ramping is built into the 'voltage' and 'current' property setters
    for safe operation.
    """

    def __init__(self, VISAaddress: str, rm: visa.ResourceManager):
        """
        Initializes the session for the device.

        :param VISAaddress: The VISA resource address (e.g., "GPIB0::1::INSTR")
        :param rm: The PyVISA ResourceManager
        """
        self.VISAaddress = VISAaddress
        try:
            self.session = rm.open_resource(VISAaddress)
            # Set terminations for robust communication
            self.session.read_termination = "\n"
            self.session.write_termination = "\n"
        except visa.Error as ex:
            raise ConnectionError(f"Couldn't connect to '{VISAaddress}'. Error: {ex}")

        # --- Default Ramping Parameters ---
        # These can be changed on the fly, e.g., `yoko.voltage_ramp_step = 1e-5`
        self.voltage_ramp_step = 1e-4  # Step size for voltage ramp
        self.current_ramp_step = 1e-8  # Step size for current ramp
        self.ramp_interval = 0.01  # Dwell time (s) for each ramp step

        # --- Helper maps for properties ---
        self._output_map = {
            "on": "1",
            "1": "1",
            1: "1",
            True: "1",
            "off": "0",
            "0": "0",
            0: "0",
            False: "0",
        }
        self._output_map_inv = {"1": "on", "0": "off"}

        self._mode_map = {"voltage": "VOLT", "current": "CURR"}
        self._mode_map_inv = {"VOLT": "voltage", "CURR": "current"}

        self.connect_message()

    def connect_message(self) -> None:
        """Queries and prints the device IDN to confirm connection."""
        try:
            idn = self.session.query("*IDN?")
            print(f"Connected to: {idn.strip()}")
        except visa.Error as e:
            print(f"Could not query IDN. Error: {e}")

    def close(self) -> None:
        """Closes the VISA session."""
        print(f"Disconnecting from {self.VISAaddress}")
        self.session.close()

    # =========================================================================#
    #  Primary Control Properties
    # =========================================================================#

    @property
    def output(self) -> str:
        """
        Gets the output state ('on' or 'off').
        (SCPI: OUTPut?)
        """
        val = self.session.query("OUTPut?").strip()
        return self.output_map_inv.get(val, f"unknown_state_{val}")

    @output.setter
    def output(self, value: Union[str, int, bool]):
        """
        Sets the output state ('on' or 'off').
        (SCPI: OUTPut 1/0)
        """
        val_str = str(value).lower()
        cmd_val = self._output_map.get(val_str)
        if cmd_val is None:
            raise ValueError(
                f"Invalid output value: {value}. Use 'on', 'off', 1, or 0."
            )
        self.session.write(f"OUTPut {cmd_val}")

    def on(self) -> None:
        """Convenience method to turn output ON."""
        self.output = "on"

    def off(self) -> None:
        """Convenience method to turn output OFF."""
        self.output = "off"

    @property
    def mode(self) -> str:
        """
        Gets the source function mode ('voltage' or 'current').
        (SCPI: SOURce:FUNCtion?)
        """
        val = self.session.query("SOURce:FUNCtion?").strip()
        return self._mode_map_inv.get(val, f"unknown_mode_{val}")

    @mode.setter
    def mode(self, value: Literal["voltage", "current"]):
        """
        Sets the source function mode ('voltage' or 'current').
        (SCPI: SOURce:FUNCtion VOLT/CURR)
        """
        val_str = str(value).lower()
        cmd_val = self._mode_map.get(val_str)
        if cmd_val is None:
            raise ValueError(f"Invalid mode: {value}. Use 'voltage' or 'current'.")
        self.session.write(f"SOURce:FUNCtion {cmd_val}")

    @property
    def level(self) -> float:
        """
        Gets the raw output level (V or A) as a float,
        without changing the mode.
        (SCPI: SOURce:LEVel?)
        """
        result = self.session.query("SOURce:LEVel?")
        return float(result.strip())

    @level.setter
    def level(self, value: float):
        """
        Sets the raw output level (V or A) immediately,
        without ramping and without turning the output on.
        (SCPI: :SOURce:LEVel:AUTO)
        """
        self.session.write(f":SOURce:LEVel:AUTO {value:.8f}")

    @property
    def voltage(self) -> float:
        """
        Gets the output voltage.
        Note: This will set the instrument mode to 'voltage'.
        (SCPI: SOURce:FUNCtion VOLTage, SOURce:LEVel?)
        """
        self.mode = "voltage"
        return self.level

    @voltage.setter
    def voltage(self, new_voltage: float):
        """
        Ramps the voltage (V) to a new value.
        Turns output ON and sets mode to 'voltage'.
        Ramp speed is controlled by `self.voltage_ramp_step`
        and `self.ramp_interval`.
        """
        self.mode = "voltage"
        start = self.level
        stop = new_voltage

        steps = max(1, round(abs(stop - start) / self.voltage_ramp_step))
        temp_volts = np.linspace(start, stop, num=steps + 1, endpoint=True)

        self.on()
        for v in temp_volts:
            self.level = v  # Uses the raw 'level' setter
            time.sleep(self.ramp_interval)

    @property
    def current(self) -> float:
        """
        Gets the output current.
        Note: This will set the instrument mode to 'current'.
        (SCPI: SOURce:FUNCtion CURRent, SOURce:LEVel?)
        """
        self.mode = "current"
        return self.level

    @current.setter
    def current(self, new_current: float):
        """
        Ramps the current (A) to a new value.
        Turns output ON and sets mode to 'current'.
        Ramp speed is controlled by `self.current_ramp_step`
        and `self.ramp_interval`.
        """
        self.mode = "current"
        start = self.level
        stop = new_current

        steps = max(1, round(abs(stop - start) / self.current_ramp_step))
        temp_currents = np.linspace(start, stop, num=steps + 1, endpoint=True)

        self.on()
        for c in temp_currents:
            self.level = c  # Uses the raw 'level' setter
            time.sleep(self.ramp_interval)

    # =========================================================================#
    #  Helper Methods
    # =========================================================================#

    def GetValue(self) -> dict:
        """
        Returns the current value and unit based on the active mode.
        This is a "read-only" operation and will not change the mode.
        """
        current_mode = self.mode  # Uses mode getter
        current_level = self.level  # Uses level getter

        if current_mode == "voltage":
            return dict(unit="V", value=current_level)
        else:
            return dict(unit="A", value=current_level)


if __name__ == "__main__":
    # Create a resource manager
    rm = visa.ResourceManager()
    # Connect to the instrument
    yoko = YOKOGS200("GPIB0::1::INSTR", rm)

    # --- New Property-Based Usage ---

    # Set the mode
    yoko.mode = "current"

    # Set the current (this will automatically ramp)
    yoko.current = 1e-3  # Ramps from 0 to 1mA

    # Check the output state
    print(f"Output is: {yoko.output}")

    # Turn on the output (redundant if already on from setting current)
    yoko.on()

    # Change the ramp speed
    yoko.current_ramp_step = 2e-8
    yoko.ramp_interval = 0.005

    # Ramp to a new current
    yoko.current = -1e-3

    # Turn off the output
    yoko.off()

    # Close the connection
    yoko.close()
