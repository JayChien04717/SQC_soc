import pyvisa
import time
from typing import Union, Tuple

# --- Helper Dictionaries for Validation ---

# Map 'on'/'off' strings to instrument commands '1'/'0'
ON_OFF_MAP = {
    "on": "1",
    "1": "1",
    1: "1",
    True: "1",
    "off": "0",
    "0": "0",
    0: "0",
    False: "0",
}
# Inverse map for 'get' methods
ON_OFF_MAP_INV = {"1": "on", "0": "off"}

class AnritsuMG3692:
    """
    Driver for the Anritsu MG3692 signal generator.

    Provides a property-based API to control instrument parameters via PyVISA,
    mimicking the QCoDeS driver style. It adopts the same architecture as the 
    SGS100A driver.

    Parameters
    ----------
    address : str
        The VISA resource address or IP address of the instrument.
        If an IP address string is provided without VISA prefix, it will 
        automatically be prefixed with 'TCPIP::' and suffixed with '::INSTR'.

    Raises
    ------
    ValueError
        If the connection fails or initialization is unsuccessful.
    """

    def __init__(self, address: str) -> None:
        self.rm = pyvisa.ResourceManager()
        
        # Format the address correctly if a plain IP is passed
        if "::" not in address:
            self.resource_name = f"TCPIP::{address}::INSTR"
        else:
            self.resource_name = address
            
        try:
            self.instrument = self.rm.open_resource(self.resource_name)
            self.instrument.timeout = 5000  # 5 seconds timeout
        except pyvisa.Error as e:
            print(f"Could not connect to {self.resource_name}. Error: {e}")
            self.instrument = None
            raise ValueError(f"Could not connect to {self.resource_name}. Error: {e}")

        # Set terminations
        self.instrument.read_termination = "\n"
        self.instrument.write_termination = "\n"

        self.connect_message()

    def connect_message(self) -> None:
        """
        Queries and prints the IDN to confirm the connection.

        Sends the *IDN? command and fetches the identification string to ensure
        the software is talking to the correct hardware.

        Returns
        -------
        None
        """
        if self.instrument:
            try:
                idn = self.instrument.query("*IDN?")
                print(f"Connected to: {idn.strip()}")
            except pyvisa.Error as e:
                print(f"Could not query IDN. Error: {e}")

    def close(self) -> None:
        """
        Closes the VISA connection to the instrument.

        Releases the system resources utilized by the VISA session.

        Returns
        -------
        None
        """
        if self.instrument:
            print(f"Disconnecting from {self.resource_name}")
            self.instrument.close()
            self.rm.close()

    def write(self, cmd: str) -> None:
        """
        Sends a SCPI write command.

        Parameters
        ----------
        cmd : str
            The SCPI command string to send to the instrument.

        Returns
        -------
        None
        """
        if self.instrument:
            self.instrument.write(cmd)

    def query(self, cmd: str) -> str:
        """
        Sends a SCPI query command and returns the stripped result.

        Parameters
        ----------
        cmd : str
            The SCPI query command string.

        Returns
        -------
        result : str
            The response string from the instrument.
        """
        if self.instrument:
            return self.instrument.query(cmd).strip()
        return ""

    def check_error(self) -> str:
        """
        Checks the instrument's error queue.

        Queries the system error status and prints the message. This tells the user 
        if any recent syntactical or parameter range errors occurred internally.

        Returns
        -------
        err_msg : str
            The error message retrieved from the instrument.
        """
        err_msg = ""
        if self.instrument:
            err_msg = self.query("SYST:ERR?")
            print(f"Instrument Status: {err_msg}")
        return err_msg

    def get_limit(self, parameter: str) -> Tuple[float, float]:
        """
        Retrieves the operating limits for a given parameter.

        Returns the min and max limits for parameters such as
        frequency or power based on the hardware specifications. This helps 
        guard against setting target values beyond what the hardware enables.

        Parameters
        ----------
        parameter : str
            The parameter name to get limits for (e.g., 'frequency' or 'power').

        Returns
        -------
        limits : tuple
            A tuple of (min_value, max_value) representing the allowed range.

        Raises
        ------
        ValueError
            If the requested parameter has no defined limits in the driver.
        """
        limits = {
            "frequency": (2e9, 20e9),  # Example standard limits for MG3692
            "power": (-120, 10),       # Example standard power limits
        }
        param_lower = parameter.lower()
        if param_lower in limits:
            return limits[param_lower]
        raise ValueError(f"Limits not defined for parameter '{parameter}'.")

    # --- Helper methods ---

    def _map_and_write(self, cmd_template: str, value: Union[str, int, bool], name: str) -> None:
        """
        Validates boolean-like inputs and writes the mapped SCPI command.

        Parameters
        ----------
        cmd_template : str
            A template string for the command containing '{}'.
        value : str, int, bool
            The user-supplied value (e.g., 'on', 'off', True).
        name : str
            The property name for error reporting.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the value cannot be mapped to '1' or '0'.
        """
        try:
            mapped_val = ON_OFF_MAP[str(value).lower()]
            self.write(cmd_template.format(mapped_val))
        except KeyError:
            raise ValueError(f"Invalid {name} value: {value}. Use 'on' or 'off'.")

    def _query_and_map(self, cmd: str) -> str:
        """
        Queries a state and maps the numeric response back to 'on' or 'off'.

        Parameters
        ----------
        cmd : str
            The SCPI query command string.

        Returns
        -------
        state : str
            The mapped state ('on' or 'off'), or original string if mapping fails.
        """
        val = self.query(cmd)
        return ON_OFF_MAP_INV.get(val, f"unknown_val_{val}")

    # --- Properties ---

    @property
    def frequency(self) -> float:
        """
        Gets the RF output frequency.

        Returns
        -------
        freq : float
            The frequency value in Hz.
        """
        # Note: MG3692 requires FREQ? command or similar (assuming standard SCPI)
        try:
            return float(self.query("FREQ?"))
        except ValueError:
            return 0.0

    @frequency.setter
    def frequency(self, value: float) -> None:
        """
        Sets the RF output frequency.

        Validates the requested value against the hardware bounds via `get_limit()`.

        Parameters
        ----------
        value : float
            The target frequency in Hz.
            
        Returns
        -------
        None
        """
        min_f, max_f = self.get_limit("frequency")
        if not (min_f <= value <= max_f):
            print(f"Warning: Frequency {value} Hz is outside normal range ({min_f}, {max_f})")
        self.write(f"FREQ {value} HZ")

    @property
    def power(self) -> float:
        """
        Gets the RF output power.

        Returns
        -------
        power_dbm : float
            Current output power level in dBm.
        """
        try:
            return float(self.query("POW?"))
        except ValueError:
            return -999.0

    @power.setter
    def power(self, value: float) -> None:
        """
        Sets the RF output power.

        Validates the requested value against the hardware bounds via `get_limit()`.

        Parameters
        ----------
        value : float
            Target output power in dBm.
            
        Returns
        -------
        None
        """
        min_p, max_p = self.get_limit("power")
        if not (min_p <= value <= max_p):
            print(f"Warning: Power {value} dBm is outside normal range ({min_p}, {max_p})")
        self.write(f"POW {value} DBM")

    @property
    def status(self) -> str:
        """
        Gets the RF output status.

        Returns
        -------
        state : str
            'on' or 'off' indicating current output state.
        """
        # Anritsu MG3692 usually uses OUTP?
        return self._query_and_map("OUTP?")

    @status.setter
    def status(self, value: Union[str, int, bool]) -> None:
        """
        Sets the RF output status.

        Parameters
        ----------
        value : str, int, bool
            'on', 'off', True, False, 1, or 0.
            
        Returns
        -------
        None
        """
        self._map_and_write("OUTP {}", value, "status")

    def on(self) -> None:
        """
        Turns the RF output on explicitly.

        A convenience shortcut to modify the status property.

        Returns
        -------
        None
        """
        self.status = "on"

    def off(self) -> None:
        """
        Turns the RF output off explicitly.

        A convenience shortcut to modify the status property.

        Returns
        -------
        None
        """
        self.status = "off"

# Alias for backwards compatibility
MG3692Driver = AnritsuMG3692

if __name__ == "__main__":
    IP_ADDR = '192.168.10.182'
    
    try:
        device = AnritsuMG3692(IP_ADDR)
        
        if device.instrument:
            # 1. Set frequency to 5.8 GHz
            device.frequency = 5.8e9
            
            # 2. Set power to -30 dBm
            device.power = -30
            
            # 3. Turn on RF output
            device.on()
            
            # Checking for syntax or system errors
            device.check_error()
            
            # Example limit checking
            freq_limits = device.get_limit('frequency')
            print(f"Device frequency boundaries: {freq_limits}")
            
            print("Outputting... holding for 5 seconds")
            time.sleep(5)
            
            # 4. Turn off RF output
            device.off()
            
            # Close connection
            device.close()
    except Exception as e:
        print(f"Operation failed with error: {e}")