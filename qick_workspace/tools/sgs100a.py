import pyvisa
from typing import Literal, Union

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

# Sets for 'Enum' validation
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
    """
    This is a pure PyVISA driver for the Rohde & Schwarz SGS100A signal generator.

    It mimics the QCoDeS driver API, using properties (get/set) to
    control instrument parameters.
    """

    def __init__(self, address: str) -> None:
        """
        Initializes the instrument and connects.

        :param address: The VISA resource address of the instrument
                        (e.g., "TCPIP0::192.168.1.100::inst0::INSTR")
        """
        self.rm = pyvisa.ResourceManager()
        try:
            self.instrument = self.rm.open_resource(address)
        except pyvisa.Error as e:
            print(f"Could not connect to {address}. Error: {e}")
            raise

        # Set terminations based on QCoDeS driver
        self.instrument.read_termination = "\n"
        self.instrument.write_termination = "\n"

        # Print IDN message on connection
        self.connect_message()

    def connect_message(self) -> None:
        """Queries and prints the IDN to confirm connection."""
        try:
            idn = self.instrument.query("*IDN?")
            print(f"Connected to: {idn.strip()}")
        except pyvisa.Error as e:
            print(f"Could not query IDN. Error: {e}")

    def close(self) -> None:
        """Closes the VISA connection."""
        print(f"Disconnecting from {self.instrument.resource_name}")
        self.instrument.close()
        self.rm.close()

    def write(self, cmd: str) -> None:
        """Sends a SCPI write command."""
        self.instrument.write(cmd)

    def query(self, cmd: str) -> str:
        """Sends a SCPI query command and returns the stripped result."""
        return self.instrument.query(cmd).strip()

    def reset(self) -> None:
        """Resets the instrument (SCPI: *RST)"""
        print("Resetting instrument...")
        self.write("*RST")

    def run_self_tests(self) -> str:
        """Runs the instrument self-tests (SCPI: *TST?)"""
        print("Running self-tests...")
        result = self.query("*TST?")
        print(f"Self-test result: {result}")
        return result

    # --- Helper methods ---
    def _validate_and_write(
        self, cmd_template: str, value: str, valid_set: set, name: str
    ):
        """Helper function: Validates Enum types and writes."""
        val_upper = str(value).upper()
        if val_upper not in valid_set:
            raise ValueError(f"Invalid {name} value: {value}. Allowed: {valid_set}")
        # R&S instruments are generally case-insensitive, but using uppercase is good practice
        self.write(cmd_template.format(val_upper))

    def _map_and_write(
        self, cmd_template: str, value: Union[str, int, bool], name: str
    ):
        """Helper function: Maps on/off values and writes."""
        try:
            mapped_val = ON_OFF_MAP[str(value).lower()]
            self.write(cmd_template.format(mapped_val))
        except KeyError:
            raise ValueError(f"Invalid {name} value: {value}. Use 'on' or 'off'.")

    def _query_and_map(self, cmd: str) -> str:
        """Helper function: Queries and maps on/off values."""
        val = self.query(cmd)
        return ON_OFF_MAP_INV.get(val, f"unknown_val_{val}")

    # --- Parameters ---

    @property
    def frequency(self) -> float:
        """(Hz) Gets the RF frequency. (SCPI: SOUR:FREQ?)"""
        return float(self.query("SOUR:FREQ?"))

    @frequency.setter
    def frequency(self, value: float):
        """(Hz) Sets the RF frequency. (SCPI: SOUR:FREQ {:.2f}) [Range: 1e6 to 20e9]"""
        if not (1e6 <= value <= 20e9):
            print(
                f"Warning: Frequency {value} Hz is outside driver's expected range (1e6, 20e9)"
            )
        self.write(f"SOUR:FREQ {value:.2f}")

    @property
    def phase(self) -> float:
        """(deg) Gets the RF phase. (SCPI: SOUR:PHAS?)"""
        return float(self.query("SOUR:PHAS?"))

    @phase.setter
    def phase(self, value: float):
        """(deg) Sets the RF phase. (SCPI: SOUR:PHAS {:.2f}) [Range: 0 to 360]"""
        if not (0 <= value <= 360):
            print(
                f"Warning: Phase {value} deg is outside driver's expected range (0, 360)"
            )
        self.write(f"SOUR:PHAS {value:.2f}")

    @property
    def power(self) -> float:
        """(dBm) Gets the RF power. (SCPI: SOUR:POW?)"""
        return float(self.query("SOUR:POW?"))

    @power.setter
    def power(self, value: float):
        """(dBm) Sets the RF power. (SCPI: SOUR:POW {:.2f}) [Range: -120 to 25]"""
        if not (-120 <= value <= 25):
            print(
                f"Warning: Power {value} dBm is outside driver's expected range (-120, 25)"
            )
        self.write(f"SOUR:POW {value:.2f}")

    @property
    def status(self) -> str:
        """('on'/'off') Gets the RF output status. (SCPI: :OUTP:STAT?)"""
        return self._query_and_map(":OUTP:STAT?")

    @status.setter
    def status(self, value: Union[str, int, bool]):
        """('on'/'off') Sets the RF output status. (SCPI: :OUTP:STAT {})"""
        self._map_and_write(":OUTP:STAT {}", value, "status")

    # --- Shortcut methods ---
    def on(self) -> None:
        """Turns the RF output on."""
        self.status = "on"

    def off(self) -> None:
        """Turns the RF output off."""
        self.status = "off"

    @property
    def IQ_state(self) -> str:
        """('on'/'off') Gets the IQ modulation status. (SCPI: :IQ:STAT?)"""
        return self._query_and_map(":IQ:STAT?")

    @IQ_state.setter
    def IQ_state(self, value: Union[str, int, bool]):
        """('on'/'off') Sets the IQ modulation status. (SCPI: :IQ:STAT {})"""
        self._map_and_write(":IQ:STAT {}", value, "IQ_state")

    @property
    def pulsemod_state(self) -> str:
        """('on'/'off') Gets the pulse modulation status. (SCPI: :SOUR:PULM:STAT?)"""
        return self._query_and_map(":SOUR:PULM:STAT?")

    @pulsemod_state.setter
    def pulsemod_state(self, value: Union[str, int, bool]):
        """('on'/'off') Sets the pulse modulation status. (SCPI: :SOUR:PULM:STAT {})"""
        self._map_and_write(":SOUR:PULM:STAT {}", value, "pulsemod_state")

    @property
    def pulsemod_source(self) -> str:
        """('INT'/'EXT') Gets the pulse modulation source. (SCPI: SOUR:PULM:SOUR?)"""
        return self.query("SOUR:PULM:SOUR?")

    @pulsemod_source.setter
    def pulsemod_source(self, value: Literal["INT", "EXT", "int", "ext"]):
        """('INT'/'EXT') Sets the pulse modulation source. (SCPI: SOUR:PULM:SOUR {})"""
        self._validate_and_write(
            "SOUR:PULM:SOUR {}", value, PULSE_SOURCE_VALS, "pulsemod_source"
        )

    @property
    def ref_osc_source(self) -> str:
        """('INT'/'EXT') Gets the reference oscillator source. (SCPI: SOUR:ROSC:SOUR?)"""
        return self.query("SOUR:ROSC:SOUR?")

    @ref_osc_source.setter
    def ref_osc_source(self, value: Literal["INT", "EXT", "int", "ext"]):
        """('INT'/'EXT') Sets the reference oscillator source. (SCPI: SOUR:ROSC:SOUR {})"""
        self._validate_and_write(
            "SOUR:ROSC:SOUR {}", value, REF_LO_SOURCE_VALS, "ref_osc_source"
        )

    @property
    def LO_source(self) -> str:
        """('INT'/'EXT') Gets the local oscillator (LO) source. (SCPI: SOUR:LOSC:SOUR?)"""
        return self.query("SOUR:LOSC:SOUR?")

    @LO_source.setter
    def LO_source(self, value: Literal["INT", "EXT", "int", "ext"]):
        """('INT'/'EXT') Sets the local oscillator (LO) source. (SCPI: SOUR:LOSC:SOUR {})"""
        self._validate_and_write(
            "SOUR:LOSC:SOUR {}", value, REF_LO_SOURCE_VALS, "LO_source"
        )

    @property
    def ref_LO_out(self) -> str:
        """('REF'/'LO'/'OFF') Gets the REF/LO output. (SCPI: CONN:REFL:OUTP?)"""
        return self.query("CONN:REFL:OUTP?")

    @ref_LO_out.setter
    def ref_LO_out(self, value: Literal["REF", "LO", "OFF", "ref", "lo", "off"]):
        """('REF'/'LO'/'OFF') Sets the REF/LO output. (SCPI: CONN:REFL:OUTP {})"""
        self._validate_and_write(
            "CONN:REFL:OUTP {}", value, REF_LO_OUT_VALS, "ref_LO_out"
        )

    @property
    def ref_osc_output_freq(self) -> str:
        """('10MHz'/'100MHz'/'1000MHz') Gets the reference oscillator output frequency. (SCPI: SOUR:ROSC:OUTP:FREQ?)"""
        return self.query("SOUR:ROSC:OUTP:FREQ?")

    @ref_osc_output_freq.setter
    def ref_osc_output_freq(self, value: Literal["10MHz", "100MHz", "1000MHz"]):
        """('10MHz'/'100MHz'/'1000MHz') Sets the reference oscillator output frequency. (SCPI: SOUR:ROSC:OUTP:FREQ {})"""
        self._validate_and_write(
            "SOUR:ROSC:OUTP:FREQ {}", value, REF_FREQ_VALS, "ref_osc_output_freq"
        )

    @property
    def ref_osc_external_freq(self) -> str:
        """('10MHz'/'100MHz'/'1000MHz') Gets the external reference oscillator frequency. (SCPI: SOUR:ROSC:EXT:FREQ?)"""
        return self.query("SOUR:ROSC:EXT:FREQ?")

    @ref_osc_external_freq.setter
    def ref_osc_external_freq(self, value: Literal["10MHz", "100MHz", "1000MHz"]):
        """('10MHz'/'100MHz'/'1000MHz') Sets the external reference oscillator frequency. (SCPI: SOUR:ROSC:EXT:FREQ {})"""
        self._validate_and_write(
            "SOUR:ROSC:EXT:FREQ {}", value, REF_FREQ_VALS, "ref_osc_external_freq"
        )

    @property
    def IQ_impairments(self) -> str:
        """('on'/'off') Gets the IQ impairments status. (SCPI: :SOUR:IQ:IMP:STAT?)"""
        return self._query_and_map(":SOUR:IQ:IMP:STAT?")

    @IQ_impairments.setter
    def IQ_impairments(self, value: Union[str, int, bool]):
        """('on'/'off') Sets the IQ impairments status. (SCPI: :SOUR:IQ:IMP:STAT {})"""
        self._map_and_write(":SOUR:IQ:IMP:STAT {}", value, "IQ_impairments")

    @property
    def I_offset(self) -> float:
        """(%) Gets the I channel offset. (SCPI: SOUR:IQ:IMP:LEAK:I?)"""
        return float(self.query("SOUR:IQ:IMP:LEAK:I?"))

    @I_offset.setter
    def I_offset(self, value: float):
        """(%) Sets the I channel offset. (SCPI: SOUR:IQ:IMP:LEAK:I {:.2f}) [Range: -10 to 10]"""
        self.write(f"SOUR:IQ:IMP:LEAK:I {value:.2f}")

    @property
    def Q_offset(self) -> float:
        """(%) Gets the Q channel offset. (SCPI: SOUR:IQ:IMP:LEAK:Q?)"""
        return float(self.query("SOUR:IQ:IMP:LEAK:Q?"))

    @Q_offset.setter
    def Q_offset(self, value: float):
        """(%) Sets the Q channel offset. (SCPI: SOUR:IQ:IMP:LEAK:Q {:.2f}) [Range: -10 to 10]"""
        self.write(f"SOUR:IQ:IMP:LEAK:Q {value:.2f}")

    @property
    def IQ_gain_imbalance(self) -> float:
        """(dB) Gets the IQ gain imbalance. (SCPI: SOUR:IQ:IMP:IQR?)"""
        return float(self.query("SOUR:IQ:IMP:IQR?"))

    @IQ_gain_imbalance.setter
    def IQ_gain_imbalance(self, value: float):
        """(dB) Sets the IQ gain imbalance. (SCPI: SOUR:IQ:IMP:IQR {:.2f}) [Range: -1 to 1]"""
        self.write(f"SOUR:IQ:IMP:IQR {value:.2f}")

    @property
    def IQ_angle(self) -> float:
        """(deg) Gets the IQ angle offset. (SCPI: SOUR:IQ:IMP:QUAD?)"""
        return float(self.query("SOUR:IQ:IMP:QUAD?"))

    @IQ_angle.setter
    def IQ_angle(self, value: float):
        """(deg) Sets the IQ angle offset. (SCPI: SOUR:IQ:IMP:QUAD {:.2f}) [Range: -8 to 8]"""
        self.write(f"SOUR:IQ:IMP:QUAD {value:.2f}")

    @property
    def trigger_connector_mode(self) -> str:
        """Gets the [TRIG] connector mode. (SCPI: CONN:TRIG:OMOD?)"""
        return self.query("CONN:TRIG:OMOD?")

    @trigger_connector_mode.setter
    def trigger_connector_mode(self, value: str):
        """Sets the [TRIG] connector mode. (SCPI: CONN:TRIG:OMOD {})"""
        self._validate_and_write(
            "CONN:TRIG:OMOD {}", value, TRIG_MODE_VALS, "trigger_connector_mode"
        )

    @property
    def pulsemod_delay(self) -> float:
        """(s) Gets the pulse delay. (SCPI: SOUR:PULM:DEL?)"""
        return float(self.query("SOUR:PULM:DEL?"))

    @pulsemod_delay.setter
    def pulsemod_delay(self, value: float):
        """(s) Sets the pulse delay. (SCPI: SOUR:PULM:DEL {:g}) [Range: 0 to 100]"""
        self.write(f"SOUR:PULM:DEL {value:g}")

    # ... (Continue adding properties and setters for the remaining parameters) ...
    # For brevity, the remaining pulsemod parameters are omitted,
    # but you can continue adding them following the 'pulsemod_delay' format:
    # - pulsemod_double_delay
    # - pulsemod_double_width
    # - pulsemod_mode (use _validate_and_write and PULSE_MODE_VALS)
    # - pulsemod_period
    # - pulsemod_polarity (use _validate_and_write and POLARITY_VALS)
    # - pulsemod_trig_ext_gate_polarity (use _validate_and_write and POLARITY_VALS)
    # - pulsemod_trig_ext_impedance (use _validate_and_write and IMPEDANCE_VALS)
    # - pulsemod_trig_ext_slope (use _validate_and_write and SLOPE_VALS)
    # - pulsemod_trig_mode (use _validate_and_write and TRIG_MODE_EXT_VALS)
    # - pulsemod_width
    # - operation_mode (use _validate_and_write and OP_MODE_VALS)


# Alias defined in the QCoDeS driver
class RohdeSchwarz_SGS100A(RohdeSchwarzSGS100A):
    pass


# --- Example Usage ---
if __name__ == "__main__":
    # Note: Replace 'TCPIP0::...::INSTR' with your instrument's real VISA address
    # We use a simulation backend for testing
    rm = pyvisa.ResourceManager("@sim")
    # A simulated address
    sim_address = "ASRL1::INSTR"
    try:
        # Try to open a real instrument (comment out the line below if it fails)
        # inst_addr = "TCPIP0::192.168.1.100::inst0::INSTR"
        # sgs = RohdeSchwarzSGS100A(inst_addr)

        # Use the simulated instrument
        print("--- Using simulated instrument ---")
        sgs = RohdeSchwarzSGS100A(sim_address)

        # --- Test setting and reading ---

        # 1. Set frequency and power
        sgs.frequency = 5e9  # Set 5 GHz
        sgs.power = -10  # Set -10 dBm

        # 2. Read values (in simulation, this will return default values or 0)
        print(f"Read frequency: {sgs.frequency} Hz")
        print(f"Read power: {sgs.power} dBm")

        # 3. Turn RF output on/off
        sgs.on()
        print(f"RF Status: {sgs.status}")

        sgs.off()
        print(f"RF Status: {sgs.status}")

        # 4. Test Enum parameters
        sgs.ref_osc_source = "EXT"
        print(f"Reference source: {sgs.ref_osc_source}")

        # 5. Test invalid input
        try:
            sgs.ref_osc_source = "INVALID_VALUE"
        except ValueError as e:
            print(f"Successfully caught error: {e}")

        # 6. Test shortcut methods
        sgs.reset()

        # 7. Close connection
        sgs.close()

    except pyvisa.errors.VisaIOError as e:
        print(f"\nError: Could not connect to instrument. Check your VISA address.")
        print(f"PyVISA Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
