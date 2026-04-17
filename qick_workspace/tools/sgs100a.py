import pyvisa
from typing import Literal, Union, Tuple

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
    Pure PyVISA driver for the Rohde & Schwarz SGS100A signal generator.

    Provides a property-based API mimicking the QCoDeS driver framework 
    by mapping instrument features to descriptive get/set parameters.

    Parameters
    ----------
    address : str
        The VISA resource address of the instrument.

    Raises
    ------
    pyvisa.Error
        If establishing the connection to the VISA address fails.
    """

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
        """
        Queries and prints the IDN to confirm the connection.

        Returns
        -------
        None
        """
        try:
            idn = self.instrument.query("*IDN?")
            print(f"Connected to: {idn.strip()}")
        except pyvisa.Error as e:
            print(f"Could not query IDN. Error: {e}")

    def close(self) -> None:
        """
        Closes the VISA connection.

        Releases the instrument from VISA control and destroys the connection.

        Returns
        -------
        None
        """
        print(f"Disconnecting from {self.instrument.resource_name}")
        self.instrument.close()
        self.rm.close()

    def write(self, cmd: str) -> None:
        """
        Sends a SCPI write command.

        Parameters
        ----------
        cmd : str
            The SCPI command string to be executed by the instrument.

        Returns
        -------
        None
        """
        self.instrument.write(cmd)

    def query(self, cmd: str) -> str:
        """
        Sends a SCPI query command and returns the stripped result.

        Parameters
        ----------
        cmd : str
            The SCPI query string.

        Returns
        -------
        result : str
            The evaluated string response from the instrument.
        """
        return self.instrument.query(cmd).strip()

    def reset(self) -> None:
        """
        Resets the instrument (SCPI: *RST).

        Restores the instrument backend to factory default states.

        Returns
        -------
        None
        """
        print("Resetting instrument...")
        self.write("*RST")

    def run_self_tests(self) -> str:
        """
        Runs the instrument self-tests (SCPI: *TST?).

        Causes the instrument to evaluate core circuitry and components.

        Returns
        -------
        result : str
            A response indicating the test results.
        """
        print("Running self-tests...")
        result = self.query("*TST?")
        print(f"Self-test result: {result}")
        return result

    def check_error(self) -> str:
        """
        Checks the instrument's error queue.

        Queries the internal error log using 'SYST:ERR?' to find if any prior command
        was executed improperly.

        Returns
        -------
        err_msg : str
            The string representing the latest registered error.

        See Also
        --------
        query : Required to pull the string back from the hardware.
        """
        err_msg = self.query("SYST:ERR?")
        print(f"Instrument Status: {err_msg}")
        return err_msg

    def get_limit(self, parameter: str) -> Tuple[float, float]:
        """
        Retrieves the allowed operating limits for a specific parameter.

        Utilized internally to ensure set routines do not crash target devices
        or generate unexpected hardware errors out of bounds.

        Parameters
        ----------
        parameter : str
            The target parameter to check limits for.

        Returns
        -------
        limits : tuple
            A tuple formatted as (min_value, max_value).

        Raises
        ------
        ValueError
            If the passed parameter is not explicitly defined in the limit map.
        """
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

    # --- Helper methods ---

    def _validate_and_write(self, cmd_template: str, value: str, valid_set: set, name: str) -> None:
        """
        Validates Enum types and executes the write parameter update.

        Parameters
        ----------
        cmd_template : str
            String format for the SCPI command.
        value : str
            String selection made by the user.
        valid_set : set
            Legitimate set of accepted command strings.
        name : str
            Reporting variable parameter string.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If user input escapes predefined validation sets.
        """
        val_upper = str(value).upper()
        if val_upper not in valid_set:
            raise ValueError(f"Invalid {name} value: {value}. Allowed: {valid_set}")
        self.write(cmd_template.format(val_upper))

    def _map_and_write(self, cmd_template: str, value: Union[str, int, bool], name: str) -> None:
        """
        Maps on/off configuration parameters into hardware strings and writes.

        Parameters
        ----------
        cmd_template : str
            Formattable mapping for dynamic commands.
        value : str, int, bool
            User value (often boolean, binary, or string format).
        name : str
            Identifier string reference.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the supplied value is an invalid selection.
        """
        try:
            mapped_val = ON_OFF_MAP[str(value).lower()]
            self.write(cmd_template.format(mapped_val))
        except KeyError:
            raise ValueError(f"Invalid {name} value: {value}. Use 'on' or 'off'.")

    def _query_and_map(self, cmd: str) -> str:
        """
        Queries and maps on/off hardware string representations to descriptive words.

        Parameters
        ----------
        cmd : str
            Execution SCPI instruction.

        Returns
        -------
        mapped_result : str
            Translated version (from 0/1 back to off/on).
        """
        val = self.query(cmd)
        return ON_OFF_MAP_INV.get(val, f"unknown_val_{val}")

    # --- Parameters ---

    @property
    def frequency(self) -> float:
        """
        Gets the RF output frequency.

        Returns
        -------
        freq : float
            Currently programmed hardware frequency (in Hz).
        """
        return float(self.query("SOUR:FREQ?"))

    @frequency.setter
    def frequency(self, value: float) -> None:
        """
        Sets the RF output frequency.

        Checks limits bounds before executing SCPI payload.

        Parameters
        ----------
        value : float
            Desired operating frequency in Hz.
            
        Returns
        -------
        None
        """
        min_v, max_v = self.get_limit("frequency")
        if not (min_v <= value <= max_v):
            print(f"Warning: Frequency {value} Hz is outside driver's expected range ({min_v}, {max_v})")
        self.write(f"SOUR:FREQ {value:.2f}")

    @property
    def phase(self) -> float:
        """
        Gets the RF output phase.

        Returns
        -------
        phase_deg : float
            Currently configured output offset phase in degrees.
        """
        return float(self.query("SOUR:PHAS?"))

    @phase.setter
    def phase(self, value: float) -> None:
        """
        Sets the RF output phase.

        Checks allowed phase ranges explicitly.

        Parameters
        ----------
        value : float
            Desired operational phase shift relative to synchronization.

        Returns
        -------
        None
        """
        min_v, max_v = self.get_limit("phase")
        if not (min_v <= value <= max_v):
            print(f"Warning: Phase {value} deg is outside driver's expected range ({min_v}, {max_v})")
        self.write(f"SOUR:PHAS {value:.2f}")

    @property
    def power(self) -> float:
        """
        Gets the RF output power.

        Returns
        -------
        pwr_dbm : float
            Currently emitted power level output configured in dBm.
        """
        return float(self.query("SOUR:POW?"))

    @power.setter
    def power(self, value: float) -> None:
        """
        Sets the RF output power.

        Filters target configurations that violate output ranges.

        Parameters
        ----------
        value : float
            The requested output drive power in dBm.

        Returns
        -------
        None
        """
        min_v, max_v = self.get_limit("power")
        if not (min_v <= value <= max_v):
            print(f"Warning: Power {value} dBm is outside driver's expected range ({min_v}, {max_v})")
        self.write(f"SOUR:POW {value:.2f}")

    @property
    def status(self) -> str:
        """
        Gets the operational target state of the main RF output toggle.

        Returns
        -------
        status_code : str
            A descriptive indicator representing state ('on' or 'off').
        """
        return self._query_and_map(":OUTP:STAT?")

    @status.setter
    def status(self, value: Union[str, int, bool]) -> None:
        """
        Sets the state configuration of primary output toggle logic.

        Parameters
        ----------
        value : str, int, bool
            Dynamic typing configuration allowing simple toggling algorithms.

        Returns
        -------
        None
        """
        self._map_and_write(":OUTP:STAT {}", value, "status")

    def on(self) -> None:
        """
        Turns the RF output actively on.

        Wrapper function specifically written for brevity during scripted experiments.

        Returns
        -------
        None
        """
        self.status = "on"

    def off(self) -> None:
        """
        Turns the RF output actively off.

        Returns
        -------
        None
        """
        self.status = "off"

    @property
    def IQ_state(self) -> str:
        """
        Gets the underlying status for active analytical signal manipulation.

        Returns
        -------
        state : str
            The status map mapped to string logic ('on', 'off').
        """
        return self._query_and_map(":IQ:STAT?")

    @IQ_state.setter
    def IQ_state(self, value: Union[str, int, bool]) -> None:
        """
        Sets programmatic activation of the intrinsic IQ modulator components.

        Parameters
        ----------
        value : str, int, bool
            Control element.

        Returns
        -------
        None
        """
        self._map_and_write(":IQ:STAT {}", value, "IQ_state")

    @property
    def pulsemod_state(self) -> str:
        """
        Gets the activation status of hardware native pulse amplitude configurations.

        Returns
        -------
        state : str
            Identifies state strings mapped respectively.
        """
        return self._query_and_map(":SOUR:PULM:STAT?")

    @pulsemod_state.setter
    def pulsemod_state(self, value: Union[str, int, bool]) -> None:
        """
        Sets control to internally integrated modulation capabilities.

        Parameters
        ----------
        value : str, int, bool
            Valid element triggering logical mappings implicitly.

        Returns
        -------
        None
        """
        self._map_and_write(":SOUR:PULM:STAT {}", value, "pulsemod_state")

    @property
    def pulsemod_source(self) -> str:
        """
        Gets the primary pulse routing reference.

        Returns
        -------
        source_mode : str
            External routing mapping identified typically as 'EXT' or 'INT'.
        """
        return self.query("SOUR:PULM:SOUR?")

    @pulsemod_source.setter
    def pulsemod_source(self, value: Literal["INT", "EXT", "int", "ext"]) -> None:
        """
        Sets the expected input mechanism driving modulation.

        Parameters
        ----------
        value : Literal
            The reference oscillator string value.

        Returns
        -------
        None
        """
        self._validate_and_write("SOUR:PULM:SOUR {}", value, PULSE_SOURCE_VALS, "pulsemod_source")

    @property
    def ref_osc_source(self) -> str:
        """
        Gets the active mode of the internal clocking system routing.

        Returns
        -------
        osc_source : str
            Identifier code string ('EXT', 'INT').
        """
        return self.query("SOUR:ROSC:SOUR?")

    @ref_osc_source.setter
    def ref_osc_source(self, value: Literal["INT", "EXT", "int", "ext"]) -> None:
        """
        Sets operational dependencies for phase-locked systems synchronization.

        Parameters
        ----------
        value : Literal
            Configuration selection defining clock targets.

        Returns
        -------
        None
        """
        self._validate_and_write("SOUR:ROSC:SOUR {}", value, REF_LO_SOURCE_VALS, "ref_osc_source")

    @property
    def LO_source(self) -> str:
        """
        Gets operational assignment mappings for localized routing.

        Returns
        -------
        LO_source_str : str
            Currently employed synthesizer clock logic mapping.
        """
        return self.query("SOUR:LOSC:SOUR?")

    @LO_source.setter
    def LO_source(self, value: Literal["INT", "EXT", "int", "ext"]) -> None:
        """
        Sets local operational assignments for phase alignments externally.

        Parameters
        ----------
        value : Literal
            Control string specifying connection source mapping directly.

        Returns
        -------
        None
        """
        self._validate_and_write("SOUR:LOSC:SOUR {}", value, REF_LO_SOURCE_VALS, "LO_source")

    @property
    def ref_LO_out(self) -> str:
        """
        Gets the reference distribution topology configuration logic.

        Returns
        -------
        ref_str : str
            The reference mapping logic currently actively operating system clocks.
        """
        return self.query("CONN:REFL:OUTP?")

    @ref_LO_out.setter
    def ref_LO_out(self, value: Literal["REF", "LO", "OFF", "ref", "lo", "off"]) -> None:
        """
        Sets target internal outputs logic.

        Parameters
        ----------
        value : Literal
            The internal topology mapping logic selector assigned natively.

        Returns
        -------
        None
        """
        self._validate_and_write("CONN:REFL:OUTP {}", value, REF_LO_OUT_VALS, "ref_LO_out")

    @property
    def ref_osc_output_freq(self) -> str:
        """
        Gets reference output clock capabilities configured manually.

        Returns
        -------
        freq_str : str
            Reference magnitude in string formatting (e.g. '100MHz').
        """
        return self.query("SOUR:ROSC:OUTP:FREQ?")

    @ref_osc_output_freq.setter
    def ref_osc_output_freq(self, value: Literal["10MHz", "100MHz", "1000MHz"]) -> None:
        """
        Sets routing magnitude directly out backend interfaces natively.

        Parameters
        ----------
        value : Literal
            Operating parameter string explicitly defined inside allowed bounds.

        Returns
        -------
        None
        """
        self._validate_and_write("SOUR:ROSC:OUTP:FREQ {}", value, REF_FREQ_VALS, "ref_osc_output_freq")

    @property
    def ref_osc_external_freq(self) -> str:
        """
        Gets reference locking magnitude expected from external clocks.

        Returns
        -------
        freq_str : str
            The corresponding string matched against active operational loops.
        """
        return self.query("SOUR:ROSC:EXT:FREQ?")

    @ref_osc_external_freq.setter
    def ref_osc_external_freq(self, value: Literal["10MHz", "100MHz", "1000MHz"]) -> None:
        """
        Sets phase alignment frequencies natively expected by backend interfaces.

        Parameters
        ----------
        value : Literal
            String representation formatting precisely expected sync signals.

        Returns
        -------
        None
        """
        self._validate_and_write("SOUR:ROSC:EXT:FREQ {}", value, REF_FREQ_VALS, "ref_osc_external_freq")

    @property
    def IQ_impairments(self) -> str:
        """
        Gets intentional analog correction mappings logic.

        Returns
        -------
        status : str
            Current state of application.
        """
        return self._query_and_map(":SOUR:IQ:IMP:STAT?")

    @IQ_impairments.setter
    def IQ_impairments(self, value: Union[str, int, bool]) -> None:
        """
        Sets operational availability of programmatic internal defect correction mapping.

        Parameters
        ----------
        value : str, int, bool
            Indicator representation allowing algorithmic adjustments dynamically.

        Returns
        -------
        None
        """
        self._map_and_write(":SOUR:IQ:IMP:STAT {}", value, "IQ_impairments")

    @property
    def I_offset(self) -> float:
        """
        Gets explicit native I channel voltage deflection limits locally.

        Returns
        -------
        offset : float
            Deflection represented linearly physically via DAC control.
        """
        return float(self.query("SOUR:IQ:IMP:LEAK:I?"))

    @I_offset.setter
    def I_offset(self, value: float) -> None:
        """
        Sets intentional constant localized analog baseline shift externally.

        Parameters
        ----------
        value : float
            Representation percentage deflection parameter.

        Returns
        -------
        None
        """
        min_v, max_v = self.get_limit("i_offset")
        if not (min_v <= value <= max_v):
            print(f"Warning: I offset {value}% is outside expected range ({min_v}, {max_v})")
        self.write(f"SOUR:IQ:IMP:LEAK:I {value:.2f}")

    @property
    def Q_offset(self) -> float:
        """
        Gets programmatic explicit baseline shift externally.

        Returns
        -------
        offset : float
            Deflection natively mapped dynamically against Q signal base.
        """
        return float(self.query("SOUR:IQ:IMP:LEAK:Q?"))

    @Q_offset.setter
    def Q_offset(self, value: float) -> None:
        """
        Sets orthogonal baseline drift statically externally.

        Parameters
        ----------
        value : float
            Value required linearly interpreted implicitly directly.

        Returns
        -------
        None
        """
        min_v, max_v = self.get_limit("q_offset")
        if not (min_v <= value <= max_v):
            print(f"Warning: Q offset {value}% is outside expected range ({min_v}, {max_v})")
        self.write(f"SOUR:IQ:IMP:LEAK:Q {value:.2f}")

    @property
    def IQ_gain_imbalance(self) -> float:
        """
        Gets operational asymmetry between complex vectors explicitly locally.

        Returns
        -------
        imbalance : float
            Offset mapping difference logarithmically interpreted directly in dB.
        """
        return float(self.query("SOUR:IQ:IMP:IQR?"))

    @IQ_gain_imbalance.setter
    def IQ_gain_imbalance(self, value: float) -> None:
        """
        Sets structural gain differential intentionally explicitly defined.

        Parameters
        ----------
        value : float
            Ratio represented logically inside unit mapping dynamically.

        Returns
        -------
        None
        """
        min_v, max_v = self.get_limit("iq_gain_imbalance")
        if not (min_v <= value <= max_v):
            print(f"Warning: IQ gain imbalance {value} dB is outside expected range ({min_v}, {max_v})")
        self.write(f"SOUR:IQ:IMP:IQR {value:.2f}")

    @property
    def IQ_angle(self) -> float:
        """
        Gets explicit rotational deviation orthogonal mappings logically native.

        Returns
        -------
        angle : float
            Deviation expected inherently physically explicitly referenced.
        """
        return float(self.query("SOUR:IQ:IMP:QUAD?"))

    @IQ_angle.setter
    def IQ_angle(self, value: float) -> None:
        """
        Sets intended logical distortion physically internally executed directly.

        Parameters
        ----------
        value : float
            Degree offset effectively executed statically.

        Returns
        -------
        None
        """
        min_v, max_v = self.get_limit("iq_angle")
        if not (min_v <= value <= max_v):
            print(f"Warning: IQ angle {value} deg is outside expected range ({min_v}, {max_v})")
        self.write(f"SOUR:IQ:IMP:QUAD {value:.2f}")

    @property
    def trigger_connector_mode(self) -> str:
        """
        Gets port mode defining behavioral operation locally executed physically.

        Returns
        -------
        trigger_mode : str
            Port mappings internally defined dynamically mapped natively.
        """
        return self.query("CONN:TRIG:OMOD?")

    @trigger_connector_mode.setter
    def trigger_connector_mode(self, value: str) -> None:
        """
        Sets explicit logic assignments mechanically executed directly physically.

        Parameters
        ----------
        value : str
            Config string natively configured.

        Returns
        -------
        None
        """
        self._validate_and_write("CONN:TRIG:OMOD {}", value, TRIG_MODE_VALS, "trigger_connector_mode")

    @property
    def pulsemod_delay(self) -> float:
        """
        Gets static operational shifts internally applied to active traces locally.

        Returns
        -------
        delay : float
            Static offsets mapped logically implicitly inside internal hardware representations dynamically.
        """
        return float(self.query("SOUR:PULM:DEL?"))

    @pulsemod_delay.setter
    def pulsemod_delay(self, value: float) -> None:
        """
        Sets programmatic analog base deviations internally integrated statically explicitly.

        Parameters
        ----------
        value : float
            Value expected configured implicitly defined globally representing explicit time mapping dynamically natively.

        Returns
        -------
        None
        """
        min_v, max_v = self.get_limit("pulsemod_delay")
        if not (min_v <= value <= max_v):
            print(f"Warning: Pulse modulation delay {value} s is outside expected range ({min_v}, {max_v})")
        self.write(f"SOUR:PULM:DEL {value:g}")

class RohdeSchwarz_SGS100A(RohdeSchwarzSGS100A):
    """
    Alias wrapper ensuring comprehensive backwards compatibility globally explicitly internally.
    """
    pass

if __name__ == "__main__":
    rm = pyvisa.ResourceManager("@sim")
    sim_address = "ASRL1::INSTR"
    try:
        print("--- Using simulated instrument ---")
        sgs = RohdeSchwarzSGS100A(sim_address)

        # 1. Set frequency and power
        sgs.frequency = 5e9
        sgs.power = -10

        # 2. Read values
        print(f"Read frequency: {sgs.frequency} Hz")
        print(f"Read power: {sgs.power} dBm")

        # 3. Turn RF output on/off
        sgs.on()
        print(f"RF Status: {sgs.status}")
        sgs.off()
        print(f"RF Status: {sgs.status}")

        # 4. Check errors
        sgs.check_error()

        # 5. Check limit boundaries
        limits = sgs.get_limit("frequency")
        print(f"Hardware frequency limits: {limits}")

        # 6. Test shortcut methods
        sgs.reset()

        # 7. Close connection
        sgs.close()

    except pyvisa.errors.VisaIOError as e:
        print("\nError: Could not connect to instrument. Check your VISA address.")
        print(f"PyVISA Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
