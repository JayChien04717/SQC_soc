import pyvisa as visa
import numpy as np
import time
from tqdm.auto import tqdm


class YOKOGS200:
    """Legacy Yokogawa GS200 driver with SetVoltage/GetVoltage API."""

    def __init__(self, VISAaddress, rm):
        self.VISAaddress = VISAaddress
        try:
            self.session = rm.open_resource(VISAaddress)
        except visa.Error as ex:
            import sys
            sys.stderr.write("Couldn't connect to '%s', exiting now..." % VISAaddress)
            sys.exit()
        self.voltage_ramp_step = 1e-5
        self.current_ramp_step = 1e-8
        self.ramp_interval = 0.01

    def OutputOn(self):
        self.session.write("OUTPut 1")

    def OutputOff(self):
        self.session.write("OUTPut 0")

    def SetVoltage(self, voltage):
        start = self.GetVoltage()
        stop = voltage
        steps = max(1, round(abs(stop - start) / self.voltage_ramp_step))
        tempvolts = np.linspace(start, stop, num=steps + 1, endpoint=True)
        self.OutputOn()
        for tempvolt in tqdm(tempvolts, desc="Setting Voltage", leave=False):
            self.session.write(":SOURce:LEVel:AUTO %.8f" % tempvolt)
            time.sleep(self.ramp_interval)

    def SetCurrent(self, current):
        start = self.GetCurrent()
        stop = current
        steps = max(1, round(abs(stop - start) / self.current_ramp_step))
        tempcurrents = np.linspace(start, stop, num=steps)
        self.OutputOn()
        for tempcurrent in tqdm(tempcurrents, desc="Setting Current", leave=False):
            self.session.write(":SOURce:LEVel:AUTO %.8f" % tempcurrent)
            time.sleep(self.ramp_interval)

    def SetMode(self, mode):
        import sys
        if not (mode == "voltage" or mode == "current"):
            sys.stderr.write("Unknown output mode %s." % mode)
            return
        self.session.write("SOURce:FUNCtion %s" % mode)

    def GetVoltage(self):
        self.session.write("SOURce:FUNCtion VOLTage")
        self.session.write("SOURce:LEVel?")
        result = self.session.read()
        return float(result.rstrip("\n"))

    def GetCurrent(self):
        self.session.write("SOURce:FUNCtion CURRent")
        self.session.write("SOURce:LEVel?")
        result = self.session.read()
        return float(result.rstrip("\n"))

    def GetValue(self):
        mode = self.GetMode()
        if mode == "voltage":
            value = self.GetVoltage()
            return dict(unit="V", value=value)
        else:
            value = self.GetCurrent()
            return dict(unit="A", value=value)

    def GetMode(self):
        self.session.write("SOURce:FUNCtion?")
        result = self.session.read()
        result = result.rstrip("\n")
        if result == "VOLT":
            return "voltage"
        else:
            return "current"
