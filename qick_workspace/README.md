# This repostory is the code for Qick zcu tprocv2 measurent scrip


## single qubit pyscrip v1_1
This py scrip file is for simply the measurement process
from s00xscrip.py import mems_func
than `mems_func.run(average_number)`

If you want to see the data
`mems_func.plot()`

If you want to save the data to Labber
`mems_func.saveLabber(qubit_idx, yokocurrent)`

## new feature
Support live plot function which is combine in run()
`mems_func.run(average_number, liveplot=True)`

