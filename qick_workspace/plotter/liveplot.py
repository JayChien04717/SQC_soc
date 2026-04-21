import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
import pyvisa
from IPython.display import display, clear_output, update_display
from tqdm.auto import tqdm
import math

# ===================================================================
# User Imports
# ===================================================================
from ..tools.YOKOGS200 import YOKOGS200
from ..tools.system_tool import auto_unit

# ===================================================================
# 1. The Facade Function
# ===================================================================


def liveplotfun(
    prog=None,  # Applicable to: sw_avg, yoko
    soc=None,  # Applicable to: sw_avg, yoko
    py_avg=1,  # Applicable to: ALL
    x_axis_vals=None,  # Applicable to: sw_avg (as X-axis), yoko (as inner loop X-axis)
    y_axis_vals=None,  # Applicable to: sw_avg (as Y-axis for 2D plot), yoko (as outer loop Yoko values)
    x_label="X Axis",
    y_label="Y Axis",
    title_prefix="Experiment",
    # --- Yoko-specific parameters ---
    yoko_inst_addr=None,
    yoko_mode="current",
    yoko_voltage_ramp_step=1e-5,
    yoko_current_ramp_step=1e-8,
    yoko_ramp_interval=0.01,
    # --- Parameter Scan specific ---
    scan_x_axis=None,  # If provided, enables parameter scan mode
    scan_y_axis=None,  # If provided alongside scan_x_axis, enables 2D parameter scan mode
    get_prog_callback=None,  # Callback function to dynamically generate programs
    # --- General ---
    show_final_plot=True,
    iq_process="abs",  # "abs" or "real": how to convert complex IQ data for plotting
):
    """
    General-purpose live plotter (Facade pattern).

    Dispatches to one of four specialized internal routines based on the
    combination of arguments supplied:

    - **Yoko sweep** (``yoko_inst_addr`` is set): outer loop steps a
      Yokogawa source while the inner QICK program sweeps ``x_axis_vals``.
    - **2D parameter scan** (``scan_x_axis`` and ``scan_y_axis`` both set):
      a callback generates a fresh program for every (x, y) grid point.
    - **1D parameter scan** (only ``scan_x_axis`` set): a callback generates
      a fresh program for each x point.
    - **Software averaging** (default): repeats a fixed program ``py_avg``
      times and accumulates a running average.

    Parameters
    ----------
    prog : object, optional
        Pre-compiled QICK program (used in sw_avg and yoko modes).
    soc : object, optional
        QICK SoC instance required for hardware acquisition.
    py_avg : int, optional
        Number of software averaging repetitions.  Default is ``1``.
    x_axis_vals : np.ndarray, optional
        X-axis values for sw_avg mode (1-D or fast axis of 2-D); or the
        inner sweep array for yoko mode.
    y_axis_vals : np.ndarray, optional
        Y-axis values for 2-D sw_avg mode; or Yoko output values for yoko
        mode.
    x_label : str, optional
        X-axis label.  Default is ``"X Axis"``.
    y_label : str, optional
        Y-axis label.  Default is ``"Y Axis"``.
    title_prefix : str, optional
        Prefix shown in the figure title.  Default is ``"Experiment"``.
    yoko_inst_addr : str, optional
        VISA address of the Yokogawa source (e.g. ``"GPIB0::1::INSTR"``).
        When set, selects Yoko sweep mode.
    yoko_mode : str, optional
        ``"current"`` or ``"voltage"``.  Default is ``"current"``.
    scan_x_axis : np.ndarray, optional
        Parameter values for the fast (x) axis of a parameter scan.
    scan_y_axis : np.ndarray, optional
        Parameter values for the slow (y) axis of a 2-D parameter scan.
    get_prog_callback : callable, optional
        Function called with ``(x_val,)`` (1-D) or ``(x_val, y_val)`` (2-D)
        that returns a ready-to-run QICK program.
    show_final_plot : bool, optional
        Whether to render a final static figure after acquisition.
        Default is ``True``.
    iq_process : str, optional
        ``"abs"`` for amplitude or ``"real"`` for the I-quadrature.
        Default is ``"abs"``.

    Returns
    -------
    iqdata : np.ndarray
        Averaged complex IQ data (shape depends on mode).
    interrupted : bool
        ``True`` if the loop was stopped early by ``KeyboardInterrupt``.
    n_done : int
        Number of completed averages or Yoko steps.

    Raises
    ------
    ValueError
        If ``yoko_inst_addr`` is set but ``y_axis_vals`` is ``None``, or if
        ``scan_x_axis`` is set but ``get_prog_callback`` is ``None``.
    """

    # Mode 1: Yoko Parameter Sweep
    if yoko_inst_addr is not None:
        if y_axis_vals is None:
            raise ValueError("y_axis_vals must be provided for a Yoko sweep.")
        return _liveplot_sweep_yoko(
            prog=prog,
            soc=soc,
            py_avg=py_avg,
            x_axis_vals=x_axis_vals,
            y_axis_vals_yoko=y_axis_vals,
            yoko_inst_addr=yoko_inst_addr,
            yoko_mode=yoko_mode,
            yoko_voltage_ramp_step=yoko_voltage_ramp_step,
            yoko_current_ramp_step=yoko_current_ramp_step,
            yoko_ramp_interval=yoko_ramp_interval,
            x_label=x_label,
            y_label=y_label,
            title_prefix=title_prefix,
            iq_process=iq_process,
        )

    # Mode 2: Parameter Scan (1D or 2D)
    elif scan_x_axis is not None:
        if get_prog_callback is None:
            raise ValueError("get_prog_callback must be provided for parameter scan.")

        if scan_y_axis is not None:
            return _liveplot_2d_scan(
                soc=soc,
                py_avg=py_avg,
                scan_x_axis=scan_x_axis,
                scan_y_axis=scan_y_axis,
                get_prog_callback=get_prog_callback,
                x_label=x_label,
                y_label=y_label,
                title_prefix=title_prefix,
                show_final_plot=show_final_plot,
                iq_process=iq_process,
            )
        else:
            return _liveplot_1d_scan(
                soc=soc,
                py_avg=py_avg,
                scan_x_axis=scan_x_axis,
                get_prog_callback=get_prog_callback,
                x_label=x_label,
                title_prefix=title_prefix,
                show_final_plot=show_final_plot,
                iq_process=iq_process,
            )

    # Mode 3: Software Averaging (Default 1D or 2D repeat)
    else:
        return _liveplot_sw_avg(
            prog=prog,
            soc=soc,
            py_avg=py_avg,
            x_axis_vals=x_axis_vals,
            y_axis_vals=y_axis_vals,
            x_label=x_label,
            y_label=y_label,
            title_prefix=title_prefix,
            show_final_plot=show_final_plot,
            iq_process=iq_process,
        )


# ===================================================================
# 2. Internal Function: Software Averaging
# ===================================================================
def _liveplot_sw_avg(
    prog,
    soc,
    py_avg,
    x_axis_vals,
    y_axis_vals=None,
    x_label="X Axis",
    y_label="Y Axis",
    title_prefix="Experiment",
    show_final_plot=False,
    iq_process="abs",
):
    """
    Execute a software-averaged live plot using a dedicated plotter thread.

    Implements a producer-consumer pattern with a LIFO queue (max size 1)
    so that rendering never stalls data acquisition: when the queue is full,
    the stale frame is dropped and the latest frame takes its place.

    Parameters
    ----------
    prog : object
        Compiled QICK program.
    soc : object
        QICK SoC instance.
    py_avg : int
        Number of software averages.
    x_axis_vals : np.ndarray
        X-axis values for the sweep.
    y_axis_vals : np.ndarray, optional
        Y-axis values for a 2-D sweep.  When ``None``, a 1-D plot is used.
    x_label : str, optional
        X-axis label.
    y_label : str, optional
        Y-axis label (2-D mode only).
    title_prefix : str, optional
        Figure title prefix.
    show_final_plot : bool, optional
        Render a static final figure after completion.  Default is ``False``.
    iq_process : str, optional
        ``"abs"`` or ``"real"``.  Default is ``"abs"``.

    Returns
    -------
    iqdata : np.ndarray or None
        Running-averaged complex IQ data, or ``None`` if no data was acquired.
    interrupted : bool
        ``True`` if stopped early by ``KeyboardInterrupt``.
    n_done : int
        Number of completed averages.
    """
    # Use a LIFO queue with maxsize=1 to always prefer the latest data frame, implementing "frame dropping"
    data_queue = queue.LifoQueue(maxsize=1)
    stop_event = threading.Event()

    _proc = np.real if iq_process == "real" else np.abs
    _y_label_proc = "ADC Units (Real)" if iq_process == "real" else "ADC Units (Abs)"

    iq = 0
    iqdata = None
    last_i = 0
    interrupted = False

    # Initialize the figure and axes once before the loop
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_display_id = f"live-plot-optimized-{np.random.randint(1e9)}"
    # Initial display of the empty/starting figure
    display(fig, display_id=plot_display_id)

    is_2d = y_axis_vals is not None
    plot_artist = None

    # Pre-create plot artists (Lines or QuadMesh) to be updated later
    if is_2d:
        # For 2D plots, initialize with zeros
        plot_artist = ax.pcolormesh(
            x_axis_vals,
            y_axis_vals,
            np.zeros((len(y_axis_vals), len(x_axis_vals))),
            cmap="viridis",
        )
        fig.colorbar(plot_artist, ax=ax, label="Normalized Amplitude")
        ax.set_ylabel(y_label)
    else:
        # For 1D plots, initialize an empty line
        (plot_artist,) = ax.plot(
            x_axis_vals, np.zeros_like(x_axis_vals), "o-", markersize=5, alpha=0.7
        )
        ax.set_ylabel(_y_label_proc)

    ax.set_xlabel(x_label)
    ax.set_title(f"{title_prefix} (Initializing...)")

    # --- Consumer Thread (Plotter) ---
    def plotter_thread_func():
        while not stop_event.is_set():
            try:
                # Wait briefly for new data; timeout allows checking stop_event regularly
                current_i, data = data_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Efficiently update existing plot artists instead of clearing and redrawing
            if is_2d:
                plot_artist.set_array(data.ravel())
                # Optional: dynamically update color scale for better contrast
                plot_artist.set_clim(vmin=np.min(data), vmax=np.max(data))
            else:
                plot_artist.set_ydata(data)
                # Dynamically rescale Y-axis to fit new data
                data_min = np.min(data)
                data_max = np.max(data)
                data_range = data_max - data_min
                ax.set_ylim(
                    np.min(data) - 0.1 * data_range, np.max(data) + 0.1 * data_range
                )

            ax.set_title(f"{title_prefix} | Average: {current_i + 1} / {py_avg}")
            # Force a refresh of the displayed figure in the notebook
            display(fig, display_id=plot_display_id, update=True)
            data_queue.task_done()

    # Start the persistent plotter thread as a daemon
    plot_thread = threading.Thread(target=plotter_thread_func, daemon=True)
    plot_thread.start()

    # --- Producer Loop (Data Acquisition) ---
    try:
        for i in tqdm(range(py_avg), desc="Software Average Count", mininterval=0.1):
            last_i = i

            # Acquire data from hardware
            iq_list = prog.acquire(soc, rounds=1, progress=False)
            iq_data = iq_list[0][0].dot([1, 1j])
            # Perform cumulative moving average
            iq = iq_data if i == 0 else iq + iq_data
            iqdata = iq / (i + 1)
            plot_data = _proc(iqdata)

            # Prepare data for plotting (normalization for 2D if needed)
            if is_2d:
                row_mins = plot_data.min(axis=1, keepdims=True)
                row_maxs = plot_data.max(axis=1, keepdims=True)
                ranges = row_maxs - row_mins
                ranges[ranges == 0] = 1  # Avoid division by zero
                data_to_push = (plot_data - row_mins) / ranges
            else:
                data_to_push = plot_data

            # Non-blocking push to queue: if full, drop the old frame and push the new one
            try:
                data_queue.put_nowait((i, data_to_push))
            except queue.Full:
                try:
                    data_queue.get_nowait()  # Drop old frame
                    data_queue.put_nowait((i, data_to_push))  # Push new frame
                except:
                    pass  # Ignore race conditions during extreme load

    except KeyboardInterrupt:
        interrupted = True
    finally:
        # Signal plotter thread to stop and wait for it to finish nicely
        stop_event.set()
        if plot_thread.is_alive():
            plot_thread.join(timeout=1.0)

    # Clean up the live plot display
    clear_output(wait=True)
    plt.close(fig)

    # --- Final Static Plot ---
    if show_final_plot:
        final_fig, final_ax = plt.subplots(figsize=(6, 4))
        title_status = "Interrupted" if interrupted else "Completed"
        final_ax.set_title(f"{title_prefix} ({title_status} at avg {last_i + 1})")
        final_ax.set_xlabel(x_label)

        if iqdata is not None:
            plot_data = _proc(iqdata)
            if is_2d:
                # Re-apply normalization for the final static plot
                row_mins = plot_data.min(axis=1, keepdims=True)
                row_maxs = plot_data.max(axis=1, keepdims=True)
                ranges = row_maxs - row_mins
                ranges[ranges == 0] = 1
                final_data = (plot_data - row_mins) / ranges
                im = final_ax.pcolormesh(
                    x_axis_vals, y_axis_vals, final_data, cmap="viridis"
                )
                final_fig.colorbar(im, ax=final_ax, label="Normalized Amplitude")
                final_ax.set_ylabel(y_label)
            else:
                final_ax.plot(x_axis_vals, plot_data, "o-", markersize=5, alpha=0.7)
                final_ax.set_ylabel(_y_label_proc)
        else:
            final_ax.text(
                0.5,
                0.5,
                "No data acquired",
                ha="center",
                va="center",
                transform=final_ax.transAxes,
            )
        display(final_fig)
        plt.close(final_fig)

    return iqdata, interrupted, last_i + 1


# ===================================================================
# 3. Internal Function: Yoko Parameter Sweep (Restored to Original Y-axis Sweep)
# ===================================================================
def _liveplot_sweep_yoko(
    prog,
    soc,
    py_avg,
    x_axis_vals,
    y_axis_vals_yoko,
    yoko_inst_addr,
    yoko_mode="current",
    yoko_voltage_ramp_step=1e-5,
    yoko_current_ramp_step=1e-8,
    yoko_ramp_interval=0.01,
    x_label="X Axis",
    y_label="Y Axis",
    title_prefix="Experiment",
    iq_process="abs",
):
    """
    Execute a 2-D live plot while sweeping a Yokogawa source on the outer axis.

    For each Yoko output level the QICK program is run ``py_avg`` times via
    ``prog.acquire(..., rounds=py_avg)`` and a pcolormesh is updated after
    every row.

    Parameters
    ----------
    prog : object
        Compiled QICK program (inner sweep must be embedded in the program).
    soc : object
        QICK SoC instance.
    py_avg : int
        Hardware averaging rounds per Yoko step.
    x_axis_vals : np.ndarray
        Inner sweep values (plotted on the Y axis of the heatmap).
    y_axis_vals_yoko : np.ndarray
        Yokogawa output values forming the outer loop (plotted on X axis).
    yoko_inst_addr : str
        VISA address of the Yokogawa instrument.
    yoko_mode : str, optional
        ``"current"`` or ``"voltage"``.  Default is ``"current"``.
    x_label : str, optional
        Label for the inner sweep axis.
    y_label : str, optional
        Label for the Yoko axis.
    title_prefix : str, optional
        Figure title prefix.
    iq_process : str, optional
        ``"abs"`` or ``"real"``.  Default is ``"abs"``.

    Returns
    -------
    iqdata_full : np.ndarray
        Complex IQ data of shape ``(len(y_axis_vals_yoko), len(x_axis_vals))``.
    interrupted : bool
        ``True`` if stopped early by ``KeyboardInterrupt``.
    n_done : int
        Number of completed Yoko steps.
    """
    _proc = np.real if iq_process == "real" else np.abs

    rm = pyvisa.ResourceManager()
    yoko = YOKOGS200(yoko_inst_addr, rm)
    yoko.voltage_ramp_step = yoko_voltage_ramp_step
    yoko.current_ramp_step = yoko_current_ramp_step
    yoko.ramp_interval = yoko_ramp_interval

    iqdata_full = np.zeros((len(y_axis_vals_yoko), len(x_axis_vals)), dtype=complex)
    data_to_plot = np.zeros((len(y_axis_vals_yoko), len(x_axis_vals)))
    interrupted = False
    last_idx = 0

    fig, ax = plt.subplots(figsize=(6, 4))

    try:
        yoko_unit = "A" if yoko_mode == "current" else "V"
        value_info = auto_unit(y_axis_vals_yoko, yoko_unit)
        plot_x_vals = value_info["value"]
        dynamic_x_label = f"{y_label} ({value_info['unit']})"
        current_yoko_unit = value_info["unit"]
    except NameError:
        plot_x_vals = y_axis_vals_yoko
        dynamic_x_label = y_label
        current_yoko_unit = yoko_unit
    print(
        f"Yoko sweep mode: {yoko_mode} | unit: {current_yoko_unit} | "
        f"V_step: {yoko.voltage_ramp_step:.2e} V  "
        f"I_step: {yoko.current_ramp_step:.2e} A  "
        f"interval: {yoko.ramp_interval * 1e3:.1f} ms"
    )
    plot_y_vals = x_axis_vals
    dynamic_y_label = x_label

    mesh = ax.pcolormesh(
        plot_x_vals,
        plot_y_vals,
        data_to_plot.T,
        shading="nearest",
        cmap="viridis",
    )

    ax.set_xlabel(dynamic_x_label)
    ax.set_ylabel(dynamic_y_label)

    plot_display_id = f"live-plot-yoko-swapped-{np.random.randint(1e9)}"
    display_handle = display(fig, display_id=plot_display_id)

    try:
        for idx, val in enumerate(
            tqdm(y_axis_vals_yoko, desc=f"Sweeping {yoko_mode} (Plot X-axis)")
        ):
            last_idx = idx
            title = auto_unit(val)
            if yoko_mode == "current":
                yoko.SetMode("current")
                yoko.SetCurrent(val)

                ax.set_title(f"{title_prefix} | {title['value']:.2f}{title['unit']}A")
            else:
                yoko.SetMode("voltage")
                yoko.SetVoltage(val)
                ax.set_title(f"{title_prefix} | {title['value']:.2f}{title['unit']}V")

            iq_list = prog.acquire(soc, rounds=py_avg, progress=False)
            iq_data_row = iq_list[0][0].dot([1, 1j])

            iqdata_full[idx, :] = iq_data_row
            data_to_plot = _proc(iqdata_full)

            mesh.set_array(data_to_plot.T.ravel())

            # Update color scale based on *measured* data only
            # The shape is (len(y_axis_vals_yoko), len(x_axis_vals))
            # Slice up to the currently measured rows
            measured_data = data_to_plot[: idx + 1, :]

            current_max = np.max(measured_data)
            current_min = np.min(measured_data)

            if current_max > current_min:
                mesh.set_clim(vmin=current_min, vmax=current_max)
            elif current_max > 0:
                # Fallback if max == min but non-zero
                mesh.set_clim(vmin=0, vmax=current_max)

            # -----------------------------------------------

            update_display(fig, display_id=plot_display_id)

    except KeyboardInterrupt:
        interrupted = True
        pass

    clear_output(wait=True)

    if interrupted:
        print(f"KeyboardInterrupt: Interrupted at Yoko step: {last_idx + 1}")

    ax.cla()
    if interrupted:
        ax.set_title(f"{title_prefix} (Interrupted at step {last_idx + 1})")
    else:
        ax.set_title(f"{title_prefix} (Completed)")

    ax.set_xlabel(dynamic_x_label)
    ax.set_ylabel(dynamic_y_label)

    # Use final measured min/max for the static plot too
    measured_data_final = (
        data_to_plot[: last_idx + 1, :] if interrupted else data_to_plot
    )
    final_min = np.min(measured_data_final) if measured_data_final.size > 0 else 0
    final_max = np.max(measured_data_final) if measured_data_final.size > 0 else 1
    if final_min == final_max:
        final_min = 0

    im = ax.pcolormesh(
        plot_x_vals,
        plot_y_vals,
        data_to_plot.T,
        shading="nearest",
        vmin=final_min,
        vmax=final_max,
    )
    fig.colorbar(im, ax=ax, label="Amplitude")

    display(fig)
    plt.close(fig)

    return iqdata_full, interrupted, last_idx + 1


# ===================================================================
# 4. Internal Function: 1D Parameter Scan (NEW)
# ===================================================================
def _liveplot_1d_scan(
    soc,
    py_avg,
    scan_x_axis,
    get_prog_callback,
    x_label="Scan Parameter",
    title_prefix="1D Scan",
    show_final_plot=True,
    iq_process="abs",
):
    """
    Execute a 1-D parameter scan with live plotting.

    At each software average iteration, calls *get_prog_callback* for every
    x-value to obtain a freshly configured QICK program, acquires one shot,
    and accumulates a running average across all averages.

    Parameters
    ----------
    soc : object
        QICK SoC instance.
    py_avg : int
        Number of software averaging repetitions.
    scan_x_axis : np.ndarray
        Parameter values for the x axis (e.g. pulse lengths).
    get_prog_callback : callable
        Function with signature ``callback(x_val) → program`` that returns a
        ready-to-run QICK program for the given parameter value.
    x_label : str, optional
        X-axis label.  Default is ``"Scan Parameter"``.
    title_prefix : str, optional
        Figure title prefix.  Default is ``"1D Scan"``.
    show_final_plot : bool, optional
        Render a static final figure after completion.  Default is ``True``.
    iq_process : str, optional
        ``"abs"`` or ``"real"``.  Default is ``"abs"``.

    Returns
    -------
    iqdata : np.ndarray or None
        Running-averaged complex IQ data of shape ``(len(scan_x_axis),)``,
        or ``None`` if no data was acquired.
    interrupted : bool
        ``True`` if stopped early by ``KeyboardInterrupt``.
    n_done : int
        Number of completed averages.
    """
    _proc = np.real if iq_process == "real" else np.abs
    _y_label_proc = "ADC Units (Real)" if iq_process == "real" else "ADC Units (Abs)"

    iq_sum = 0
    iqdata = None
    last_avg = 0
    interrupted = False

    fig, ax = plt.subplots(figsize=(6, 4))
    # Initialize with zeros for the starting line
    (line,) = ax.plot(
        scan_x_axis, np.zeros_like(scan_x_axis), "o-", markersize=5, alpha=0.7
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(_y_label_proc)
    title = ax.set_title(f"{title_prefix} (Initializing...)")

    plot_display_id = f"live-plot-1d-{np.random.randint(1e9)}"
    display(fig, display_id=plot_display_id)

    try:
        for avg in tqdm(range(py_avg), desc="Average Count"):
            last_avg = avg
            iqlst = []
            # Scan through each point in the X axis
            for val in scan_x_axis:
                # Use the callback to get a new program for this specific scan value
                prog = get_prog_callback(val)
                # Acquire 1 round for this point
                iq_list = prog.acquire(soc, rounds=1, progress=False)
                iqlst.append(iq_list[0][0].dot([1, 1j]))

            current_iq_data = np.array(iqlst)
            # Cumulative moving average
            iq_sum = current_iq_data if avg == 0 else iq_sum + current_iq_data
            iqdata = iq_sum / (avg + 1)

            # Update plot data
            plot_data = _proc(iqdata)
            line.set_ydata(plot_data)

            # Dynamically adjust Y-axis range
            current_min, current_max = np.min(plot_data), np.max(plot_data)
            range_span = current_max - current_min
            if range_span == 0:
                range_span = 1  # Avoid zero range if data is flat
            ax.set_ylim(
                current_min - 0.05 * range_span, current_max + 0.05 * range_span
            )

            title.set_text(f"{title_prefix} | Average: {avg + 1} / {py_avg}")
            update_display(fig, display_id=plot_display_id)

    except KeyboardInterrupt:
        interrupted = True
    except Exception as e:
        print(f"An error occurred during 1D scan: {e}")
        interrupted = True

    clear_output(wait=True)
    if interrupted:
        print(f"Scan interrupted at average: {last_avg + 1}")

    if show_final_plot:
        fig_final, ax_final = plt.subplots(figsize=(6, 4))
        title_status = "Interrupted" if interrupted else "Completed"
        ax_final.set_title(f"{title_prefix} ({title_status} at avg {last_avg + 1})")
        ax_final.set_xlabel(x_label)
        ax_final.set_ylabel(_y_label_proc)

        if iqdata is not None:
            ax_final.plot(scan_x_axis, _proc(iqdata), "o-", markersize=5, alpha=0.7)
        else:
            ax_final.text(
                0.5,
                0.5,
                "No data acquired",
                ha="center",
                va="center",
                transform=ax_final.transAxes,
            )

        display(fig_final)
        plt.close(fig_final)

    plt.close(fig)
    return iqdata, interrupted, last_avg + 1


# ===================================================================
# 5. Internal Function: 2D Parameter Scan
# ===================================================================
def _liveplot_2d_scan(
    soc,
    py_avg,
    scan_x_axis,
    scan_y_axis,
    get_prog_callback,
    x_label="X Axis",
    y_label="Y Axis",
    title_prefix="2D Scan",
    show_final_plot=True,
    iq_process="abs",
):
    """
    Execute a 2-D software parameter scan with live pcolormesh updating.

    Iterates over all (y, x) grid points, calling *get_prog_callback* for
    each point to obtain a fresh QICK program, then acquires ``py_avg``
    hardware rounds.  The heatmap color scale adapts to measured data only,
    so unmeasured cells (initialized to zero) do not distort the range.

    Parameters
    ----------
    soc : object
        QICK SoC instance.
    py_avg : int
        Hardware averaging rounds per grid point (passed as ``rounds``).
    scan_x_axis : np.ndarray
        Fast (inner) axis parameter values.
    scan_y_axis : np.ndarray
        Slow (outer) axis parameter values.
    get_prog_callback : callable
        Function with signature ``callback(x_val, y_val) → program`` that
        returns a ready-to-run QICK program.
    x_label : str, optional
        X-axis label.  Default is ``"X Axis"``.
    y_label : str, optional
        Y-axis label.  Default is ``"Y Axis"``.
    title_prefix : str, optional
        Figure title prefix.  Default is ``"2D Scan"``.
    show_final_plot : bool, optional
        Display the final static heatmap after completion.  Default is
        ``True``.
    iq_process : str, optional
        ``"abs"`` or ``"real"``.  Default is ``"abs"``.

    Returns
    -------
    iqdata_full : np.ndarray
        Complex IQ data of shape ``(len(scan_y_axis), len(scan_x_axis))``.
    interrupted : bool
        ``True`` if stopped early by ``KeyboardInterrupt``.
    py_avg : int
        The hardware averaging rounds value (returned for bookkeeping).
    """
    _proc = np.real if iq_process == "real" else np.abs
    _colorbar_label = "ADC Units (Real)" if iq_process == "real" else "ADC Units (Abs)"

    iqdata_full = np.zeros((len(scan_y_axis), len(scan_x_axis)), dtype=complex)
    data_to_plot = np.zeros((len(scan_y_axis), len(scan_x_axis)))
    interrupted = False
    last_y_idx, last_x_idx = 0, 0

    fig, ax = plt.subplots(figsize=(6, 4))

    mesh = ax.pcolormesh(
        scan_x_axis,
        scan_y_axis,
        data_to_plot,
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(mesh, ax=ax, label=_colorbar_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title_prefix} (Initializing...)")

    plot_display_id = f"live-plot-2d-scan-{np.random.randint(1e9)}"
    display(fig, display_id=plot_display_id)

    try:
        # Software 2D sweeps compile often, so it's much faster to accumulate all py_avg shots at the
        # innermost level by using py_avg in the acquire command (as in the 1D parameter search).
        for y_idx, y_val in enumerate(
            tqdm(scan_y_axis, desc=f"Outer Sweep: {y_label}")
        ):
            last_y_idx = y_idx
            for x_idx, x_val in enumerate(
                tqdm(scan_x_axis, desc=f"Inner Sweep: {x_label}", leave=False)
            ):
                last_x_idx = x_idx

                prog = get_prog_callback(x_val, y_val)
                # Note: Py_avg here effectively acts as soft_avgs for the specific (X, Y) point
                iq_list = prog.acquire(soc, rounds=py_avg, progress=False)
                iq_data_pt = iq_list[0][0].dot([1, 1j])

                iqdata_full[y_idx, x_idx] = iq_data_pt
                data_to_plot = _proc(iqdata_full)

                mesh.set_array(data_to_plot.ravel())

                # Update color scale based on *measured* data only
                # Flatten the array and slice up to the currently measured points
                total_measured = y_idx * len(scan_x_axis) + x_idx + 1
                measured_data = data_to_plot.ravel()[:total_measured]

                current_max = np.max(measured_data)
                current_min = np.min(measured_data)

                if current_max > current_min:
                    mesh.set_clim(vmin=current_min, vmax=current_max)
                elif current_max > 0:
                    # Fallback if max == min but non-zero
                    mesh.set_clim(vmin=0, vmax=current_max)

                ax.set_title(
                    f"{title_prefix} | {y_label}={y_val:.2f}, {x_label}={x_val:.2f}"
                )
                update_display(fig, display_id=plot_display_id)

    except KeyboardInterrupt:
        interrupted = True

    clear_output(wait=True)
    if interrupted:
        print(
            f"Scan interrupted at {y_label}: {scan_y_axis[last_y_idx]}, {x_label}: {scan_x_axis[last_x_idx]}"
        )

    ax.cla()
    title_status = "Interrupted" if interrupted else "Completed"
    ax.set_title(f"{title_prefix} ({title_status})")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Use final measured min/max for the static plot too
    total_measured = (
        last_y_idx * len(scan_x_axis) + last_x_idx + 1
        if interrupted
        else data_to_plot.size
    )
    final_measured = data_to_plot.ravel()[:total_measured]
    final_min = np.min(final_measured) if final_measured.size > 0 else 0
    final_max = np.max(final_measured) if final_measured.size > 0 else 1
    if final_min == final_max:
        final_min = 0

    im = ax.pcolormesh(
        scan_x_axis,
        scan_y_axis,
        data_to_plot,
        shading="auto",
        cmap="viridis",
        vmin=final_min,
        vmax=final_max,
    )
    fig.colorbar(im, ax=ax, label="ADC Units (Abs)")

    if show_final_plot:
        display(fig)

    plt.close(fig)

    return iqdata_full, interrupted, py_avg
