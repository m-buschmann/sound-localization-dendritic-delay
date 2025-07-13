from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from scipy.interpolate import make_interp_spline
from numpy import interp
import json
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import sys

prefs.codegen.target = "numpy"
defaultclock.dt = 0.01 * ms


def calculate_threshold_worker(args):
    """Worker function for calculating thresholds in parallel.

    Args:
        args: Tuple containing (left_index, n_comp, lambda_um, min_angle, max_angle, step)

    Returns:
        tuple: (left_index, threshold_value)
    """
    left_index, n_comp, lambda_um, min_angle, max_angle, step = args
    right_index = 2 * n_comp + 1 - left_index

    try:
        # Disable verbose output for worker processes
        _, _, max_voltages, _ = simulate_response_per_angle(
            n_comp=n_comp,
            lambda_um=lambda_um,
            left_index=left_index,
            right_index=right_index,
            min_angle=min_angle,
            max_angle=max_angle,
            step=step,
            plot=False,
            verbose=False,  # Add this to reduce output
        )
        threshold = calculate_threshold(max_voltages)
        return left_index, threshold
    except Exception as e:
        return left_index, None


def simulate_response_worker(args):
    """Worker function for simulating responses in parallel.

    Args:
        args: Tuple containing (left_index, n_comp, lambda_um, min_angle, max_angle, step, thresh_filepath)

    Returns:
        tuple: (left_index, angles, all_voltages, max_voltages, spike_counts)
    """
    left_index, n_comp, lambda_um, min_angle, max_angle, step, thresh_filepath = args
    right_index = 2 * n_comp + 1 - left_index

    try:
        # Load threshold for this specific neuron
        threshold = load_thresholds(thresh_filepath, l=left_index)

        angles, all_voltages, max_voltages, spike_counts = simulate_response_per_angle(
            left_index=left_index,
            right_index=right_index,
            n_comp=n_comp,
            lambda_um=lambda_um,
            min_angle=min_angle,
            max_angle=max_angle,
            step=step,
            threshold=threshold,
            verbose=False,  # Add this to reduce output
        )

        return left_index, angles, all_voltages, max_voltages, spike_counts
    except Exception as e:
        return left_index, None, None, None, None


def excite_both_dendrites(
    N=6,
    f_stim_Hz=500,
    f_pre_Hz=350,
    tmax_ms=10,
    jitter_ms=0,
    sound_angle=0,
    n_comp=11,
    lambda_um=200,
    left_comp_index=None,
    right_comp_index=None,
    plot=False,
    threshold=-55.0,
    verbose=True,
):
    start_scope()

    # Key morphology
    lambda_ = lambda_um * um
    compartment_length = 0.05 * lambda_
    dend_length = n_comp * compartment_length
    diameter = 2 * um  # From paper

    morpho = Soma(diameter=20 * um)
    morpho.L = Cylinder(length=dend_length, diameter=diameter, n=n_comp)
    morpho.R = Cylinder(length=dend_length, diameter=diameter, n=n_comp)

    # show()

    eqs = """
    Im = gl * (El - v) + gsyn*(Esyn-v) : amp/meter**2
    dgsyn/dt = -gsyn / tau_syn : siemens/meter**2
    gl : siemens/meter**2
    El : volt
    Esyn : volt
    tau_syn : second (shared)
    """

    neuron = SpatialNeuron(
        morphology=morpho,
        model=eqs,
        threshold=f"v > {threshold}*mV",
        threshold_location=0,
        reset="v = -65*mV",
        refractory="2*ms",
        Cm=1 * uF / cm**2,
        Ri=200 * ohm * cm,
        method="exponential_euler",
    )

    neuron.v = -65 * mV
    # neuron.gl = 0.0005*siemens/cm**2
    neuron.gl = (
        0.008 * siemens / cm**2
    )  # TODO play with this, this influences threshold crossing!
    neuron.El = -62.5 * mV
    neuron.gsyn = 0 * siemens / cm**2
    neuron.Esyn = 0 * mV
    neuron.tau_syn = 0.5 * ms
    w_syn = 14 * nS  # (14-26 in paper)

    # Calculate arrival times for both dendrites
    time_left, time_right = calculate_arrival_times(sound_angle)
    itd = time_right - time_left

    left_i, left_t = binomial_spike_train(
        N, f_stim_Hz, f_pre_Hz, tmax_ms, phase=0, jitter_ms=jitter_ms
    )
    right_i, right_t = binomial_spike_train(
        N, f_stim_Hz, f_pre_Hz, tmax_ms, phase=itd, jitter_ms=jitter_ms
    )

    # Shift times if negative
    all_times = np.concatenate([left_t, right_t])
    min_time = np.min(all_times)
    if min_time < 0:
        left_t, right_t = left_t - min_time, right_t - min_time
        tmax_ms = tmax_ms - min_time

    input_left = SpikeGeneratorGroup(N, left_i, left_t * ms)
    input_right = SpikeGeneratorGroup(N, right_i, right_t * ms)

    # Compartment centers (for info/indices)
    compartment_centers = np.linspace(
        compartment_length / 2, dend_length - compartment_length / 2, n_comp
    )
    syn_dist = 0.1 * lambda_  # 0.1 lambda as in paper

    # Defaults: closest to 0.1 lambda in each dendrite
    default_left = 1 + np.argmin(np.abs(compartment_centers - syn_dist))
    default_right = n_comp + 1 + np.argmin(np.abs(compartment_centers - syn_dist))
    left_index = default_left if left_comp_index is None else left_comp_index
    right_index = default_right if right_comp_index is None else right_comp_index

    # Synapse connections
    syn_left = Synapses(input_left, neuron, on_pre="gsyn_post += w_syn / area_post")
    syn_left.connect(i=range(N), j=left_index)
    syn_right = Synapses(input_right, neuron, on_pre="gsyn_post += w_syn / area_post")
    syn_right.connect(i=range(N), j=right_index)

    M = StateMonitor(neuron, "v", record=True)
    spikemon = SpikeMonitor(neuron)
    run(tmax_ms * ms)

    if verbose:
        print(
            f"[INFO] n_comp={n_comp}, lambda={lambda_um} um, dendrite length={dend_length/um:.1f} um, left_index={left_index}, right_index={right_index}"
        )
    all_v = M.v[0] / mV
    max_v = np.max(M.v[0] / mV)
    if verbose:
        print("Max soma voltage:", max_v, "mV")
        if spikemon.count[0] > 0:
            print("Spike times (ms):", spikemon.t / ms)
        else:
            print("Soma did NOT spike.")

    if plot:
        left_label = f"{left_index}L"
        right_label = f"{right_index - n_comp}R"
        print(f"Left index: {left_label}, Right index: {right_label}")

        plt.plot(M.t / ms, M.v[left_index] / mV, label=f"left dend {left_index}")
        plt.plot(M.t / ms, M.v[0] / mV, label="soma")
        plt.plot(M.t / ms, M.v[right_index] / mV, label=f"right dend {right_index}")
        plt.xlabel("Time (ms)")
        plt.ylabel("v (mV)")
        plt.legend()
        plt.title("Stimulus on dendrites")
        plt.show()

        # Space–time map
        voltmap = np.vstack(
            [
                M.v[n_comp + 1 :][::-1] / mV,  # right dendrite, distal->proximal
                M.v[0][np.newaxis] / mV,  # soma
                M.v[1 : n_comp + 1] / mV,  # left dendrite, prox->distal
            ]
        )
        times = M.t / ms

        plt.figure(figsize=(10, 4))
        plt.imshow(
            voltmap,
            aspect="auto",
            cmap="viridis",
            extent=[times[0], times[-1], dend_length / um, -dend_length / um],
        )
        plt.colorbar(label="Voltage (mV)")
        # Get the limits of the y-axis
        y_min, y_max = plt.ylim()

        # Calculate positions at 1/4, 1/2, and 3/4 of the y-axis
        ytick_positions = [
            y_min + (y_max - y_min) * 0.25,
            y_min + (y_max - y_min) * 0.5,
            y_min + (y_max - y_min) * 0.75,
        ]

        # Set the y-ticks and labels
        plt.yticks(ytick_positions, ["left dendrite", "Soma", "right dendrite"])
        plt.xlabel("Time (ms)")
        plt.ylabel("Position (μm)")
        plt.title("Space–time voltage map")
        plt.show()

    return max_v, spikemon.count[0], all_v


def excite_one_dendrite(
    N=6,
    f_stim_Hz=500,
    f_pre_Hz=350,
    tmax_ms=10,
    jitter_ms=0,
    sound_angle=0,
    n_comp=11,
    lambda_um=200,
    left_comp_index=None,
    right_comp_index=None,
    plot=False,
):
    start_scope()

    # Key morphology
    lambda_ = lambda_um * um
    compartment_length = 0.05 * lambda_
    dend_length = n_comp * compartment_length
    diameter = 2 * um  # From paper

    morpho = Soma(diameter=20 * um)
    morpho.L = Cylinder(length=dend_length, diameter=diameter, n=n_comp)
    morpho.R = Cylinder(length=dend_length, diameter=diameter, n=n_comp)

    # show()

    eqs = """
    Im = gl * (El - v) + gsyn*(Esyn-v) : amp/meter**2
    dgsyn/dt = -gsyn / tau_syn : siemens/meter**2
    gl : siemens/meter**2
    El : volt
    Esyn : volt
    tau_syn : second (shared)
    """

    neuron = SpatialNeuron(
        morphology=morpho,
        model=eqs,
        threshold="v > -55*mV",
        threshold_location=0,
        reset="v = -65*mV",
        refractory="2*ms",
        Cm=1 * uF / cm**2,
        Ri=200 * ohm * cm,
        method="exponential_euler",
    )

    neuron.v = -65 * mV
    # neuron.gl = 0.0005*siemens/cm**2
    neuron.gl = (
        0.008 * siemens / cm**2
    )  # TODO play with this, this influences threshold crossing!
    neuron.El = -62.5 * mV
    neuron.gsyn = 0 * siemens / cm**2
    neuron.Esyn = 0 * mV
    neuron.tau_syn = 0.5 * ms
    w_syn = 14 * nS  # (14-26 in paper)

    # Calculate arrival times for both dendrites
    time_left, time_right = calculate_arrival_times(sound_angle)
    itd = time_right - time_left

    left_i, left_t = binomial_spike_train(
        N, f_stim_Hz, f_pre_Hz, tmax_ms, phase=0, jitter_ms=jitter_ms
    )
    right_i, right_t = binomial_spike_train(
        N, f_stim_Hz, f_pre_Hz, tmax_ms, phase=itd, jitter_ms=jitter_ms
    )

    # Shift times if negative
    all_times = np.concatenate([left_t, right_t])
    min_time = np.min(all_times)
    if min_time < 0:
        left_t, right_t = left_t - min_time, right_t - min_time
        tmax_ms = tmax_ms - min_time

    input_left = SpikeGeneratorGroup(N, left_i, left_t * ms)
    input_right = SpikeGeneratorGroup(N, right_i, right_t * ms)

    # Compartment centers (for info/indices)
    compartment_centers = np.linspace(
        compartment_length / 2, dend_length - compartment_length / 2, n_comp
    )
    syn_dist = 0.1 * lambda_  # 0.1 lambda as in paper

    # Defaults: closest to 0.1 lambda in each dendrite
    default_left = 1 + np.argmin(np.abs(compartment_centers - syn_dist))
    default_right = n_comp + 1 + np.argmin(np.abs(compartment_centers - syn_dist))
    left_index = default_left if left_comp_index is None else left_comp_index
    right_index = default_right if right_comp_index is None else right_comp_index

    # Synapse connections
    syn_left = Synapses(input_left, neuron, on_pre="gsyn_post += w_syn / area_post")
    syn_left.connect(i=range(N), j=left_index)

    M = StateMonitor(neuron, "v", record=True)
    spikemon = SpikeMonitor(neuron)
    run(tmax_ms * ms)

    print(
        f"[INFO] n_comp={n_comp}, lambda={lambda_um} um, dendrite length={dend_length/um:.1f} um, left_index={left_index}, right_index={right_index}"
    )
    max_v = np.max(M.v[0] / mV)
    print("Max soma voltage:", max_v, "mV")
    if spikemon.count[0] > 0:
        print("Soma spiked!")
        print("Spike times (ms):", spikemon.t / ms)
    else:
        print("Soma did NOT spike.")

    if plot:
        left_label = f"{left_index}L"
        right_label = f"{right_index - n_comp}R"
        print(f"Left index: {left_label}, Right index: {right_label}")

        plt.plot(M.t / ms, M.v[left_index] / mV, label=f"left dend {left_index}")
        plt.plot(M.t / ms, M.v[0] / mV, label="soma")
        plt.plot(M.t / ms, M.v[right_index] / mV, label=f"right dend {right_index}")
        plt.xlabel("Time (ms)")
        plt.ylabel("v (mV)")
        plt.legend()
        plt.title("Stimulus on dendrites")
        plt.show()

        # Space–time map
        voltmap = np.vstack(
            [
                M.v[n_comp + 1 :][::-1] / mV,  # right dendrite, distal->proximal
                M.v[0][np.newaxis] / mV,  # soma
                M.v[1 : n_comp + 1] / mV,  # left dendrite, prox->distal
            ]
        )
        times = M.t / ms

        plt.figure(figsize=(10, 4))
        plt.imshow(
            voltmap,
            aspect="auto",
            cmap="viridis",
            extent=[times[0], times[-1], dend_length / um, -dend_length / um],
        )
        plt.colorbar(label="Voltage (mV)")
        # Get the limits of the y-axis
        y_min, y_max = plt.ylim()

        # Calculate positions at 1/4, 1/2, and 3/4 of the y-axis
        ytick_positions = [
            y_min + (y_max - y_min) * 0.25,
            y_min + (y_max - y_min) * 0.5,
            y_min + (y_max - y_min) * 0.75,
        ]

        # Set the y-ticks and labels
        plt.yticks(ytick_positions, ["left dendrite", "Soma", "right dendrite"])
        plt.xlabel("Time (ms)")
        plt.ylabel("Position (μm)")
        plt.title("Space–time voltage map")
        plt.show()

    return max_v


def simulate_different_frequencies(
    min_frequency=50, max_frequency=1001, step=10, threshold=-55.0
):

    max_voltages = []
    # iterate over different sound frequencies
    for sound_frequency in range(min_frequency, max_frequency, step):
        print(f"Sound frequency: {sound_frequency} Hz")
        max_v, spike_count = excite_both_dendrites(
            N=6,
            f_stim_Hz=sound_frequency,
            f_pre_Hz=350,
            tmax_ms=20,
            jitter_ms=0,
            sound_angle=0,
            threshold=threshold,
        )
        max_voltages.append(max_v)

    # plot max voltages for different sound frequencies
    plt.figure(figsize=(10, 5))
    plt.plot(range(min_frequency, max_frequency, 10), max_voltages, marker="o")
    plt.xlabel("Sound Frequency (Hz)")
    plt.ylabel("Max Soma Voltage (mV)")
    plt.title("Max Soma Voltage vs Sound Frequency")
    plt.grid()
    plt.tight_layout()
    plt.show()

    return max_voltages


def simulate_response_per_angle(
    left_index,
    right_index,
    n_comp=10,
    lambda_um=200,
    min_angle=90,
    max_angle=270,
    step=1,
    threshold=-55.0,
    plot=False,
    verbose=True,
):
    angles = []
    all_voltages = []
    max_voltages = []
    left_label = f"{left_index}L"
    right_label = f"{right_index - n_comp}R"
    spike_counts = []

    if verbose:
        print(
            f"Left dendrite compartment: {left_index}, Right dendrite compartment: {right_index}"
        )

    # Create progress bar for angle iteration
    angle_range = range(min_angle, max_angle, step)
    if verbose:
        angle_iterator = tqdm(
            angle_range,
            desc=f"Processing angles for neuron L{left_index}_R{right_index - n_comp}",
            leave=False,
            disable=False,
        )
    else:
        angle_iterator = angle_range

    # Iterate over sound angles
    for angle in angle_iterator:
        if verbose and not isinstance(angle_iterator, tqdm):
            print(f"Sound angle: {angle}°")

        max_v, spike_count, all_v = excite_both_dendrites(
            N=6,
            f_stim_Hz=500,
            f_pre_Hz=350,
            tmax_ms=20,
            jitter_ms=0,
            sound_angle=angle,
            n_comp=n_comp,
            lambda_um=lambda_um,
            left_comp_index=left_index,
            right_comp_index=right_index,
            threshold=threshold,
            verbose=verbose,
        )
        all_voltages.append(all_v)
        max_voltages.append(max_v)
        angles.append(angle)
        spike_counts.append(spike_count)

        if plot:
            # Smooth using a moving average
            smoothed_voltages = smooth_data(angles, max_voltages, window_size=20)
            plt.figure(figsize=(10, 5))
            plt.plot(range(min_angle, max_angle, step), max_voltages, marker="o")
            plt.plot(
                angles, smoothed_voltages, color="red", linewidth=2, label="Smooth fit"
            )
            plt.xlabel("Sound Angle (degrees)")
            plt.ylabel("Max Soma Voltage (mV)")
            plt.title("Max Soma Voltage vs Sound Angle")
            plt.grid()
            plt.tight_layout()
            plt.show()

    return angles, all_voltages, max_voltages, spike_counts


def main():
    # parameters
    n_comp = 11
    lambda_um = 200  # from paper
    left_start_index = 1
    left_end_index = 11
    min_angle = 0
    angle = 0
    max_angle = 360
    step = 1
    threshold_percentile = 0.9  # gets applied to max voltages
    thresh_filepath = Path(__file__).parent / "thresholds.json"
    response_filepath = Path(__file__).parent / "response_data.json"

    # assertions
    assert (
        left_start_index <= left_end_index
    ), "left_start_index index must be less than or equal to end index."
    assert (
        left_end_index <= n_comp
    ), "left_end_index index must be less than or equal to n_comp."

    # flags
    # computation:
    calc_thresholds = False  # if False, loads from file.
    simulate_response = False  # if False, loads from file.
    # single neuron plots:
    do_single_combo = False
    polar_plot_spikes = False
    polar_plot_max_voltages = False
    # multi-neuron plots:
    polar_plot_v_grid = False
    polar_plot_spk_grid = False
    polar_plot_v_multi = False
    polar_plot_spk_multi = False
    multiple_curves = False
    plot_thresholds = True  # plot thresholds for all neurons
    # multiprocessing flag:
    use_mp = True

    # data collection for multi-neuron plots:
    all_voltage_data = {}
    all_spike_data = {}

    # Multiprocessing setup
    left_indices = list(range(left_start_index, left_end_index + 1))
    num_processes = min(mp.cpu_count(), len(left_indices))

    print(f"Simulation Configuration:")
    print(
        f"   - Neurons: {len(left_indices)} (left_index {left_start_index} to {left_end_index})"
    )
    print(
        f"   - Angles: {min_angle}° to {max_angle}° (step={step}°, total={len(range(min_angle, max_angle, step))} angles)"
    )
    print(f"   - Compartments: {n_comp}, Lambda: {lambda_um}μm")

    if use_mp:
        print(f"   - Parallel processing: {num_processes} processes")
    else:
        print(f"   - Single-threaded processing")
    print()

    # PHASE 1: Calculate thresholds (must complete before response simulation)
    if calc_thresholds:
        print("Calculating thresholds...")
        thresholds = load_thresholds(thresh_filepath)  # might print that no file exists
        if not thresholds:
            thresholds = {}

        # Prepare arguments for threshold calculation
        threshold_args = [
            (left_index, n_comp, lambda_um, min_angle, max_angle, step)
            for left_index in left_indices
        ]

        if use_mp:
            print(
                "   Running threshold calculations in parallel (may take a while until progress is visible)"
            )
            with mp.Pool(processes=num_processes) as pool:
                # Use tqdm to show progress for threshold calculation
                threshold_results = []
                with tqdm(
                    total=len(threshold_args),
                    desc="Calculating thresholds",
                    unit="neuron",
                ) as pbar:
                    for result in pool.imap(calculate_threshold_worker, threshold_args):
                        threshold_results.append(result)
                        pbar.update(1)
        else:
            print("   Running threshold calculations sequentially...")
            threshold_results = []
            for args in tqdm(
                threshold_args, desc="Calculating thresholds", unit="neuron"
            ):
                threshold_results.append(calculate_threshold_worker(args))

        # Update thresholds dictionary and save to file
        for left_index, threshold in threshold_results:
            if threshold is not None:
                thresholds[str(left_index)] = threshold

        # Save all thresholds to file before proceeding
        save_thresholds(thresholds, thresh_filepath)
        print(f"   Thresholds calculated and saved")

    # PHASE 2: Simulate responses (can be parallelized now that thresholds are ready)
    if simulate_response:
        print("\nSimulating responses...")

        # Prepare arguments for response simulation
        response_args = [
            (left_index, n_comp, lambda_um, min_angle, max_angle, step, thresh_filepath)
            for left_index in left_indices
        ]

        if use_mp:
            print(
                "   Running response simulations in parallel (may take a while until progress is visible)"
            )
            with mp.Pool(processes=num_processes) as pool:
                # Use tqdm to show progress for response simulation
                response_results = []
                with tqdm(
                    total=len(response_args), desc="Simulating responses", unit="neuron"
                ) as pbar:
                    for result in pool.imap(simulate_response_worker, response_args):
                        response_results.append(result)
                        pbar.update(1)
        else:
            print("   Running response simulations sequentially...")
            response_results = []
            for args in tqdm(response_args, desc="Simulating responses", unit="neuron"):
                response_results.append(simulate_response_worker(args))

        # Store all results sequentially to avoid file conflicts
        print("   Saving results to file...")
        for left_index, angles, all_voltages, max_voltages, spike_counts in tqdm(
            response_results, desc="Saving data", unit="neuron"
        ):
            if angles is not None:
                store_response_per_angle(
                    left_index,
                    angles,
                    all_voltages,
                    max_voltages,
                    spike_counts,
                    filepath=response_filepath,
                )

        # Collect results for plotting
        for (
            left_index,
            angles,
            all_voltages,
            max_voltages,
            spike_counts,
        ) in response_results:
            if angles is not None:
                right_index = 2 * n_comp + 1 - left_index
                neuron_label = f"L{left_index}_R{right_index - n_comp}"
                all_voltage_data[neuron_label] = (angles, max_voltages)
                all_spike_data[neuron_label] = (angles, spike_counts)
        print(f"     Response simulations completed and saved")
    else:
        # Load existing response data
        print("\n  Loading existing response data...")
        for left_index in tqdm(left_indices, desc="Loading data", unit="neuron"):
            try:
                angles, all_voltages, max_voltages, spike_counts = (
                    load_response_per_angle(
                        left_index=left_index,
                        response_filepath=response_filepath,
                        min_angle=min_angle,
                        max_angle=max_angle,
                        step=step,
                    )
                )
                if angles is not None:
                    right_index = 2 * n_comp + 1 - left_index
                    neuron_label = f"L{left_index}_R{right_index - n_comp}"
                    all_voltage_data[neuron_label] = (angles, max_voltages)
                    all_spike_data[neuron_label] = (angles, spike_counts)
            except Exception as e:
                print(f"Error loading data for left_index {left_index}: {e}")
        print(f"     Data loaded for {len(all_voltage_data)} neurons")

    # PHASE 3: Single neuron plots and analysis (sequential, as these are typically quick)
    if any([do_single_combo, polar_plot_spikes, polar_plot_max_voltages]):
        print(f"\n  Individual plots and analysis...")
        for left_index in tqdm(left_indices, desc="Individual analysis", unit="neuron"):
            right_index = 2 * n_comp + 1 - left_index

            # Load data for this neuron if available
            neuron_label = f"L{left_index}_R{right_index - n_comp}"
            if neuron_label in all_voltage_data:
                angles, max_voltages = all_voltage_data[neuron_label]
                _, spike_counts = all_spike_data[neuron_label]
            else:
                continue

            if do_single_combo:
                excite_both_dendrites(
                    N=6,
                    f_stim_Hz=500,
                    f_pre_Hz=350,
                    tmax_ms=10,
                    jitter_ms=0,
                    sound_angle=angle,
                    n_comp=n_comp,
                    lambda_um=lambda_um,
                    plot=True,
                    left_comp_index=left_index,
                    right_comp_index=right_index,
                    threshold=load_thresholds(thresh_filepath, l=left_index),
                )

            if polar_plot_spikes:
                polar_bar_plot(
                    angles,
                    spike_counts,
                    title=f"Spike Count vs Sound Angle (Left: {left_index}, Right: {right_index})",
                    xlabel="Sound Angle (degrees)",
                    ylabel="Spike Count",
                )

            if polar_plot_max_voltages:
                polar_bar_plot(
                    angles,
                    max_voltages,
                    title=f"Max Soma Voltage vs Sound Angle (Left: {left_index}, Right: {right_index})",
                    xlabel="Sound Angle (degrees)",
                    ylabel="Max Soma Voltage (mV)",
                )

    # PHASE 4: Multi-neuron plots
    if any(
        [
            polar_plot_v_multi,
            polar_plot_spk_multi,
            polar_plot_v_grid,
            polar_plot_spk_grid,
            multiple_curves,
            plot_thresholds,
        ]
    ):
        print(f"\n  Generating Multi-neuron plots")
        if polar_plot_v_multi and all_voltage_data:
            print("   Creating multi-series voltage plot...")
            polar_bar_plot_multi(all_voltage_data, title="All Neurons - Max Voltage")
        if polar_plot_spk_multi and all_spike_data:
            print("   Creating multi-series spike plot...")
            polar_bar_plot_multi(all_spike_data, title="All Neurons - Spike Count")
        if polar_plot_v_grid and all_voltage_data:
            print("   Creating voltage grid plots...")
            polar_bar_plot_grid(all_voltage_data, title="All Neurons - Max Voltage")
        if polar_plot_spk_grid and all_spike_data:
            print("   Creating spike grid plots...")
            polar_bar_plot_grid(all_spike_data, title="All Neurons - Spike Count")
        if multiple_curves and all_voltage_data:
            print("   Creating multiple curves plot...")
            plot_multiple_curves(
                n_comp=n_comp,
                lambda_um=lambda_um,
                angles=angles,
                all_max_voltages=all_voltage_data,
                left_start_index=left_start_index,
                left_end_index=left_end_index,
            )
        if plot_thresholds:
            print("   Plotting thresholds for all neurons...")
            plot_neuron_thresholds(
                thresholds_path=thresh_filepath,
                n_comp=n_comp,
                left_start_index=left_start_index,
                left_end_index=left_end_index,
                title="Threshold for each neuron",
                xlabel="Neuron Index (left)",
                ylabel="Threshold (mV)",
            )

        print("     Multi-neuron plots completed")

    print(f"\n  Summary:")
    print(f"   - {len(all_voltage_data)} neurons processed")
    print(f"   - Data saved to: {response_filepath}")
    print(f"   - Thresholds saved to: {thresh_filepath}")


if __name__ == "__main__":
    # Cross-platform multiprocessing setup
    if not setup_multiprocessing():
        print("Failed to setup multiprocessing. Running in single-threaded mode.")
        # You could set use_mp = False here or handle the error appropriately

    main()
