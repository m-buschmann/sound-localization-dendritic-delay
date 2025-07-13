import math
import numpy as np
from brian2 import *
import matplotlib.pyplot as plt
from pathlib import Path
import json
import platform


def setup_multiprocessing():
    """Setup multiprocessing with appropriate settings for different platforms."""
    import multiprocessing as mp

    # Determine the best start method for the current platform
    if platform.system() == "Windows":
        # Windows requires 'spawn' method
        if "spawn" not in mp.get_all_start_methods():
            print("Warning: 'spawn' method not available on this Windows system")
            return False
        mp.set_start_method("spawn", force=True)
        return True
    elif platform.system() == "Darwin":  # macOS
        # macOS supports both 'fork' and 'spawn', but 'spawn' is safer for Brian2
        if "spawn" in mp.get_all_start_methods():
            mp.set_start_method("spawn", force=True)
        elif "fork" in mp.get_all_start_methods():
            mp.set_start_method("fork", force=True)
            print(
                "Warning: Using 'fork' method on macOS. Consider upgrading to Python 3.8+ for 'spawn' support."
            )
        else:
            print("Error: No suitable multiprocessing start method available")
            return False
        return True
    else:  # Linux and other Unix-like systems
        # Linux typically defaults to 'fork', but 'spawn' is more compatible with complex libraries
        if "spawn" in mp.get_all_start_methods():
            mp.set_start_method("spawn", force=True)
        elif "fork" in mp.get_all_start_methods():
            mp.set_start_method("fork", force=True)
        else:
            print("Error: No suitable multiprocessing start method available")
            return False
        return True


# -- Utlity functions --
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_arrival_times(
    angle_deg, speed_of_sound=343, inter_ear_distance=0.08, source_distance=1.0
):
    """
    Calculates the arrival times of sound to two ear points given the source position in ms.
    Front = 0, right = 90, back = 180, left = 270 or -90 degrees.

    Parameters:
        angle_rad (float): Angle to the sound source in radians (0 = in front, positive = to the right).
        speed_of_sound (float): Speed of sound in the medium (e.g., 343 m/s in air).
        inter_ear_distance (float): Distance between the two ears (meters).
        source_distance (float): Distance from the center point between the ears to the sound source (meters).

    Returns:
        tuple: (arrival_time_left, arrival_time_right)
    """

    angle_rad = math.radians(angle_deg)

    # Ear positions (assuming head center at (0,0), ears on y-axis)
    left_ear = (0, -inter_ear_distance / 2)
    right_ear = (0, inter_ear_distance / 2)

    # Source position in Cartesian coordinates
    source_x = source_distance * math.cos(angle_rad)
    source_y = source_distance * math.sin(angle_rad)
    source_pos = (source_x, source_y)

    # Distance from source to each ear
    dist_left = euclidean(source_pos, left_ear)
    dist_right = euclidean(source_pos, right_ear)

    # Arrival times in ms
    time_left = dist_left / speed_of_sound * 1000
    time_right = dist_right / speed_of_sound * 1000

    return (time_left, time_right)


def binomial_spike_train(N, f_stim_Hz, f_pre_Hz, tmax_ms, phase=0, jitter_ms=0):
    cycle_length = 1000 / f_stim_Hz  # ms per period
    n_cycles = int(tmax_ms / cycle_length)
    p = f_pre_Hz / f_stim_Hz
    indices = []
    times = []
    for cycle in range(n_cycles):
        base_time = cycle * cycle_length + phase
        for axon in range(N):
            if np.random.rand() < p:
                time = base_time + np.random.normal(0, jitter_ms)
                indices.append(axon)
                times.append(time)
    return indices, times


def smooth_data(angles, max_voltages, window_size=5):
    """Smooth data using a moving average over nearby points."""
    smoothed_voltages = np.zeros_like(max_voltages)
    for i in range(len(max_voltages)):
        # Define the window range
        start = max(0, i - window_size // 2)
        end = min(len(max_voltages), i + window_size // 2 + 1)
        # Average over the window
        smoothed_voltages[i] = np.mean(max_voltages[start:end])
    return smoothed_voltages


def calculate_threshold(max_voltages, percentile=0.75):
    """Get the max voltages for N angles at one neuron. Return the ideal threshold for spikes based on a percentile of N."""
    sorted_voltages = np.sort(max_voltages)
    cutoff = int(len(sorted_voltages) * percentile)
    threshold = sorted_voltages[cutoff]
    return threshold


# -- Printing and plotting functions --
def polar_bar_plot(
    angles,
    values,
    title="Polar Bar Plot",
    xlabel="Angle (degrees)",
    ylabel="Value",
    threshold=None,
):
    """Create a polar bar plot.

    Args:
        angles (list): List of angles in degrees
        values (list): List of values corresponding to the angles
        title (str): Title of the plot
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        threshold (float): Threshold for color coding bars above threshold (optional)
    """
    # when a threshold is crossed, plot this value in a different color
    offset = -1 * np.min(values)  # add to avoid negative bars
    values = [v + offset for v in values]  # Adjust values to avoid negative bars
    angles_rad = np.radians(angles)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    if threshold is not None:
        bars = ax.bar(
            angles_rad,
            values,
            width=np.deg2rad(1),
            alpha=0.5,
            bottom=0.0,
            color=["red" if v - offset > threshold else "blue" for v in values],
        )
    else:
        bars = ax.bar(
            angles_rad,
            values,
            width=np.deg2rad(1),
            alpha=0.5,
            bottom=0.0,
            color="blue",
        )

    ax.set_theta_zero_location("N")  # Optional: 0° at the top
    ax.set_theta_direction(-1)  # Optional: clockwise

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def polar_bar_plot_multi(
    data_dict,
    title="Polar Bar Plot",
    xlabel="Angle (degrees)",
    ylabel="Value",
    threshold=-20.0,
    colors=None,
):
    """Create a polar bar plot with multiple data series.

    Args:
        data_dict (dict): Dictionary with {label: (angles, values)} pairs
        title (str): Title of the plot
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        threshold (float): Threshold for color coding
        colors (list): List of colors for each series (optional)
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    colormap = "RdBu"  # Use opposite colormap to the one used in multiple curves

    if colors is None:
        # pull linearly spaced colors from a diverging colormap.
        # some options: RdBu, PiYG, coolwarm, etc
        cmap = plt.get_cmap(colormap)
        n_series = len(data_dict)
        colors = [cmap(i / n_series) for i in range(n_series)]

    # Calculate global offset to avoid negative bars
    all_values = [v for angles, values in data_dict.values() for v in values]
    offset = -1 * np.min(all_values) if np.min(all_values) < 0 else 0

    # Calculate bar width based on number of series and angular resolution
    n_series = len(data_dict)
    base_width = np.deg2rad(
        360 / len(list(data_dict.values())[0][0])
    )  # Assuming same angles for all
    bar_width = base_width / n_series

    for i, (label, (angles, values)) in enumerate(data_dict.items()):
        angles_rad = np.radians(angles)
        adjusted_values = [v + offset for v in values]

        # Offset bars slightly for each series for better visibility
        offset_angles = angles_rad + (i - n_series / 2) * bar_width

        bars = ax.bar(
            offset_angles,
            adjusted_values,
            width=bar_width * 3.5,  # Increase width for better visibility
            alpha=0.7,
            label=label,
            color=colors[i],
            bottom=0.0,
        )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.show()


def polar_bar_plot_grid(
    data_dict, title="Polar Bar Plots Grid", threshold=-20.0, cols=3
):
    """Create a grid of polar bar plots.

    Args:
        data_dict (dict): Dictionary with {label: (angles, values)} pairs
        title (str): Overall title
        threshold (float): Threshold for color coding
        cols (int): Number of columns in the grid
    """
    n_plots = len(data_dict)
    rows = (n_plots + cols - 1) // cols  # Ceiling division

    fig = plt.figure(figsize=(5 * cols, 4 * rows))
    fig.suptitle(title, fontsize=16)

    # Calculate global offset
    all_values = [v for angles, values in data_dict.values() for v in values]
    offset = -1 * np.min(all_values) if np.min(all_values) < 0 else 0

    for i, (label, (angles, values)) in enumerate(data_dict.items()):
        ax = fig.add_subplot(rows, cols, i + 1, projection="polar")

        angles_rad = np.radians(angles)
        adjusted_values = [v + offset for v in values]

        bars = ax.bar(
            angles_rad,
            adjusted_values,
            width=np.deg2rad(1),  # increase for better visibility
            alpha=0.5,
            bottom=0.0,
            color=[
                "red" if v - offset > threshold else "blue" for v in adjusted_values
            ],
        )

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"Neuron {label}", pad=20)

    plt.tight_layout()
    plt.show()


def plot_multiple_curves(
    n_comp=11,
    lambda_um=200,
    angles=None,
    all_max_voltages=None,
    left_start_index=0,
    left_end_index=11,
):
    # use a diverging colormap without white. options are "managua", "berlin", "vanimo"
    # ideally, use the "opposite" colormap to the one used in the polar plots. i.e.:
    # berlin + RdBu, vanimo + PiYG, managua + BrBG
    colormap = "berlin"

    plt.figure(figsize=(12, 6))

    # pull linearly spaced colors from the colormap "managua"
    cmap = plt.get_cmap(colormap)
    colors = [
        cmap((idx) / (left_end_index - left_start_index))
        for idx in range(left_start_index, left_end_index + 1)
    ]
    # all_max_voltages is a dict where all_voltage_data[neuron_label] = (angles, max_voltages)
    # convert to a list of max voltages for each neuron:

    for left_index in range(left_start_index, left_end_index + 1):
        right_index = 2 * n_comp + 1 - left_index
        neuron_label = f"L{left_index}_R{right_index - n_comp}"
        angles, max_voltages = all_max_voltages[neuron_label]
        left_label = f"{left_index}L"
        right_label = f"{right_index - n_comp}R"

        max_voltages = np.array(max_voltages)
        smoothed_voltages = smooth_data(angles, max_voltages, window_size=20)

        plt.plot(
            angles,
            smoothed_voltages,
            label=f"{left_label}, {right_label}",
            color=colors[left_index - left_start_index],
        )
    plt.xlabel("Sound Angle (degrees)")
    plt.ylabel("Max Soma Voltage (mV)")
    plt.title("Interpolated Average Curves for Different Indices")
    plt.legend(loc="upper right", fontsize="small")
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_neuron_thresholds(
    thresholds_path,
    n_comp=11,
    left_start_index=0,
    left_end_index=11,
    title="Thresholds for Different Neurons",
    xlabel="Neuron Index",
    ylabel="Threshold (mV)",
):
    """Plot thresholds for different neurons."""
    indices = []
    values = []

    for left_index in range(left_start_index, left_end_index + 1):
        right_index = 2 * n_comp + 1 - left_index
        neuron_label = f"L{left_index}_R{right_index - n_comp}"
        threshold = load_thresholds(thresholds_path, left_index)
        if threshold is None:
            print(f"No threshold found for neuron {neuron_label}. Skipping.")
        else:
            indices.append(left_index)
            values.append(threshold)

    plt.figure(figsize=(10, 5))
    plt.plot(indices, values, marker="o", linestyle="-", color="b")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(indices)
    plt.grid()
    plt.tight_layout()
    plt.show()


# -- Storage and loading functions --
def store_response_per_angle(
    left_index, angles, all_voltages, max_voltages, spike_counts, filepath=None
):
    """Store the response data in a JSON file."""
    if filepath is None:
        filepath = Path(__file__).parent / "response_data.json"

    # Load existing data if file exists
    if Path(filepath).exists():
        with open(filepath, "r") as f:
            try:
                response_data = json.load(f)
            except json.JSONDecodeError:
                response_data = {}
    else:
        response_data = {}

    neuron_response = {}
    for i, angle in enumerate(angles):
        neuron_response[str(angle)] = {
            "max_voltage": max_voltages[i],
            "spike_count": int(spike_counts[i]),
            "all_voltages": all_voltages[i].tolist(),
        }

    response_data[str(left_index)] = neuron_response

    with open(filepath, "w") as f:
        json.dump(response_data, f, indent=4)

    # Reduced output - only show summary
    # print(f"Response data for left_idx {left_index} stored in {filepath}")


def load_response_per_angle(
    left_index, response_filepath=None, min_angle=1, max_angle=360, step=1
):
    """Load response data from a JSON file."""
    if not Path(response_filepath).exists():
        print(f"Response data file {response_filepath} does not exist.")
        return
    with open(response_filepath, "r") as f:
        angles, all_voltages, max_voltages, spike_counts = [], [], [], []
        try:
            response_data = json.load(f)
            if str(left_index) not in response_data:
                print(f"No data for left index {left_index} in response file.")
                return
            response_data = response_data[str(left_index)]
            for angle in range(min_angle, max_angle, step):
                if str(angle) in response_data:
                    data = response_data[str(angle)]
                    angles.append(angle)
                    all_voltages.append(np.array(data["all_voltages"]))
                    max_voltages.append(data["max_voltage"])
                    spike_counts.append(data["spike_count"])
                # Removed per-angle missing data warnings to reduce output
        except json.JSONDecodeError:
            print(
                "Response data file is empty or invalid. Please run the simulation first."
            )
            return
    return angles, all_voltages, max_voltages, spike_counts


def load_thresholds(filepath, l=None):
    """Load thresholds from a JSON file."""
    if not Path(filepath).exists():
        print(f"Thresholds file {filepath} does not exist.")
        return {}

    with open(filepath, "r") as f:
        try:
            thresholds = json.load(f)
            threshold = thresholds.get(str(l), None) if l is not None else thresholds
        except json.JSONDecodeError:
            thresholds = {}
            print("JSON file is empty or invalid. Starting fresh.")

    return threshold


def save_thresholds(thresholds, filepath):
    """Save thresholds to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(thresholds, f, indent=4)
    # Reduced output - only show when verbose
    # print(f"Thresholds saved to {filepath}")


if __name__ == "__main__":
    # test
    angle = 90
    arrival_times = calculate_arrival_times(angle)
    print(
        f"Arrival times: Left Ear = {arrival_times[0]:.6f}ms, Right Ear = {arrival_times[1]:.6f}ms"
    )
    print(f"Difference: {arrival_times[1] - arrival_times[0]:.6f}ms")
