# dendritic-delay
Project for the Neural Information Processing course at University of Osnabrück, summer semester of 2025.

## Installation

Install required packages:
```bash
pip install brian2 numpy matplotlib scipy tqdm pathlib
```

## Usage

Run the main script:
```bash
python conductance_based_synapse.py
```

### Configuration Flags

Edit flags in `main()` function to control behavior:

**Computation:**
- `calc_thresholds`: Calculate thresholds (False = load from file)
- `simulate_response`: Run response simulations (False = load from file)
- `use_mp`: Enable multiprocessing

**Single neuron plots:**
- `do_single_combo`: Show voltage traces for one neuron
- `polar_plot_spikes`: Polar plot of spike counts
- `polar_plot_max_voltages`: Polar plot of max voltages

**Multi-neuron plots:**
- `polar_plot_v_grid`: Grid of voltage polar plots
- `polar_plot_spk_grid`: Grid of spike polar plots
- `polar_plot_v_multi`: Combined voltage polar plot
- `polar_plot_spk_multi`: Combined spike polar plot
- `multiple_curves`: Line plot of voltage curves

### Parameters

Key parameters in `main()`:
- `n_comp`: Number of compartments per dendrite
- `lambda_um`: Space constant in micrometers
- `left_start_index`, `left_end_index`: Range of neurons to simulate
- `min_angle`, `max_angle`, `step`: Sound angle range and resolution

### Output Files

- `thresholds.json`: Calculated spike thresholds
- `response_data.json`: Simulation results

### Workflow

1. Set `calc_thresholds=True` and `simulate_response=False` to calculate thresholds
2. Set `calc_thresholds=False` and `simulate_response=True` to run simulations
3. Set both to `False` and enable plot flags to generate plots from saved data
´

### Notes
Simulation is computationally heavy and generates a large file. All relevant info (including all membrane voltages) will be stored. Use the generated file instead of recalculating the neuron responses whenever it is possible.