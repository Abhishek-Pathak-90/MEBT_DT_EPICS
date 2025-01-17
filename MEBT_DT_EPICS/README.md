# MEBT Digital Twin EPICS Implementation

This document provides a detailed explanation of the MEBT Digital Twin implementation using EPICS. The code is designed to:
- Parse a beamline lattice
- Compute beam RMS sizes
- Update EPICS PVs (for Phoebus visualization)
- Plot results in Python
- React to file changes and EPICS callbacks

## Code Structure and Components

### 1. Imports and Setup
```python
import os, time, sys, traceback, numpy as np, threading, random
import matplotlib, pvaccess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
```
- Basic Python modules for file operations, system functions
- numpy for numerical computations
- threading for safe data handling
- matplotlib for plotting
- pvaccess for EPICS integration
- watchdog for file monitoring

### 2. Global Data from External Modules
```python
from ble import ble_quadrupoles
from bli import bli_bpms
```
Provides:
- quad_list: List of quadrupole names
- quad_dict: Maps quad names to pvaccess.Channel
- bpm_list: List of BPM names
- bpmxpos_dict, bpmxrms_dict, bpmypos_dict, bpmyrms_dict: BPM channel dictionaries

### 3. Core Functions

#### Lattice and Matrix Parsing
- `parse_lattice_file(lattice_filename)`: Reads lattice elements
- `read_individual_matrices(filename)`: Parses transfer matrices
- `compute_quad_matrix_swapped(ds_m, G, brho, gamma)`: Calculates quad transport matrices

#### Beam Propagation
- `define_initial_sigma()`: Sets initial beam parameters
- `propagate_sigma_through_lattice(sigma0, M_list)`: Propagates beam through elements
- `compute_rms_sizes(sigma_list)`: Extracts RMS sizes

### 4. Main Pipeline
`run_pipeline_and_get_rms()`:
1. Parses lattice and matrices
2. Updates quad matrices
3. Propagates sigma matrix
4. Computes RMS sizes
5. Updates EPICS PVs
6. Returns data for plotting

### 5. File Monitoring
`GradFileHandler` class:
- Watches for changes in gradient files
- Triggers pipeline re-runs
- Updates plots and EPICS PVs

### 6. EPICS Integration
- Monitors quad gradient changes via EPICS
- Updates BPM PVs with computed values
- Provides real-time feedback to control system

### 7. Visualization
- Real-time plotting of beam sizes
- Separate plots for full lattice and BPM positions
- Interactive matplotlib interface

## Usage

1. Ensure EPICS environment is configured
2. Run the main script:
```bash
python MEBT_BPM_RMS.py
```

## Data Flow

1. Gradient changes can come from:
   - File modifications (quad_gradients.txt)
   - EPICS channel updates
2. Pipeline recomputes beam properties
3. Results are:
   - Written to EPICS PVs
   - Displayed in matplotlib plots
4. External GUIs (e.g., Phoebus) see updated values

## Notes

- Check timeouts and PV connections if updates aren't visible
- Verify dictionary keys match IOC PV names
- Allow time for channel connections
- Code can be extended for additional physics or monitoring
