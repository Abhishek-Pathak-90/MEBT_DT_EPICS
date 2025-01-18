#!/usr/bin/env python3

import os
import time
import sys
import traceback
import numpy as np
import threading
import random

try:
    import matplotlib
    matplotlib.use('TkAgg')  # or your preferred backend
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"Error setting up matplotlib: {e}")
    traceback.print_exc()
    sys.exit(1)

import pvaccess

# -----------------------------------------------------------------------------
# 1. Imports from your custom modules
#    (These define how we get quad/bpm dictionaries and channel references)
# -----------------------------------------------------------------------------
from ble import ble_quadrupoles
from bli import bli_bpms

# Obtain lists/dicts of quads and BPMs
quad_list, quad_dict = ble_quadrupoles()
bpm_list, bpmxpos_dict, bpmxrms_dict, bpmypos_dict, bpmyrms_dict = bli_bpms()

# DEBUG Print to confirm we have matching BPM dictionary keys
print("DEBUG: quad_list =>", quad_list)
print("DEBUG: bpm_list =>", bpm_list)
print("DEBUG: bpmxpos_dict.keys =>", list(bpmxpos_dict.keys()))
print("DEBUG: bpmxrms_dict.keys =>", list(bpmxrms_dict.keys()))
print("DEBUG: bpmypos_dict.keys =>", list(bpmypos_dict.keys()))
print("DEBUG: bpmyrms_dict.keys =>", list(bpmyrms_dict.keys()))

# Give channels time to connect
time.sleep(1.0)

# -----------------------------------------------------------------------------
# 2. Global data structures for new pipeline results, protected by a lock
# -----------------------------------------------------------------------------
grad_dict = {}  # holds current quad gradients from EPICS
epics_data_lock = threading.Lock()
new_data_ready = False
new_positions = None
new_x_rms = None
new_y_rms = None

# -----------------------------------------------------------------------------
# 3. Lattice & Matrix Parsing
# -----------------------------------------------------------------------------
def parse_lattice_file(lattice_filename):
    elements = []
    element_index = 1
    quad_counter  = 1

    with open(lattice_filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            # Skip instrumentation lines that end with ":" if they have <=2 tokens
            if line.endswith(":") and len(line.split()) <= 2:
                continue

            parts = line.split()
            cmd = parts[0].upper()

            if cmd == "DRIFT":
                ds_mm_str = parts[1].replace(";", "")
                ds_mm = float(ds_mm_str)
                ds_m  = ds_mm * 1e-3
                elements.append({
                    "index": element_index,
                    "etype": "DRIFT",
                    "length": ds_m,
                    "gradient": None,
                    "dx": None,
                    "dy": None,
                    "atype": None,
                    "name": f"DRIFT_{element_index}",
                    "parameters": []
                })
                element_index += 1

            elif cmd == "QUAD":
                ds_mm_str = parts[1].replace(";", "")
                ds_mm = float(ds_mm_str)
                ds_m  = ds_mm * 1e-3
                try:
                    G_val = float(parts[2].replace(";", ""))
                except ValueError:
                    G_val = 0.0

                qg_name = quad_list[quad_counter - 1]
                elements.append({
                    "index": element_index,
                    "etype": "QUAD",
                    "length": ds_m,
                    "gradient": G_val,
                    "dx": None,
                    "dy": None,
                    "atype": None,
                    "name": qg_name,
                    "parameters": []
                })
                element_index += 1
                quad_counter += 1

            elif cmd == "APERTURE":
                dx_mm = float(parts[1].replace(";", ""))
                dy_mm = float(parts[2].replace(";", ""))
                n_type = int(parts[3].replace(";", ""))
                elements.append({
                    "index": element_index,
                    "etype": "APERTURE",
                    "length": 0.0,
                    "gradient": None,
                    "dx": dx_mm,
                    "dy": dy_mm,
                    "atype": n_type,
                    "name": f"APERTURE_{element_index}",
                    "parameters": []
                })
                element_index += 1

            elif cmd == "FIELD_MAP":
                extra_params = parts[1:]
                elements.append({
                    "index": element_index,
                    "etype": "FIELD_MAP",
                    "length": 0.0,
                    "gradient": None,
                    "dx": None,
                    "dy": None,
                    "atype": None,
                    "name": f"FIELD_MAP_{element_index}",
                    "parameters": extra_params
                })
                element_index += 1

            elif cmd == "THIN_STEERING":
                extra_params = parts[1:]
                elements.append({
                    "index": element_index,
                    "etype": "THIN_STEERING",
                    "length": 0.0,
                    "gradient": None,
                    "dx": None,
                    "dy": None,
                    "atype": None,
                    "name": f"THIN_STEERING_{element_index}",
                    "parameters": []
                })
                element_index += 1
            else:
                # Unrecognized => skip
                pass

    return elements

def read_individual_matrices(filename):
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("ELE#"):
            parts = line.split()
            elem_idx = int(parts[1])
            pos_m    = float(parts[3])
            mat_rows = []
            for r in range(6):
                i += 1
                row_str = lines[i].strip().split()
                row_data = [float(x) for x in row_str]
                mat_rows.append(row_data)
            M = np.array(mat_rows)
            data.append({
                "index": elem_idx,
                "pos":   pos_m,
                "matrix": M
            })
        i += 1
    return data

# -----------------------------------------------------------------------------
# 4. Quad Matrix Calculation
# -----------------------------------------------------------------------------
def compute_quad_matrix_swapped(ds_m, G, brho, gamma):
    """
    Creates a 6x6 transport matrix for a quadrupole,
    with the 'swapped' convention: G>0 => x-plane is defocusing, etc.
    """
    M = np.eye(6)
    M[4,4] = 1.0
    M[4,5] = ds_m / (gamma**2)
    M[5,5] = 1.0

    tiny = 1e-14
    if abs(G) < tiny:
        # Approx drift
        M[0,1] = ds_m
        M[2,3] = ds_m
        return M

    if abs(brho) < 1e-12:
        raise ValueError("Beam rigidity brho is too small or zero!")

    k = np.sqrt(abs(G) / brho)
    kl= k * ds_m

    cos_kl  = np.cos(kl)
    sin_kl  = np.sin(kl)
    cosh_kl = np.cosh(kl)
    sinh_kl = np.sinh(kl)

    if G > 0:
        # x-plane => defocus => cosh/sinh
        M[0,0] = cosh_kl
        M[0,1] = sinh_kl / k
        M[1,0] = k*sinh_kl
        M[1,1] = cosh_kl
        # y-plane => focus => cos/sin
        M[2,2] = cos_kl
        M[2,3] = sin_kl / k
        M[3,2] = -k*sin_kl
        M[3,3] = cos_kl
    else:
        # G<0 => x-plane => focus => cos/sin
        M[0,0] = cos_kl
        M[0,1] = sin_kl / k
        M[1,0] = -k*sin_kl
        M[1,1] = cos_kl
        # y-plane => defocus => cosh/sinh
        M[2,2] = cosh_kl
        M[2,3] = sinh_kl / k
        M[3,2] = k*sinh_kl
        M[3,3] = cosh_kl

    return M

def update_matrices(elements, matrix_data, quad_grad_dict, brho, gamma):
    updated = []
    for elem, md in zip(elements, matrix_data):
        e_idx   = elem["index"]
        e_type  = elem["etype"]
        old_mat = md["matrix"]
        pos_m   = md["pos"]
        ds_m    = elem["length"]

        if e_type == "QUAD":
            qname = elem["name"]
            if qname in quad_grad_dict:
                new_grad = quad_grad_dict[qname]
            else:
                new_grad = elem["gradient"]

            M_new = compute_quad_matrix_swapped(ds_m, new_grad, brho, gamma)
            updated.append({
                "index": e_idx,
                "pos":   pos_m,
                "matrix": M_new
            })
        else:
            updated.append({
                "index": e_idx,
                "pos":   pos_m,
                "matrix": old_mat
            })
    return updated

# -----------------------------------------------------------------------------
# 5. Sigma Matrix Propagation
# -----------------------------------------------------------------------------
def define_initial_sigma():
    sigma = np.array([
        [ 9.8651981e-07, -3.8336909e-06,  0.0,            0.0,            0.0,            0.0 ],
        [-3.8336909e-06,  2.4777439e-05,  0.0,            0.0,            0.0,            0.0 ],
        [ 0.0,            0.0,            3.5277449e-07,  2.9781136e-07,  0.0,            0.0 ],
        [ 0.0,            0.0,            2.9781136e-07,  2.7878829e-05,  0.0,            0.0 ],
        [ 0.0,            0.0,            0.0,            0.0,            6.0380809e-06,  0.0 ],
        [ 0.0,            0.0,            0.0,            0.0,            0.0,            4.2311437e-06]
    ])
    return sigma

def propagate_sigma_through_lattice(sigma0, M_list):
    sigma_all = []
    sigma_current = sigma0.copy()
    sigma_all.append(sigma_current)

    for M in M_list:
        sigma_next = M @ sigma_current @ M.T
        sigma_all.append(sigma_next)
        sigma_current = sigma_next

    return sigma_all

def compute_rms_sizes(sigma_list):
    x_rms_list = []
    y_rms_list = []
    for sigma in sigma_list:
        x_rms = np.sqrt(abs(sigma[0,0]))
        y_rms = np.sqrt(abs(sigma[2,2]))
        x_rms_list.append(x_rms)
        y_rms_list.append(y_rms)
    return np.array(x_rms_list), np.array(y_rms_list)

# -----------------------------------------------------------------------------
# 6. The Pipeline => parse, update, propagate => get (positions, x_rms, y_rms)
# -----------------------------------------------------------------------------
def run_pipeline_and_get_rms(
    lattice_file="mebt_final.dat",
    matrix_file="Individual_matrix.dat",
    gamma=1.0022599,
    brho=0.21075678
):
    elements    = parse_lattice_file(lattice_file)
    matrix_data = read_individual_matrices(matrix_file)

    updated_data= update_matrices(elements, matrix_data, grad_dict, brho, gamma)

    M_list  = []
    pos_list= []
    for item in updated_data:
        M_list.append(item["matrix"])
        pos_list.append(item["pos"])

    sigma0 = define_initial_sigma()
    sigma_all = propagate_sigma_through_lattice(sigma0, M_list)
    x_rms, y_rms = compute_rms_sizes(sigma_all)

    plot_positions = [0.0]
    plot_positions.extend(pos_list)
    plot_positions= np.array(plot_positions)

    # Put BPM data to illustrate updates (writes to EPICS)
    n_bpm = len(bpm_list)
    if len(x_rms) < n_bpm or len(y_rms) < n_bpm:
        print(f"WARNING: x_rms or y_rms shorter than bpm_list length.")
    for ndx in range(n_bpm):
        bpm_name = bpm_list[ndx]
        # x/y RMS
        if bpm_name in bpmxrms_dict and bpm_name in bpmyrms_dict:
            bpmxrms_dict[bpm_name].putFloat(x_rms[ndx])
            bpmyrms_dict[bpm_name].putFloat(y_rms[ndx])
        else:
            print(f"WARNING: {bpm_name} not in bpmxrms_dict or bpmyrms_dict.")
        # x/y centroid
        if bpm_name in bpmxpos_dict and bpm_name in bpmypos_dict:
            bpmxpos_dict[bpm_name].putFloat(2*(random.random()-0.5))
            bpmypos_dict[bpm_name].putFloat(2*(random.random()-0.5))
        else:
            print(f"WARNING: {bpm_name} not in bpmxpos_dict or bpmypos_dict.")

    return plot_positions, x_rms, y_rms

# -----------------------------------------------------------------------------
# 7. parse_bpm_positions => for plotting BPM marker lines
# -----------------------------------------------------------------------------
def parse_bpm_positions(lattice_filename):
    bpm_positions = []
    bpm_names     = []
    z_current     = 0.0
    element_index = 1
    quad_counter  = 1

    with open(lattice_filename, "r") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith(";"):
                continue
            # If line ends with ':' => instrumentation marker
            if line.endswith(":") and len(line.split())<=2:
                if parts := line.split():
                    if parts[0].upper()=="BPM":
                        bpm_positions.append(z_current)
                        bpm_names.append(f"BPM_{element_index}")
                continue

            cmd = line.split()[0].upper()
            if cmd=="DRIFT":
                ds_mm = float(line.split()[1].replace(";", ""))
                ds_m  = ds_mm*1e-3
                z_current+=ds_m
            elif cmd=="QUAD":
                ds_mm = float(line.split()[1].replace(";", ""))
                ds_m  = ds_mm*1e-3
                z_current+=ds_m
                quad_counter+=1
            # etc. for other elements with length

    return np.array(bpm_positions), bpm_names

# -----------------------------------------------------------------------------
# 8. EPICS Callback => for quadrupole changes
# -----------------------------------------------------------------------------
def copyPV(pv):
    """
    Called whenever a quadrupole PV changes.
    We'll parse the quad name & value from the PV,
    store them in grad_dict, re-run the pipeline,
    store new results so main loop updates the plot.
    """
    global new_data_ready, new_positions, new_x_rms, new_y_rms

    if "display" in pv and "description" in pv["display"]:
        desc_field = pv["display"]["description"]
        quad_name = desc_field.split(":")[0].strip()
    else:
        quad_name = None
        print("No 'description' field in 'display' substructure for PV")

    if "value" in pv:
        quad_grad_value = pv["value"]
    else:
        quad_grad_value = None
        print("No gradient value found in 'pv' structure")

    if quad_name is not None and quad_grad_value is not None:
        print(f"EPICS: Quad {quad_name} => {quad_grad_value:.4f}")
        grad_dict[quad_name] = quad_grad_value

    # Re-run pipeline
    positions, x_rms, y_rms = run_pipeline_and_get_rms("mebt_final.dat", "Individual_matrix.dat")

    # Store for main loop to update the plot
    with epics_data_lock:
        new_positions = positions
        new_x_rms     = x_rms
        new_y_rms     = y_rms
        new_data_ready= True

# -----------------------------------------------------------------------------
# 9. MAIN => No file watcher, only EPICS updates
# -----------------------------------------------------------------------------
def main():
    # You can parametrize these if needed:
    lattice_file = "mebt_final.dat"
    matrix_file  = "Individual_matrix.dat"
    gamma        = 1.0022599
    brho         = 0.21075678

    # Subscribe to each quadrupole's channel => copyPV callback
    print("Starting EPICS monitor for quadrupole gradient changes...")
    for quad in quad_dict:
        quad_dict[quad].subscribe("copyPV", copyPV)
        quad_dict[quad].startMonitor()

    # Prepare Matplotlib figures
    plt.ion()

    # Figure 1: RMS vs. full lattice
    fig, ax = plt.subplots()
    line_x, = ax.plot([], [], 'b-', label='x RMS')
    line_y, = ax.plot([], [], 'r-', label='y RMS')
    ax.set_xlabel("Longitudinal Position (m)")
    ax.set_ylabel("RMS Beam Size (mm)")
    ax.set_title("Dynamic RMS update from EPICS quad changes")
    ax.legend()
    ax.grid(True)
    plt.show(block=False)

    # Figure 2: BPM interpolation
    fig_bpm, ax_bpm = plt.subplots()
    line_bpmx, = ax_bpm.plot([], [], 'bo-', label='BPM x RMS')
    line_bpmy, = ax_bpm.plot([], [], 'ro-', label='BPM y RMS')
    ax_bpm.set_xlabel("BPM Position (m)")
    ax_bpm.set_ylabel("RMS (mm)")
    ax_bpm.set_title("RMS at BPM positions (interpolated)")
    ax_bpm.legend()
    ax_bpm.grid(True)
    plt.show(block=False)

    # Initial pipeline run
    positions, x_rms, y_rms = run_pipeline_and_get_rms(lattice_file, matrix_file, gamma, brho)
    line_x.set_xdata(positions)
    line_x.set_ydata(x_rms * 1e3)
    line_y.set_xdata(positions)
    line_y.set_ydata(y_rms * 1e3)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

    # BPM interpolation
    bpm_positions, bpm_names = parse_bpm_positions(lattice_file)
    if len(bpm_positions) > 0:
        bpm_x = np.interp(bpm_positions, positions, x_rms)
        bpm_y = np.interp(bpm_positions, positions, y_rms)
        line_bpmx.set_xdata(bpm_positions)
        line_bpmx.set_ydata(bpm_x * 1e3)
        line_bpmy.set_xdata(bpm_positions)
        line_bpmy.set_ydata(bpm_y * 1e3)
        ax_bpm.relim()
        ax_bpm.autoscale_view()
        fig_bpm.canvas.draw_idle()
    else:
        print("No BPM found or no BPM : markers?")

    # Main loop => check EPICS-based updates
    try:
        while True:
            time.sleep(0.1)     # yield CPU, ~10 Hz loop
            plt.pause(0.001)   # update GUI

            # Check if new data from EPICS
            local_new_positions = None
            global epics_data_lock, new_data_ready
            with epics_data_lock:
                if new_data_ready:
                    local_new_positions = new_positions
                    local_new_x_rms     = new_x_rms
                    local_new_y_rms     = new_y_rms
                    new_data_ready      = False

            # If new data => update plots
            if local_new_positions is not None:
                line_x.set_xdata(local_new_positions)
                line_x.set_ydata(local_new_x_rms * 1e3)
                line_y.set_xdata(local_new_positions)
                line_y.set_ydata(local_new_y_rms * 1e3)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw_idle()

                if len(bpm_positions) > 0:
                    bpm_x = np.interp(bpm_positions, local_new_positions, local_new_x_rms)
                    bpm_y = np.interp(bpm_positions, local_new_positions, local_new_y_rms)
                    line_bpmx.set_xdata(bpm_positions)
                    line_bpmx.set_ydata(bpm_x * 1e3)
                    line_bpmy.set_xdata(bpm_positions)
                    line_bpmy.set_ydata(bpm_y * 1e3)
                    ax_bpm.relim()
                    ax_bpm.autoscale_view()
                    fig_bpm.canvas.draw_idle()

    except KeyboardInterrupt:
        print("Exiting main loop.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
        sys.exit(1)
