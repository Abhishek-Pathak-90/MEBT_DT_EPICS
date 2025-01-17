#!/usr/bin/env python3
import sys
import traceback
import numpy as np
import time
import matplotlib
import os

# We'll use a non-blocking backend
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt

# We use watchdog to watch the file
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

###############################################################################
# PART A: The lattice & quad update logic (same as before)
###############################################################################

def parse_lattice_file(lattice_filename):
    elements = []
    element_index = 1
    quad_counter  = 1

    with open(lattice_filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.endswith(":") and len(line.split()) <= 2:
                continue

            parts = line.split()
            cmd = parts[0].upper()

            if cmd == "DRIFT":
                ds_mm = float(parts[1].replace(";", ""))
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
                ds_mm = float(parts[1].replace(";", ""))
                ds_m  = ds_mm * 1e-3
                try:
                    G_val = float(parts[2].replace(";", ""))
                except ValueError:
                    G_val = 0.0

                qg_name = f"QG{quad_counter}"
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
                quad_counter  += 1

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
                    "parameters": extra_params
                })
                element_index += 1
            else:
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

def load_quad_gradients(gradient_file):
    grad_dict = {}
    with open(gradient_file, "r") as gf:
        for line in gf:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.replace('=', ' ')
            parts = line.split()
            if len(parts) >= 2:
                qname = parts[0]
                gval  = float(parts[1])
                grad_dict[qname] = gval
    return grad_dict

def compute_quad_matrix_swapped(ds_m, G, brho, gamma):
    """
    Swapped convention:
      G>0 => x-plane defocus => cosh/sinh, y-plane focus => cos/sin
      G<0 => x-plane focus => cos/sin, y-plane defocus => cosh/sinh
    """
    M = np.eye(6)
    M[4,4] = 1.0
    M[4,5] = ds_m/(gamma**2)
    M[5,5] = 1.0

    tiny = 1e-14
    if abs(G) < tiny:
        M[0,1] = ds_m
        M[2,3] = ds_m
        return M

    if abs(brho) < 1e-12:
        raise ValueError("Beam rigidity brho is too small or zero!")

    k  = np.sqrt(abs(G)/brho)
    kl = k*ds_m

    cos_kl  = np.cos(kl)
    sin_kl  = np.sin(kl)
    cosh_kl = np.cosh(kl)
    sinh_kl = np.sinh(kl)

    if G>0:
        # x-plane => defocus => cosh/sinh
        M[0,0] =  cosh_kl
        M[0,1] =  sinh_kl / k
        M[1,0] =  k*sinh_kl
        M[1,1] =  cosh_kl

        # y-plane => focus => cos/sin
        M[2,2] =  cos_kl
        M[2,3] =  sin_kl / k
        M[3,2] = -k*sin_kl
        M[3,3] =  cos_kl
    else:
        # G<0 => x-plane => focus => cos/sin
        M[0,0] =  cos_kl
        M[0,1] =  sin_kl/k
        M[1,0] = -k*sin_kl
        M[1,1] =  cos_kl

        # y-plane => defocus => cosh/sinh
        M[2,2] =  cosh_kl
        M[2,3] =  sinh_kl/k
        M[3,2] =  k*sinh_kl
        M[3,3] =  cosh_kl
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

###############################################################################
# PART B: SIGMA PROPAGATION
###############################################################################

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

###############################################################################
# PART C: A single function to run the pipeline => returns (positions, x_rms, y_rms)
###############################################################################

def run_pipeline_and_get_rms(
    lattice_file="mebt_final.dat",
    matrix_file="Individual_matrix.dat",
    grad_file="quad_gradients.txt",
    gamma=1.0022599,
    brho=0.21075678
):
    elements = parse_lattice_file(lattice_file)
    matrix_data = read_individual_matrices(matrix_file)
    quad_dict = load_quad_gradients(grad_file)
    updated_data = update_matrices(elements, matrix_data, quad_dict, brho, gamma)

    # Build M_list, pos_list
    M_list  = []
    pos_list= []
    for item in updated_data:
        M_list.append(item["matrix"])
        pos_list.append(item["pos"])

    # Propagate sigma
    sigma0 = define_initial_sigma()
    sigma_all = propagate_sigma_through_lattice(sigma0, M_list)
    x_rms, y_rms = compute_rms_sizes(sigma_all)
    plot_positions = [0.0]
    plot_positions.extend(pos_list)

    return plot_positions, x_rms, y_rms

###############################################################################
# PART D: WATCHDOG FILE HANDLER => triggers pipeline re-run only when changed
###############################################################################

class GradFileHandler(FileSystemEventHandler):
    def __init__(self, lattice_file, matrix_file, grad_file, gamma, brho,
                 line_x, line_y, ax):
        super().__init__()
        self.lattice_file = lattice_file
        self.matrix_file  = matrix_file
        self.grad_file    = grad_file
        self.gamma        = gamma
        self.brho         = brho
        self.line_x       = line_x
        self.line_y       = line_y
        self.ax           = ax

    def on_modified(self, event):
        # Check if the changed file is our 'grad_file'
        if not event.is_directory and event.src_path.endswith(self.grad_file):
            print("quad_gradients file changed => re-run pipeline.")
            positions, x_rms, y_rms = run_pipeline_and_get_rms(
                self.lattice_file,
                self.matrix_file,
                self.grad_file,
                self.gamma,
                self.brho
            )
            self.line_x.set_xdata(positions)
            self.line_x.set_ydata(x_rms*1e3)
            self.line_y.set_xdata(positions)
            self.line_y.set_ydata(y_rms*1e3)

            self.ax.relim()
            self.ax.autoscale_view()
            self.ax.figure.canvas.draw_idle()  # trigger redraw
            # figure should remain in background => no continuous popup


###############################################################################
# PART E: MAIN => sets up the figure + watchdog observer
###############################################################################

def main():
    import time
    from watchdog.observers import Observer

    # Filenames
    lattice_file = "mebt_final.dat"
    matrix_file  = "Individual_matrix.dat"
    grad_file    = "quad_gradients.txt"
    gamma        = 1.0022599
    brho         = 0.21075678

    # Create a figure once (non-blocking)
    plt.ion()
    fig, ax = plt.subplots()
    line_x, = ax.plot([], [], 'b-', label='x RMS')
    line_y, = ax.plot([], [], 'r-', label='y RMS')
    ax.set_xlabel("Longitudinal Position (m)")
    ax.set_ylabel("RMS Beam Size (mm)")
    ax.set_title("Watchdog-based approach => re-run pipeline only on file change")
    ax.legend()
    ax.grid(True)
    plt.show(block=False)

    # Start by doing an initial run
    positions, x_rms, y_rms = run_pipeline_and_get_rms(
        lattice_file, matrix_file, grad_file, gamma, brho
    )
    line_x.set_xdata(positions)
    line_x.set_ydata(x_rms*1e3)
    line_y.set_xdata(positions)
    line_y.set_ydata(y_rms*1e3)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

    # Create the event handler
    event_handler = GradFileHandler(
        lattice_file, matrix_file, grad_file,
        gamma, brho,
        line_x, line_y, ax
    )

    # Create an observer for the directory containing 'grad_file'
    directory_to_watch = os.path.dirname(os.path.abspath(grad_file))
    observer = Observer()
    observer.schedule(event_handler, path=directory_to_watch, recursive=False)
    observer.start()
    print(f"Watching {grad_file} in directory: {directory_to_watch}")

    try:
        # We'll do a small loop so we can keep the UI alive
        while True:
            plt.pause(0.1)  # process UI events
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error in main:", e)
        traceback.print_exc()
        sys.exit(1)
