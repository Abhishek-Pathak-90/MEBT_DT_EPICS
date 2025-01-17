#!/usr/bin/env python3
import os
import time
import sys
import hashlib
import traceback
import numpy as np

# For plotting:
try:
    import matplotlib
    matplotlib.use('TkAgg')  # or your preferred backend
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"Error setting up matplotlib: {e}")
    traceback.print_exc()
    sys.exit(1)

###############################################################################
#                       PART A: ORIGINAL WORKING LOGIC                         #
###############################################################################

def parse_lattice_file(lattice_filename):
    """
    YOUR UNCHANGED PARSE LOGIC:
    Reads the lattice, ignoring BPM lines or instrumentation markers,
    so the index/quad counting is the same as your 'last working version'.
    """
    elements = []
    element_index = 1
    quad_counter  = 1

    with open(lattice_filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            # Skip instrumentation markers, e.g. 'BPM :'
            if line.endswith(":") and len(line.split()) <= 2:
                # we do NOT parse BPM here => skip
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
                grad_str = parts[2].replace(";", "")
                try:
                    G_val = float(grad_str)
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
                dx_str = parts[1].replace(";", "")
                dy_str = parts[2].replace(";", "")
                n_str  = parts[3].replace(";", "")
                dx_mm = float(dx_str)
                dy_mm = float(dy_str)
                n_type = int(n_str)
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
    If G>0 => x-plane defocus => cosh/sinh
              y-plane focus   => cos/sin
    If G<0 => x-plane focus   => cos/sin
              y-plane defocus => cosh/sinh
    """
    M = np.eye(6)
    M[4,4] = 1.0
    M[4,5] = ds_m / (gamma**2)
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

    if G > 0:
        # x-plane defocus => cosh/sinh
        M[0,0] =  cosh_kl
        M[0,1] =  sinh_kl / k
        M[1,0] =  k*sinh_kl
        M[1,1] =  cosh_kl

        # y-plane focus => cos/sin
        M[2,2] =  cos_kl
        M[2,3] =  sin_kl / k
        M[3,2] = -k*sin_kl
        M[3,3] =  cos_kl
    else:
        # G<0 => x-plane focus => cos/sin
        M[0,0] =  cos_kl
        M[0,1] =  sin_kl / k
        M[1,0] = -k*sin_kl
        M[1,1] =  cos_kl

        # y-plane defocus => cosh/sinh
        M[2,2] =  cosh_kl
        M[2,3] =  sinh_kl / k
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
# PART B: SIGMA PROPAGATION (unchanged)
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
# PART C: PIPELINE => parse + update + propagate => get positions, x_rms, y_rms
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
    quad_dict   = load_quad_gradients(grad_file)
    updated_data= update_matrices(elements, matrix_data, quad_dict, brho, gamma)

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
    plot_positions= np.array(plot_positions)

    return plot_positions, x_rms, y_rms

###############################################################################
#    PART D: NEW FUNCTION => SECOND PASS TO PARSE BPM, track cumulative z
###############################################################################

def parse_bpm_positions(lattice_filename):
    """
    We do a second pass over the same 'mebt_final.dat', ignoring your original 
    indexing logic. We keep a separate 'z_current' for DRIFT/QUAD lengths, 
    so we can find where each 'BPM :' occurs.

    This DOES NOT affect the main pipeline. It's purely for BPM info.
    Returns (bpm_positions, bpm_names).
    """
    bpm_positions = []
    bpm_names     = []
    z_current     = 0.0
    element_index = 1
    quad_counter  = 1

    with open(lattice_filename,"r") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith(";"):
                continue
            parts=line.split()

            # if line ends with ':' and has <=2 tokens => instrumentation marker
            if line.endswith(":") and len(parts)<=2:
                # might be 'BPM :'
                if parts[0].upper()=="BPM":
                    # store zero length => BPM location = z_current
                    bpm_positions.append(z_current)
                    bpm_names.append(f"BPM_{element_index}")
                # skip other markers if any
                continue

            cmd = parts[0].upper()
            if cmd=="DRIFT":
                ds_mm = float(parts[1].replace(";", ""))
                ds_m  = ds_mm *1e-3
                z_current += ds_m

            elif cmd=="QUAD":
                ds_mm= float(parts[1].replace(";", ""))
                ds_m= ds_mm*1e-3
                z_current+= ds_m
                quad_counter +=1
            # etc. if there are other elements with lengths, we add to z_current

    return np.array(bpm_positions), bpm_names


###############################################################################
# PART E: MAIN => dynamic watch & plot => plus after the pipeline we do BPM plot
###############################################################################

def md5_of_file(filename):
    hasher = hashlib.md5()
    with open(filename,"rb") as f:
        buf=f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def dynamic_plot(lattice_file, matrix_file, grad_file, gamma, brho):
    old_hash = None

    plt.ion()
    fig, ax = plt.subplots()
    line_x, = ax.plot([], [], 'b-', label='x RMS')
    line_y, = ax.plot([], [], 'r-', label='y RMS')
    ax.set_xlabel("Longitudinal Position (m)")
    ax.set_ylabel("RMS Beam Size (mm)")
    ax.set_title("Dynamic RMS update + (Optional) BPM positions")
    ax.legend()
    ax.grid(True)

    # We'll do a separate figure for BPMs
    fig_bpm, ax_bpm = plt.subplots()
    line_bpmx, = ax_bpm.plot([], [], 'bo-', label='BPM x RMS')
    line_bpmy, = ax_bpm.plot([], [], 'ro-', label='BPM y RMS')
    ax_bpm.set_xlabel("BPM Position (m)")
    ax_bpm.set_ylabel("RMS (mm)")
    ax_bpm.set_title("RMS at BPM positions (interpolated)")
    ax_bpm.legend()
    ax_bpm.grid(True)

    while True:
        new_hash= md5_of_file(grad_file)
        if new_hash!= old_hash:
            old_hash= new_hash
            print("Gradient file content changed, re-running pipeline...")

            # (1) run your original pipeline => positions, x_rms, y_rms
            positions, x_rms, y_rms = run_pipeline_and_get_rms(
                lattice_file, matrix_file, grad_file, gamma, brho
            )

            # Print RMS values
            print("\nRMS Values:")
            print(f"X RMS (mm): min={min(x_rms)*1e3:.3f}, max={max(x_rms)*1e3:.3f}")
            print(f"Y RMS (mm): min={min(y_rms)*1e3:.3f}, max={max(y_rms)*1e3:.3f}\n")

            # update the main figure lines
            line_x.set_xdata(positions)
            line_x.set_ydata(x_rms*1e3)
            line_y.set_xdata(positions)
            line_y.set_ydata(y_rms*1e3)

            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()

            # (2) parse BPM positions in a *separate pass*
            bpm_positions, bpm_names = parse_bpm_positions(lattice_file)
            if len(bpm_positions)>0:
                # we do a linear interpolation to get x,y RMS at those positions
                bpm_x = np.interp(bpm_positions, positions, x_rms)
                bpm_y = np.interp(bpm_positions, positions, y_rms)

                line_bpmx.set_xdata(bpm_positions)
                line_bpmx.set_ydata(bpm_x*1e3)
                line_bpmy.set_xdata(bpm_positions)
                line_bpmy.set_ydata(bpm_y*1e3)

                ax_bpm.relim()
                ax_bpm.autoscale_view()
                fig_bpm.canvas.draw()
            else:
                print("No BPM found in the lattice, or no BPM : markers?")

        plt.pause(0.5)


def main():
    lattice_file  = "mebt_final.dat"
    matrix_file   = "Individual_matrix.dat"
    grad_file     = "quad_gradients.txt"
    gamma         = 1.0022599
    brho          = 0.21075678

    print("Starting dynamic watch with content-check for:", grad_file)
    dynamic_plot(lattice_file, matrix_file, grad_file, gamma, brho)

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
        sys.exit(1)
