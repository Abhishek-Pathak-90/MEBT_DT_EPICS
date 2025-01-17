import numpy as np
import matplotlib
import sys
import traceback

try:
    matplotlib.use('TkAgg')  # Set backend before importing pyplot
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"Error setting up matplotlib: {e}")
    traceback.print_exc()
    sys.exit(1)

def define_initial_sigma():
    """
    Define the initial 6x6 beam sigma matrix.
    Order: (x, x', y, y', z, dP/P).
    """
    # Fill in with your actual initial values from the screenshot:
    sigma = np.array([
        [ 9.8651981e-07, -3.8336909e-06,  0.0,            0.0,            0.0,            0.0           ],
        [-3.8336909e-06,  2.4777439e-05,  0.0,            0.0,            0.0,            0.0           ],
        [ 0.0,            0.0,            3.5277449e-07,  2.9781136e-07,  0.0,            0.0           ],
        [ 0.0,            0.0,            2.9781136e-07,  2.7878829e-05,  0.0,            0.0           ],
        [ 0.0,            0.0,            0.0,            0.0,            6.0380809e-06,  0.0           ],
        [ 0.0,            0.0,            0.0,            0.0,            0.0,            4.2311437e-06]
    ])
    return sigma

def read_individual_matrices_with_positions(filename):
    """
    Reads the 6x6 individual element transfer matrices from 'filename',
    each preceded by a line "ELE# n : X m", where X is the *end position*.

    Returns:
        M_list   (list of 6x6 ndarrays): The element transfer matrices.
        pos_list (list of float)       : The position at the end of each element.
    """
    M_list = []
    pos_list = []

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("ELE#"):
            # Example: "ELE# 1 : 0.25 m"
            parts = line.split()  # ["ELE#", "1", ":", "0.25", "m"]
            pos_val = float(parts[3])  # "0.25"
            pos_list.append(pos_val)

            # Next 6 lines for the matrix
            mat_rows = []
            for row_idx in range(6):
                i += 1
                row_data_str = lines[i].strip().split()
                row_data = [float(x) for x in row_data_str]
                mat_rows.append(row_data)
            M = np.array(mat_rows)
            M_list.append(M)
        i += 1

    return M_list, pos_list

def propagate_sigma_through_lattice(sigma0, M_list):
    """
    Propagate the beam sigma matrix through each element:
      sigma_out = M * sigma_in * M^T.
    Return a list of sigma matrices for each station:
      [sigma_start, sigma_after_1, sigma_after_2, ...].
    """
    sigma_all = []
    sigma_current = sigma0.copy()
    sigma_all.append(sigma_current)  # sigma at the very start

    for M in M_list:
        sigma_next = M @ sigma_current @ M.T
        sigma_all.append(sigma_next)
        sigma_current = sigma_next

    return sigma_all

def compute_rms_sizes(sigma_list):
    """
    For each sigma in sigma_list, compute rms_x = sqrt(sigma(0,0)),
    and rms_y = sqrt(sigma(2,2)).
    """
    x_rms_list = []
    y_rms_list = []
    for sigma in sigma_list:
        x_rms = np.sqrt(abs(sigma[0,0]))
        y_rms = np.sqrt(abs(sigma[2,2]))
        x_rms_list.append(x_rms)
        y_rms_list.append(y_rms)
    return np.array(x_rms_list), np.array(y_rms_list)

def main():
    # 1. Initial sigma
    sigma0 = define_initial_sigma()

    # 2. Read the *individual* element transfer matrices, and positions
    transfer_file = "Individual_matrix_updated.dat"  # or your filename
    M_list, pos_list = read_individual_matrices_with_positions(transfer_file)

    # 3. Propagate sigma
    sigma_all = propagate_sigma_through_lattice(sigma0, M_list)

    # 4. Compute RMS x and y
    x_rms, y_rms = compute_rms_sizes(sigma_all)

    # 5. Build the positions array for plotting:
    #    sigma_all has length = #elements + 1.
    #    We have pos_list for the end of each element, so len(pos_list) == #elements.
    #    The first sigma_all entry is the start (z=0), then after element 1, we are at pos_list[0], etc.
    plot_positions = [0.0]  # start is at z=0
    plot_positions.extend(pos_list)  # now we have N more positions for the ends of elements

    # Sanity check: len(plot_positions) == len(sigma_all)
    # Both should be #elements + 1
    assert len(plot_positions) == len(sigma_all), "Positions array does not match sigma arrays length."

    # 6. Plot
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(plot_positions, x_rms*1e3, 'b-', label='x RMS')
        plt.plot(plot_positions, y_rms*1e3, 'r-', label='y RMS')
        plt.xlabel("Longitudinal Position (m)")
        plt.ylabel("RMS Beam Size (mm)")
        plt.title("RMS x and y along the Linac")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        print("Attempting to show plot...")
        plt.show()
        print("Plot shown successfully")
    except Exception as e:
        print(f"Error during plotting: {e}")
        traceback.print_exc()
        # Save the plot as a fallback
        try:
            plt.savefig('beam_sizes.png')
            print("Plot saved as 'beam_sizes.png'")
        except Exception as save_error:
            print(f"Error saving plot: {save_error}")
        plt.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
        sys.exit(1)
