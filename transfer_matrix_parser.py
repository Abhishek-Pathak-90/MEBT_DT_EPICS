import numpy as np

def read_cumulative_matrices(filename):
    """
    Reads the 6x6 cumulative transfer matrices from the specified file.
    Returns:
        matrices (list of numpy.ndarray): List of 6x6 arrays for each element
        element_info (list of dict): List of dictionaries with keys
                                     'element_number' and 'length'
    """
    matrices = []
    element_info = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for the line like: "ELE# 1 : 0.25 m"
        if line.startswith("ELE#"):
            parts = line.split()  # e.g. ["ELE#", "1", ":", "0.25", "m"]
            elem_num = int(parts[1])    # "1" -> 1
            length_str = parts[3]       # "0.25"
            length_val = float(length_str)

            # Store element info
            element_info.append({
                'element_number': elem_num,
                'length': length_val
            })

            # Next 6 lines describe the 6x6 matrix
            mat_rows = []
            for row_idx in range(6):
                i += 1
                row_data_str = lines[i].strip().split()
                row_data = [float(x) for x in row_data_str]
                mat_rows.append(row_data)

            # Convert to numpy array (6x6)
            mat_cum = np.array(mat_rows)
            matrices.append(mat_cum)
        i += 1

    return matrices, element_info


def factor_out_individual_matrices(matrices):
    """
    Given a list of cumulative matrices:
        M_cum(1), M_cum(2), ..., M_cum(N),
    with the convention:
        M_cum(n) = M_elem(n) * M_cum(n-1),
    we get:
        M_elem(n) = M_cum(n) * inv(M_cum(n-1)).
    
    For n = 1, M_cum(1) = M_elem(1) (since there's no previous element).
    """
    individual = []
    for n in range(len(matrices)):
        if n == 0:
            # First element is the same as the first cumulative
            individual.append(matrices[0])
        else:
            prev_cum = matrices[n - 1]
            current_cum = matrices[n]
            # M_elem(n) = M_cum(n) * inv(M_cum(n-1))
            prev_cum_inv = np.linalg.inv(prev_cum)
            M_elem = current_cum @ prev_cum_inv
            individual.append(M_elem)
    return individual


def write_individual_matrices(filename, individual_mats, element_info):
    """
    Writes the individual element matrices to a file in the same format as the input:
       ELE# <n> : <length> m
       +x.xxxxxxeyy ...
       ...
    for all 6 lines of the 6×6 matrix.
    """
    with open(filename, "w") as f_out:
        for i, M_elem in enumerate(individual_mats):
            elem_num = element_info[i]['element_number']
            length = element_info[i]['length']
            # Header line
            f_out.write(f"ELE# {elem_num} : {length} m\n")
            # Each row in scientific notation with explicit sign
            for row in M_elem:
                row_str = " ".join(f"{val:+.6e}" for val in row)
                f_out.write(row_str + "\n")
            # Optional: blank line between elements
            # f_out.write("\n")


def main():
    input_file = "Transfer_matrix1.dat"    # The cumulative-matrix file
    output_file = "Individual_matrix.dat"  # Output file for individual matrices

    # 1. Read the cumulative matrices from file
    cumulative_mats, element_info = read_cumulative_matrices(input_file)

    # 2. Factor out the individual element matrices
    individual_mats = factor_out_individual_matrices(cumulative_mats)

    # 3. Write the results in the same format
    write_individual_matrices(output_file, individual_mats, element_info)

    print(f"Done! Wrote individual-element matrices to '{output_file}'.")


if __name__ == "__main__":
    main()
