#!/usr/bin/env python3

def load_gradients(gradient_file):
    """
    Reads lines like "QG1 = 12.2" (or "QG1  =  12.2") and
    returns a dictionary: {"QG1": 12.2, "QG2": -10.88, ...}.
    """
    gradients = {}
    with open(gradient_file, "r") as gf:
        for line in gf:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # Skip blanks or comments
            # Expect lines like: QG1 = 12.2
            parts = line.replace("=", " ").split()
            # parts might be ["QG1", "12.2"]
            if len(parts) >= 2:
                qg_name = parts[0]
                qg_value = float(parts[-1])
                gradients[qg_name] = qg_value
    return gradients

def generate_lattice(param_lattice_file, gradient_file, output_file):
    # Load the numeric values
    gradients_dict = load_gradients(gradient_file)

    with open(param_lattice_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            # For each QG# that appears in the line, replace it with the numeric value
            # We can do a simple approach: check each QG in the dictionary
            new_line = line
            for qg_name, qg_val in gradients_dict.items():
                # Convert e.g. "QG1" to "12.2" in the text
                if qg_name in new_line:
                    # We insert the numeric value as a string, e.g. "12.2"
                    new_line = new_line.replace(qg_name, f"{qg_val}")
            fout.write(new_line)

if __name__ == "__main__":
    import sys

    # Example usage:
    # python generate_lattice.py parameterized_lattice.txt quad_gradients.txt final_lattice.out

    if len(sys.argv) < 4:
        print("Usage: python generate_lattice.py <param_lattice> <gradients.txt> <output_lattice>")
        sys.exit(1)

    param_lattice_file = sys.argv[1]
    gradient_file = sys.argv[2]
    output_file = sys.argv[3]

    generate_lattice(param_lattice_file, gradient_file, output_file)
    print(f"Done! Wrote final lattice to {output_file}")
