#!/usr/bin/env python3
import numpy as np

def parse_lattice_file(lattice_filename):
    """
    Reads 'mebt_final.dat' line by line.
    Recognized elements:
      - DRIFT <length_mm>
      - QUAD  <length_mm> <gradient_T/m>
      - APERTURE dx dy n
      - FIELD_MAP ...
      - THIN_STEERING ...
    Skips:
      - lines starting with ';' (comments)
      - instrumentation markers ending with ':' (e.g. "BPM :")
      - unrecognized lines (like "FREQ 162.5")

    Returns a list of dict, each describing one beam element:
      {
        "index": <int>,         # global index of recognized element
        "etype": "DRIFT"/"QUAD"/"APERTURE"/"FIELD_MAP"/"THIN_STEERING",
        "length": <float in meters>,
        "gradient": <float or None>,
        "dx": None or float,    # for APERTURE
        "dy": None or float,
        "atype": None or int,
        "name": "QG<N>" for quads (N is the quad_counter),
        "parameters": <list>    # extra tokens if needed
      }
    """
    elements = []
    element_index = 1     # increments for every recognized element
    quad_counter  = 1     # increments only for QUAD

    with open(lattice_filename, "r") as f:
        for line in f:
            line = line.strip()
            # 1) Skip blank or comment lines
            if (not line) or line.startswith(";"):
                continue

            # 2) Skip instrumentation markers: line ends with ':' and <= 2 tokens
            if line.endswith(":") and len(line.split()) <= 2:
                continue

            parts = line.split()
            cmd = parts[0].upper()

            if cmd == "DRIFT":
                # e.g. DRIFT 50 ...
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
                # e.g. QUAD 100 12.2 ...
                ds_mm_str = parts[1].replace(";", "")
                ds_mm = float(ds_mm_str)
                ds_m  = ds_mm * 1e-3
                grad_str = parts[2].replace(";", "")
                try:
                    G_val = float(grad_str)
                except ValueError:
                    G_val = 0.0

                # Use separate quad_counter for naming quads:
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
                quad_counter += 1  # only increment for a QUAD

            elif cmd == "APERTURE":
                # e.g. APERTURE 15 15 1
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
                # e.g. "FREQ 162.5" => skip
                pass

    return elements

def read_individual_matrices(filename):
    """
    Reads 'Individual_matrix.dat' lines like:
      ELE# i : 0.25 m
      (6 lines of matrix data)
      ELE# i+1 : ...
    Returns: [{"index": i, "pos": pos_m, "matrix": M(6x6)}, ...].
    """
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
            # next 6 lines => 6x6 matrix
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
    """
    Reads 'quad_gradients.txt' lines like:
      QG1  =  0
      QG2  =  0
      QG3  =  7.75
      QG4  =  -6.05
      ...
    Returns dict, e.g. {"QG1":0.0, "QG2":0.0, "QG3":7.75, ...}
    """
    grad_dict = {}
    with open(gradient_file, "r") as gf:
        for line in gf:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # e.g. "QG3 = 7.75"
            line = line.replace('=', ' ')
            parts = line.split()
            if len(parts) >= 2:
                qname = parts[0]
                gval  = float(parts[1])
                grad_dict[qname] = gval
    return grad_dict

def compute_quad_matrix(ds_m, G, brho, gamma):
    """
    6x6 thick-lens QUAD matrix with your "swapped" convention:

    If G>0 => 
       x-plane = defocusing => (cosh, sinh),
       y-plane = focusing   => (cos, sin).

    If G<0 => 
       x-plane = focusing   => (cos, sin),
       y-plane = defocusing => (cosh, sinh).

    Also includes (z, delta) block: 
       [1, ds_m/gamma^2]
       [0, 1           ]

    ds_m : quad length in meters
    G    : gradient in T/m
    brho : beam rigidity in T·m
    gamma: Lorentz factor
    """
    M = np.eye(6)

    # (z,delta) sub-block
    M[4,4] = 1.0
    M[4,5] = ds_m / (gamma**2)
    M[5,5] = 1.0

    tiny = 1e-14
    if abs(G) < tiny:
        # treat as drift
        M[0,1] = ds_m
        M[2,3] = ds_m
        return M

    if abs(brho) < 1e-12:
        raise ValueError("Beam rigidity brho is too small or zero!")

    k  = np.sqrt(abs(G) / brho)
    kl = k * ds_m

    # Precompute cos, sin, cosh, sinh
    cos_kl  = np.cos(kl)
    sin_kl  = np.sin(kl)
    cosh_kl = np.cosh(kl)
    sinh_kl = np.sinh(kl)

    if G > 0:
        # G>0 => x-plane defocus => cosh/sinh, y-plane focus => cos/sin
        # x-plane (defocus)
        M[0,0] =  cosh_kl
        M[0,1] =  sinh_kl / k
        M[1,0] =  k*sinh_kl
        M[1,1] =  cosh_kl

        # y-plane (focus)
        M[2,2] =  cos_kl
        M[2,3] =  sin_kl / k
        M[3,2] = -k*sin_kl
        M[3,3] =  cos_kl

    else:
        # G<0 => x-plane focus => cos/sin, y-plane defocus => cosh/sinh
        # x-plane (focus)
        M[0,0] =  cos_kl
        M[0,1] =  sin_kl / k
        M[1,0] = -k*sin_kl
        M[1,1] =  cos_kl

        # y-plane (defocus)
        M[2,2] =  cosh_kl
        M[2,3] =  sinh_kl / k
        M[3,2] =  k*sinh_kl
        M[3,3] =  cosh_kl

    return M

def update_matrices(elements, matrix_data, quad_grad_dict, brho, gamma):
    """
    For each element in 'elements', if it's QUAD => 
       find name e.g. QG3 in quad_grad_dict, 
       compute new 6x6 using compute_quad_matrix with the 'swapped' logic.
    Otherwise, keep the old matrix from matrix_data.
    """
    updated = []

    for elem, md in zip(elements, matrix_data):
        e_idx   = elem["index"]
        e_type  = elem["etype"]
        old_mat = md["matrix"]
        pos_m   = md["pos"]
        ds_m    = elem["length"]

        if e_type == "QUAD":
            qname = elem["name"]  # e.g. "QG1", "QG2", etc.
            if qname in quad_grad_dict:
                new_grad = quad_grad_dict[qname]
            else:
                new_grad = elem["gradient"]

            print(f"[DEBUG] QUAD index={e_idx}, name={qname}, G={new_grad:+.6f} T/m, L={ds_m:.6f} m")
            M_new = compute_quad_matrix(ds_m, new_grad, brho, gamma)
            updated.append({
                "index": e_idx,
                "pos":   pos_m,
                "matrix": M_new
            })
        else:
            # DRIFT/APERTURE/etc => keep old
            updated.append({
                "index": e_idx,
                "pos":   pos_m,
                "matrix": old_mat
            })

    return updated

def write_updated_matrices(output_file, updated_data):
    """
    Writes the updated matrix file in the format:
      ELE# i : pos m
      row1
      row2
      ...
      row6
    """
    with open(output_file, "w") as f:
        for item in updated_data:
            idx = item["index"]
            pos = item["pos"]
            M   = item["matrix"]
            f.write(f"ELE# {idx} : {pos:.6f} m\n")
            for row in M:
                row_str = " ".join(f"{val:+.6e}" for val in row)
                f.write(row_str + "\n")

    print(f"[INFO] Wrote updated matrix file: {output_file}")

def main():
    # Filenames
    lattice_file  = "mebt_final.dat"
    matrix_file   = "Individual_matrix.dat"
    grad_file     = "quad_gradients.txt"
    updated_file  = "Individual_matrix_updated.dat"

    # Example beam parameters for ~2 MeV H-:
    gamma = 1.0022599
    brho  = 0.21075678  # T·m

    # 1) Parse the final lattice (with separate counters for quads)
    elements = parse_lattice_file(lattice_file)

    # 2) Read original 6x6 matrix data
    matrix_data = read_individual_matrices(matrix_file)

    # 3) Load updated quad gradients
    quad_dict = load_quad_gradients(grad_file)
    print("[DEBUG] Loaded quad gradient dictionary:", quad_dict)

    # 4) Recompute quads with the swapped convention
    updated_data = update_matrices(elements, matrix_data, quad_dict, brho, gamma)

    # 5) Write the updated matrix file
    write_updated_matrices(updated_file, updated_data)

    print("[INFO] Done with swapped logic:")
    print("       G>0 => x-plane defocus => cosh/sinh, y-plane focus => cos/sin")
    print("       Quads named QG1, QG2, etc. ignoring drifts/apertures.")

if __name__ == "__main__":
    main()
