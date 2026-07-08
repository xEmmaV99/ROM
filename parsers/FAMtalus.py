import os
import numpy as np
import numba as nb

@nb.njit
def _fill_component_numba(target_matrix, freq_idx, component_idx, offset, raw_data, mapping_table):
    """
    target_matrix: (n_omega, 2, 2*n_rows)
    raw_data: the [mu, nu, real, imag] array from a file
    mapping_table: 2D array mapping [mu, nu] to row index
    """
    for i in range(raw_data.shape[0]):
        mu = int(raw_data[i, 0])
        nu = int(raw_data[i, 1])
        real_val = raw_data[i, 2]
        imag_val = raw_data[i, 3]

        idx = mapping_table[mu, nu]
        if idx != -1:
            target_matrix[freq_idx, component_idx, idx + offset] = real_val + 1j * imag_val

@nb.njit
def _fill_F_numba(F,raw_data, mapping_table):
    for i in range(raw_data.shape[0]):
        mu = int(raw_data[i, 0])
        nu = int(raw_data[i, 1])
        real_val = raw_data[i, 2]
        imag_val = raw_data[i, 3]

        idx = mapping_table[mu, nu]
        if idx != -1:
            F[idx] = real_val + 1j * imag_val

def read_file(filepath):
    """Fastest way to read these specific space-separated files into NumPy"""
    # Using usecols to skip the '+' and 'i' characters
    # Expects: mu(0) nu(1) real(2) + (3) imag(4) i(5)
    try:
        data = np.loadtxt(filepath, usecols=(0, 1, 2, 4), skiprows=19)
        return data

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return np.empty((0, 4))

def identify_frequencies(folder):
    import re
    all_files = os.listdir(folder)
    freq_strings = sorted(list(set([
        re.search(r"om_i(.*)", f).group(1)
        for f in all_files if f.startswith("om_i")
    ])))
    return np.array([float(f) for f in freq_strings])

def parse_from_folder(folder, freq_strings=None):
    if freq_strings is None:
        freq_strings = identify_frequencies(folder)

    n_omegas = len(freq_strings)
    print(f"Found {n_omegas} frequencies: {freq_strings}")

    # Load static F files to create the mapping and F_vector
    f20p_data = read_file(os.path.join(folder, "F20_prot.out"))
    f20n_data = read_file(os.path.join(folder, "F20_neut.out"))
    f02p_data = read_file(os.path.join(folder, "F02_prot.out"))
    f02n_data = read_file(os.path.join(folder, "F02_neut.out"))

    # Setup mapping table
    flarge = f20p_data if f20p_data.shape[0] >= f20n_data.shape[0] else f20n_data
    n_rows = flarge.shape[0]  # Use the maximum number of rows from F20p and F20n
    print("nrows:", n_rows)
    # select which one is largest
    max_mu = int(np.max(flarge[:, 0]))
    max_nu = int(np.max(flarge[:, 1]))
    mapping_table = np.full((max_mu + 1, max_nu + 1), -1, dtype=np.int32)
    for i in range(n_rows):
        mapping_table[int(flarge[i, 0]), int(flarge[i, 1])] = i

    f20p = np.zeros(n_rows, dtype=np.complex128)
    f20n = np.zeros(n_rows, dtype=np.complex128)
    f02p = np.zeros(n_rows, dtype=np.complex128)
    f02n = np.zeros(n_rows, dtype=np.complex128)

    #fill mapping tbale
    _fill_F_numba(f20p, f20p_data, mapping_table)
    _fill_F_numba(f20n, f20n_data, mapping_table)
    _fill_F_numba(f02n, f02n_data, mapping_table)
    _fill_F_numba(f02p, f02p_data, mapping_table)

    # print(f"shapes\n F20p: {f20p.shape}", f"F20n: {f20n.shape}", f"F02p: {f02p.shape}", f"F02n: {f02n.shape}")
    # Create F_data vector
    F_data = np.concatenate([ f20p,  f20n, f02p, f02n ])

    matrices = np.zeros((n_omegas, 2, 2 * n_rows), dtype=np.complex128)

    # 5. Loop through frequencies and fill components using Numba worker
    omegas = []
    for f_idx, f_str in enumerate(freq_strings):
        # Convert frequency string to complex/float number if possible
        omegas.append(float(f_str) + 1.0j) #imaginary part is 1.0

        comps = [
            ("X_prot", 0, 0),  # (prefix, target_dim1, offset_multiplier)
            ("X_neut", 0, n_rows),
            ("Y_prot", 1, 0),
            ("Y_neut", 1, n_rows)
        ]

        for prefix, dim1, offset in comps:
            fname = f"om_i{np.real(f_str):.2f}/{prefix}.out"
            data = read_file(os.path.join(folder, fname))
            if data.size > 0:
                _fill_component_numba(matrices, f_idx, dim1, offset, data, mapping_table)

    return np.array(omegas), matrices, F_data
