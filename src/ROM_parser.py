import numpy as np
from numba import njit
from numba.typed import List

@njit(inline='always', cache=True)
def _skip_ws(data, ptr, n):
    """Skips spaces (32), tabs (9), newlines (10, 13)."""
    while ptr < n and (data[ptr] == 32 or data[ptr] == 9 or
                       data[ptr] == 10 or data[ptr] == 13):
        ptr += 1
    return ptr


@njit(inline='always', cache=True)
def _parse_int(data, ptr, n):
    ptr = _skip_ws(data, ptr, n)
    if ptr >= n: return 0, ptr

    sign = 1
    val = 0
    if data[ptr] == 45:  # '-'
        sign = -1
        ptr += 1
    elif data[ptr] == 43:  # '+'
        ptr += 1

    while ptr < n:
        b = data[ptr]
        if b >= 48 and b <= 57:  # '0'-'9'
            val = val * 10 + (b - 48)
            ptr += 1
        else:
            break
    return val * sign, ptr


@njit(inline='always', cache=True)
def _parse_float(data, ptr, n):
    ptr = _skip_ws(data, ptr, n)
    if ptr >= n: return 0.0, ptr

    # 1. Sign
    sign = 1.0
    if data[ptr] == 45:  # '-'
        sign = -1.0
        ptr += 1
    elif data[ptr] == 43:  # '+'
        ptr += 1

    # 2. Integer Part
    val = 0.0
    while ptr < n:
        b = data[ptr]
        if b >= 48 and b <= 57:
            val = val * 10.0 + (b - 48)
            ptr += 1
        else:
            break

    if ptr < n and data[ptr] == 46:  # '.'
        ptr += 1
        frac_val = 0.0
        power_of_ten = 1.0
        while ptr < n:
            b = data[ptr]
            if b >= 48 and b <= 57:
                frac_val = frac_val * 10.0 + (b - 48)
                power_of_ten *= 10.0
                ptr += 1
            else:
                break
        val += frac_val / power_of_ten

    if ptr < n and (data[ptr] == 101 or data[ptr] == 69 or
                    data[ptr] == 100 or data[ptr] == 68):  # e, E, d, D
        ptr += 1
        esign = 1
        eval_ = 0
        if ptr < n and data[ptr] == 45:
            esign = -1
            ptr += 1
        elif ptr < n and data[ptr] == 43:
            ptr += 1

        while ptr < n:
            b = data[ptr]
            if b >= 48 and b <= 57:
                eval_ = eval_ * 10 + (b - 48)
                ptr += 1
            else:
                break
        if eval_ != 0:
            val = val * (10.0 ** (eval_ * esign))

    return val * sign, ptr

@njit(cache=True)
def _scan_pass(data, nwn):
    n = len(data)
    ptr = 0

    # -1 indicates coord not found yet
    idx_map = np.full((nwn, nwn), -1, dtype=np.int32)

    # Pre-allocate coordinate list with an estimate ~10% of the data
    est_lines = n // 10
    coords_temp = np.zeros((est_lines, 2), dtype=np.int32)

    unique_count = 0
    n_omega = 0
    omega_idx = -1

    while ptr < n:
        ptr = _skip_ws(data, ptr, n)
        if ptr >= n: break

        # Comment
        if data[ptr] == 35:  # '#'
            while ptr < n and data[ptr] != 10: ptr += 1
            continue

        # New omega block
        if data[ptr] == 38:  # '&'
            omega_idx += 1
            n_omega += 1
            # Skip header line: & omega idx re im
            while ptr < n and data[ptr] != 10: ptr += 1
            continue

        # Data
        # Parse i, j just to index them
        i, ptr = _parse_int(data, ptr, n)
        j, ptr = _parse_int(data, ptr, n)

        # Convert to 0-based
        i -= 1
        j -= 1

        # Validate bounds
        if 0 <= i < nwn and 0 <= j < nwn:
            # Check if seen before
            if idx_map[i, j] == -1:
                idx_map[i, j] = unique_count

                # Store + grow if needed
                if unique_count >= len(coords_temp):
                    new_size = len(coords_temp) * 2
                    new_arr = np.zeros((new_size, 2), dtype=np.int32)
                    new_arr[:len(coords_temp)] = coords_temp
                    coords_temp = new_arr

                coords_temp[unique_count, 0] = i
                coords_temp[unique_count, 1] = j
                unique_count += 1

        # Skip the rest
        while ptr < n and data[ptr] != 10: ptr += 1

    return n_omega, idx_map, unique_count, coords_temp[:unique_count]

@njit(cache=True)
def _fill_pass(data, nwn, idx_map, RESULT, F_data, omegas):
    n = len(data)
    ptr = 0
    omega_idx = -1

    while ptr < n:
        ptr = _skip_ws(data, ptr, n)
        if ptr >= n: break

        # Comment
        if data[ptr] == 35:
            while ptr < n and data[ptr] != 10: ptr += 1
            continue

        # Omega Header
        if data[ptr] == 38:  # '&'
            omega_idx += 1
            # Format: & omega [index] [re] [im]
            ptr += 1  # skip &

            # skip omega string
            ptr = _skip_ws(data, ptr, n)
            while ptr < n and data[ptr] > 32: ptr += 1

            # Skip the index token
            ptr = _skip_ws(data, ptr, n)
            while ptr < n and data[ptr] > 32: ptr += 1

            # Parse real
            re_o, ptr = _parse_float(data, ptr, n)

            # Parse imaginary
            im_o, ptr = _parse_float(data, ptr, n)

            if omega_idx < len(omegas):
                omegas[omega_idx] = re_o + 1j * im_o

            # Skip any trailing comments/garbage on this line
            while ptr < n and data[ptr] != 10: ptr += 1
            continue

        # Data Line: i j ...
        i, ptr = _parse_int(data, ptr, n)
        j, ptr = _parse_int(data, ptr, n)
        i -= 1
        j -= 1

        # Retrieve sparse index
        if 0 <= i < nwn and 0 <= j < nwn:
            q = idx_map[i, j]

            if omega_idx == -1:
                # F-block: i j re im
                re_v, ptr = _parse_float(data, ptr, n)
                im_v, ptr = _parse_float(data, ptr, n)
                if q != -1:
                    F_data[q] = re_v + 1j * im_v
            else:
                # Omega-block: i j x_re x_im y_re y_im
                x_re, ptr = _parse_float(data, ptr, n)
                x_im, ptr = _parse_float(data, ptr, n)
                y_re, ptr = _parse_float(data, ptr, n)
                y_im, ptr = _parse_float(data, ptr, n)

                if q != -1:
                    RESULT[omega_idx, 0, q] = x_re + 1j * x_im
                    RESULT[omega_idx, 1, q] = y_re + 1j * y_im
        else:
            # Out of bounds, just skip line
            while ptr < n and data[ptr] != 10: ptr += 1

        # Ensure we move to the next line
        while ptr < n and data[ptr] != 10: ptr += 1


def parse_XY_numba(filename, sparse_bool=True, return_idx=False):
    if not sparse_bool:
        raise NotImplementedError("Not implemented")

    with open(filename, 'rb') as f:
        raw_data = f.read()

    # Numba needs a numpy array wrapper around bytes
    data_arr = np.frombuffer(raw_data, dtype=np.uint8)

    first_nl = raw_data.find(b'\n')
    second_nl = raw_data.find(b'\n', first_nl + 1)
    header_line = raw_data[first_nl + 1: second_nl].decode('utf-8')
    parts = header_line.replace(",", "").split()
    nwn = int(parts[3]) + int(parts[6])

    # This assumes nwn*nwn fits in RAM
    n_omega, idx_map, num_coords, coords_arr = _scan_pass(data_arr, nwn)

    # Allocate
    RESULT = np.zeros((n_omega, 2, num_coords), dtype=complex)
    omegas = np.zeros(n_omega, dtype=complex)
    F_data = np.zeros(num_coords, dtype=complex)

    _fill_pass(data_arr, nwn, idx_map, RESULT, F_data, omegas)

    if return_idx:
        K = idx_map.max() + 1
        idx_list = [None] * K  # preallocate list
        for k in range(K):
            # vectorized: get all positions where idx_map == k
            rows, cols = np.where(idx_map == k)
            idx_list[k] = (rows[0], cols[0])
        return RESULT, omegas, F_data, idx_list

    return RESULT, omegas, F_data