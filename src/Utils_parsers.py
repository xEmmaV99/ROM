from pathlib import Path
import numpy as np
from numba import njit

def _filepath():
    return Path(__file__)

@njit(inline='always', cache=True)
def _skip_ws(data, ptr, n):
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

    # 3. Fraction Part (The Fix)
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

    # 4. Exponent Part
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


def parse_XY_numba(filename, return_idx=False):
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


def merge_FAM_outputs(folder, master_file=None, output_dir=None, cleanup=True):
    if master_file is None:
        out_name=''
    else:
        out_name = str(master_file).split('xy')[-2][1:-1]

    if output_dir is None:
        output_dir = _filepath().parent.parent.joinpath("_outputs")

    path_to_snapshot = combine_FAM_output(
            directory=folder,
            output_dir=output_dir,
            output_name=out_name
        )
    print("Merge completed. Snapshot file created at: ", path_to_snapshot)

    if cleanup:
        # remove files and folders in the directory
        for f in Path(folder).glob('*'):
            f.unlink()


def combine_FAM_output(directory, output_name='', master_file=None, output_dir=None, verbose=False):
    """
    :param directory: str, directory of the FAM output TO BE MERGED
    :param output_name: str, name of the output (fam.{}.fam and xy.{}.xy)
    :param master_file: if previous data exists, provide the path to the .xy file!
    :param output_dir:  directory to put the output, Path() object!
    :param verbose:  if True, print info
    :return:
    """
    import glob
    import numpy as np
    import pandas as pd
    print('Combining FAM output')
    def concat_fam_files(directory):
        # get a list of all the .fam files in the directory
        if verbose: print("glob", glob.glob(directory + '/*.famV*')[0])

        if master_file is not None:
            file_path =str(master_file.parent.joinpath("fam_data."+str(master_file).split('xy')[-2][1:-1]+".fam"))
        else:
            file_path = glob.glob(directory + '/*.famV*')[0]

        name = glob.glob(directory + '/*.famV*')[0].split('\\')[-1].split('/')[-1]
        if verbose: print('Processing:', name)

        fam_files=[]
        for f in list(Path(directory).glob(f"{name[:-1]}*")):
            fam_files.append(f)
        if len(fam_files) > 0:
            if file_path not in fam_files:
                fam_files.append(file_path) #master
            if verbose: print(f'Merging', fam_files)

            # open the last file to get the header (trailing '#' signs)
            with open(fam_files[-1], 'r') as f:
                lines = f.readlines()
            header = [line for line in lines if line.strip().startswith('#')]
            # join lines into string and remove last newline sign
            header = ''.join(header)[:-1]

            # check if the other headers are the same (as a simple consistency check)
            for fam_file in fam_files[:-1]:
                with open(fam_file, 'r') as f:
                    lines = f.readlines()

                header_i_lines = [line for line in lines if line.lstrip().startswith('#')]
                header_i = ''.join(header_i_lines)[:-1]

                if header_i != header:
                    raise Exception(f'Header in file {fam_file} differs from that in {fam_files[0]}')

            # read the data in all fam files using pandas
            df_list = [df for fam_file in fam_files if (df := read_fam_data(fam_file)) is not None]
            # Filter out empty DataFrames
            df_list = [df for df in df_list if not df.empty]

            # concatenate all the dataframes, ignoring the arbitrary index
            df_combined = pd.concat(df_list, ignore_index=True)

            # sort on omega
            df_cleaned = df_combined.drop_duplicates().sort_values(by=df_combined.columns[0])

            # write combined output to famfileout
            if verbose: print("name", name)

            if output_name == '' and master_file is None:
                famfileout = output_dir.joinpath(name.split('\\')[-1].split('.famV')[0] + '.fam')
            elif output_name != '':
                famfileout = output_dir.joinpath('fam_data.'+output_name + '.fam')
            else:
                # we need to convert master_file (xy) to fam_data
                dir = master_file.parent
                _name = master_file.name.split("xy")[-2][1:-1]
                famfileout = dir.joinpath('fam_data.'+_name+'.fam')

            write_fam_data(df_cleaned, header, famfileout)
        return famfileout

    def concat_xy_files(directory):
        """
        concatenate .xy files from different runs together.
        """
        if master_file is not None:
            file_path = str(master_file)
        else:
            file_path = glob.glob(directory + '/*.xyV*')[0] # if multiple famV0

        name = glob.glob(directory + '/*.xyV*')[0].split('\\')[-1].split('/')[-1]
        if verbose: print('Processing:', name)
        xy_files=[]

        all_header = None
        all_blocks = []
        for f in list(Path(directory).glob(name[:-1] + '*')):
            xy_files.append(f)

        if len(xy_files) > 0:
            if file_path not in xy_files:
                xy_files.append(file_path)
            if verbose: print(f'Merging', xy_files)

        for i, file_path in enumerate(xy_files):
            header, omega_blocks = XY_parse_file(file_path)
            # Keep header only from first file
            if all_header is None:
                all_header = header
            # Check if other headers are the same
            else:
                if header != all_header:
                    raise Exception(f'Header in file {file_path} differs from that in {xy_files[0]}')
            all_blocks.extend(omega_blocks)

        all_blocks.sort(key=lambda x: x[0])
        # don't allow for duplicate x[0]'s so the omegas
        tmp = []
        for block in all_blocks:
            if block[0] in [tmp[k][0] for k in range(len(tmp))]:
                pass
            else:
                tmp.append(block)
        all_blocks = tmp

        if output_name == '' and master_file is None:
            out = output_dir.joinpath(name.split('\\')[-1].split('/')[-1].split('.xy')[0]+'.xy')
        elif output_name != '':
            out = output_dir.joinpath('xy.' + output_name + '.xy')
        else:
            out = Path(master_file)

        with open(out, "w") as out:
            out.writelines(all_header)
            for omega, block_lines in all_blocks:
                out.writelines(block_lines)

            if verbose: print(f"Written combined file: {out}")

        return out.name

    def read_fam_data(filename):
        df = pd.read_csv(filename, sep=r'\s+', comment='#',
                         names=["omega", "gamma", "iter", "S_complex_re", "S_complex_im", "S_n+", "S_n-", "S_p+", "S_p-", "S_tot"])
        return df

    def XY_parse_file(path):
        import re
        #omega_header_re = re.compile(r"&\s*omega\s*=\s*([0-9.\-Ee]+)")
        omega_header_re = re.compile(r"&\s*omega\s*=\s*([0-9.\-Ee]+)\s+([0-9.\-Ee]+)")

        header = []
        omega_blocks = []

        with open(path, "r") as f:
            lines = f.readlines()

        current_block = []
        current_omega = None
        in_omega_section = False

        for line in lines:
            m = omega_header_re.search(line)
            if m:  # Found a new & omega block
                if current_omega is not None:
                    omega_blocks.append((current_omega, current_block))

                # Extract both values
                omega_val = float(m.group(1))
                smearing_val = float(m.group(2))
                current_omega = (omega_val, smearing_val)  # store new omega as a tuple
                current_block = [line]  # start new block
                in_omega_section = True
                continue

            if not in_omega_section:
                header.append(line)
            else:
                current_block.append(line)

        # store last block
        if current_omega is not None:
            omega_blocks.append((current_omega, current_block))

        return header, omega_blocks

    def write_fam_data(df, header, filename):
        fmt = '%10.3f %10.3f %8i %25.12E %25.12E %25.12E %25.12E %25.12E %25.12E %25.12E'
        with open(filename, 'w') as f:
            np.savetxt(f, df.values, fmt=fmt, header=header, comments='')

    fam_file = concat_fam_files(directory)
    out_file = concat_xy_files(directory)

    if verbose: print(f'Combined output written to "{fam_file}" and "{out_file}" ')

    return Path(out_file)