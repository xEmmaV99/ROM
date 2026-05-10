from pathlib import Path

import numpy as np
import pandas as pd
from src.Emulator import Emulator
from src.ROM_basis import ROM_basis
from src.ROM_basis import ROM_builder

# test greedy SAMPLER
# test equidistant SAMPLER
# test sumrule SAMPLER

# test greedy PREDICTION
# test equidistant PREDICTION
# test sumrule PREDICTION

# todo: how to test both ? or not required ?

test_input = Path(__file__).parent.resolve().joinpath('_inputs')
test_output = Path(__file__).parent.resolve().joinpath('_outputs')
test_reference = Path(__file__).parent.resolve().joinpath('_references')
path_tantalus = '/home/emma/code/tantalus/'#'$HOME/code/tantalus/'

class TestSamplers:
    """
    Can only be used when tantalus is available.
    """
    def test_equidistant(self):
        WAVE_FUNCTION = Path.joinpath(test_input, 'wf.12.12.30.30')#'/mnt/c/users/emmav/PycharmProjects/ROM/tests/_inputs/wf.12.12.30.30'
        MF_OUT = Path.joinpath(test_input, 'mf.12.12.30.30.out')#'/mnt/c/users/emmav/PycharmProjects/ROM/tests/_inputs/mf.12.12.30.30.out'
        FAM_INPUT = {'w_min': 0.0,
                     'w_max': 40.0,
                     'max_num_snapshots': 5,
                     'smear': 1.0,
                     'l': 0,  # multipolarity
                     'm': 0,  # magnetic quantum number
                     }

        rom_builder = ROM_builder(path_to_meanfield_wf=str(WAVE_FUNCTION),
                                  path_to_meanfield_out=str(MF_OUT),
                                  FAM=FAM_INPUT,
                                  tantalus_path=path_tantalus)

        rom_builder.build_type = 'equidistant_1D'
        rom_builder.output_dir = test_output

        rom_builder.build_snapshot_basis()

        compare_xy_files(rom_builder.path_to_snapshot, reference_path=test_reference.joinpath('emulator_equid.xy'))


    def test_greedy(self):
        WAVE_FUNCTION = Path.joinpath(test_input, 'wf.12.12.30.30') #'/mnt/c/users/emmav/PycharmProjects/ROM/tests/_inputs/wf.12.12.30.30'
        MF_OUT = Path.joinpath(test_input, 'mf.12.12.30.30.out')#'/mnt/c/users/emmav/PycharmProjects/ROM/tests/_inputs/mf.12.12.30.30.out'
        FAM_INPUT = {'w_min': 0.0,
                     'w_max': 40.0,
                     'max_num_snapshots': 5,
                     'smear': 1.0,
                     'l': 0,  # multipolarity
                     'm': 0,  # magnetic quantum number
                     }

        rom_builder = ROM_builder(path_to_meanfield_wf=str(WAVE_FUNCTION),
                                  path_to_meanfield_out=str(MF_OUT),
                                  FAM=FAM_INPUT,
                                  tantalus_path=path_tantalus)

        rom_builder.build_type = 'greedy'
        rom_builder.output_dir = test_output

        rom_builder.build_snapshot_basis()

        compare_xy_files(rom_builder.path_to_snapshot, reference_path=test_reference.joinpath('emulator_greedy.xy'))


    def test_contour(self):
        # TODO
        # code for contour sampler
        assert True

class TestPredictions:
    def test_strength(self):
        for emul_name in ['emulator_greedy', 'emulator_equid']:
            e = Emulator(basis=ROM_basis().load(path_to_snapshot=test_input.joinpath(f'{emul_name}.xy')))

            e.projection_method = "G"
            compare_emulated_strength(emulator=e, reference_emulator_output=test_reference.joinpath(f'{emul_name}_G'))

            e.projection_method = "PG"
            compare_emulated_strength(emulator=e, reference_emulator_output=test_reference.joinpath(f'{emul_name}_PG'))

    def test_sumrule(self):
        # TODO
        assert True
        # # code for sumrule prediction
        # compare_m1() # using G
        # compare_m1() # using PG






def compare_xy_files(output_path, reference_path, tol=1e-6):
    def load(path):
        with open(path) as f:
            rows = [
                [float(x) for x in line.split()]
                for line in f
                if line.strip() and not line.startswith(("#", "&"))
            ]
        return np.array(rows)

    output = load(output_path)
    reference = load(reference_path)

    if not np.allclose(output, reference, atol=tol):
        diff = np.abs(output - reference)
        idx = np.unravel_index(np.argmax(diff), diff.shape)

        raise AssertionError(
            f"Max difference {diff[idx]:.3e} at row {idx[0]}, col {idx[1]}: "
            f"{output[idx]} != {reference[idx]}"
        )


def compare_emulated_strength(emulator : Emulator, reference_emulator_output):
    df = pd.read_csv(
        reference_emulator_output,
        converters={"omega": complex}  # This forces the string to be parsed as a complex number
    )
    d = df.to_dict(orient="list")

    targets, S = emulator.evaluate(targets= d['omega'])
    assert np.all(np.isclose(0, S - d['S_tot'], rtol=1e-10))


def compare_m1(emulator : Emulator, reference_emulator_output):
    assert True
