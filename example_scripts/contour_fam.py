from src.Utils_parsers import parse_XY_numba, combine_FAM_output
from src.ROM_basis import ROM_builder
import numpy as np

wfs = [100, 150, 250]
sizes = [6, 8]#[8]#6]#, 8]

for wf in wfs:
    for size in sizes:
        print(f"wf: {wf}, size: {size}")

        WAVE_FUNCTION = f'/mnt/c/users/emmav/PycharmProjects/ROM/_input/wf.12.12.{wf}.{wf}.BSkG2.{size}'
        MF_OUT = f'/mnt/c/users/emmav/PycharmProjects/ROM/_input/mf.12.12.{wf}.{wf}.BSkG2.{size}.out'


        FAM_INPUT = {'w_min': 0.0,
                     'w_max': 40.0,
                     'num_snapshots' : 80,
                     'smear': 1.0,
                     'l': 0 , # multipolarity
                     'm': 0,  # magnetic quantum number
                     }

        rom_builder = ROM_builder(path_to_meanfield_wf=WAVE_FUNCTION,
                                  path_to_meanfield_out=MF_OUT,
                                  FAM=FAM_INPUT)

        # combine_FAM_output(directory=fr"{rom_builder.working_directory.joinpath('TMP_SNAPSHOTS')}",
        #                    output_dir=rom_builder.working_directory.parent.parent.joinpath(
        #                        "_output"), output_name=str(f'../_output/xy.12.12.{wf}.{wf}.0.0.BSkG2.{size:.0f}.xy').split('xy')[-2][1:-1])
        # exit(0)
        rom_builder.set_build_type('contour')
        rom_builder.set_contour_parameters(num_points=100, radius=40.0, center=0.0)

        #rom_builder.set_build_type('equidistant')

        RUN = False
        if RUN:
            try:
                path_to_snapshot = f'../_output/xy.12.12.{wf}.{wf}.0.0.BSkG2.{size:.0f}.xy'
                rom_builder.load(path_to_snapshot)
            except FileNotFoundError:
                print("snapshot not found, building from scratch")
            rom_builder.build_snapshot_basis(max_workers=6)
            path_to_snapshot = str(rom_builder.path_to_snapshot)
        else:
            # path_to_snapshot = '../_output/backup_sumrule/xy.12.12.30.30.0.0.xy'
            # path_to_snapshot = '../_output/xy.12.12.30.30.0.0.xy'
            path_to_snapshot = f'../_output/xy.12.12.{wf}.{wf}.0.0.BSkG2.{size:.0f}.xy'
            rom_builder.load(path_to_snapshot)


            def integrate_complex_arc(z, F_z, moment):
                from scipy.integrate import simpson
                phi = np.arctan2(np.imag(z), np.real(z))

                integrand = F_z * 1j * z

                if moment != 0:
                    integrand *= z ** moment
                # print("integration: ")
                # [print(f, i, x) for f, i, x in zip(z, integrand, phi)]
                integral =  simpson(integrand, x=phi)

                return integral

            def integrate_along_A1(omega, gamma, S, moment):
                sort_idx = np.argsort(gamma)
                omega_complex = omega[sort_idx] + 1j * gamma[sort_idx]
                # plt.scatter(np.real(omega_complex), np.imag(omega_complex), c='red')
                # plt.show()
                return integrate_complex_arc(omega_complex, S[sort_idx], moment=moment)

            wA1 = rom_builder.basis.omegas
            sA1 = rom_builder.basis.strength
            # print(sA1)
            from matplotlib import pyplot as plt
            plt.scatter(np.real(wA1), np.imag(wA1))

            if 'backup_sumrule' in path_to_snapshot:
                m1 = integrate_along_A1(np.real(wA1), np.imag(wA1), sA1, moment=1) / (2 * np.pi * 1j)
                # 4797.933373106706 ## 20 points or so, no symmetry, 30 30 wf
            else:
                m1 = 2*np.real(integrate_along_A1(np.real(wA1), np.imag(wA1), sA1, moment=1) / (2 * np.pi * 1j))
                # 4797.9347540970875 ## 100 points, symmetry, 30 30 wf

                # 21 points
                # 9222.151018518663 100,100 wavefunction, symmetry


            print(m1)
            m1_file = (3.0934/3.0087)**(-2) * 1.90478056E+04

            hbm = 20.73553 # in bskg5, hbar^2/(2m)
            e_eff = 1.0
            r2_ch = 3.0934**2
            r2_ma = 3.0087**2
            m1_formula = 4.0 * e_eff**2 * hbm * (24) * r2_ma
            #print("from file and calculated: ", m1_file, m1_formula)
            print("Sum rule exhausted: ", m1/m1_formula * 100 , "%")


plt.show()