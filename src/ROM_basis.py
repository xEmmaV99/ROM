"""
For a given meanfield solution, start to build a ROM basis for the response function.
"""
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

from src.Utils import combine_FAM_output


class ROM_basis:
    def __init__(self):
        self.omegas = None
        self.snapshots = None # todo condsider merging omegas and  snapshots together
        self.F = None

    def is_loaded(self):
        return self.omegas is not None and self.snapshots is not None

    def load(self, path_to_snapshot):
        from src.Utils import parse_XY_numba
        # load snapshots and omegas from file
        self.snapshots, self.omegas, self.F = parse_XY_numba(path_to_snapshot)

class data:
    def __init__(self, file):
        parse = self._parse_mf(file)
        self.num_proton = parse["num_proton"]
        self.num_neutron = parse["num_neutron"]
        self.num_proton_wf = parse["num_proton_wf"]
        self.num_neutron_wf = parse["num_neutron_wf"]
        self.param = parse["param"]


        self.param_prefix = {"BSkG2": "BXL",
                             "BSkG5": "BXL-N2LO"}  # dict for parameterization prefixes @ tantalus specific

    def set_FAM_parameters(self, param_dict):
        # FAM data
        for key in ['w_min', 'w_max', 'num_snapshots', 'smear', 'l', 'm']:
            if key not in param_dict:
                raise ValueError(f"Missing FAM parameter: {key}")
            setattr(self, key, param_dict[key])

    def _parse_mf(self, file):
        # change linux path to windows path
        text = Path(file.replace("/mnt/c/", "C:/")).read_text()

        match = re.search(r"General Information\s*-+\s*(.*)", text, re.DOTALL)
        if match:
            text = match.group(1) # check below General Information

        def grab(pattern, cast):
            m = re.search(pattern, text)
            if not m:
                raise ValueError(f"Could not find pattern: {pattern}")
            return cast(m.group(1))

        return {
            "num_neutron": grab(r"N =\s*([0-9]+)", int),
            "num_proton": grab(r"Z =\s*([0-9]+)", int),
            "num_neutron_wf": grab(r"nwn\s*=\s*(\d+)", int),
            "num_proton_wf": grab(r"nwp\s*=\s*(\d+)", int),
            "param": grab(r"Parameterization name\s+(\S+)", str),
        }

    @property
    def prefix(self):
        return self.param_prefix[self.param]

class ROM_builder:
    def __init__(self,
                 path_to_meanfield_wf,
                 path_to_meanfield_out,
                 FAM):

        self.data = data(path_to_meanfield_out)
        self.data.set_FAM_parameters(FAM)

        self.build_type = 'equidistant_1D'
        self.path_to_meanfield_wf = path_to_meanfield_wf

        BASE_DIR = Path(__file__).resolve().parent
        self.working_directory = BASE_DIR.joinpath('work_dir')
        print(f"Working directory set to: {self.working_directory}")

        # todo maybe check if wf file is actually valid
        if self._check_tantalus_installation(): print("Tantalus installation found.")
        else: raise Exception("Tantalus installation not found. No FAM runs can be launched.")

        self.path_to_snapshot = None # to be set

    def _clear_working_directory(self):
        """
        Clear the working directory
        """
        if self.working_directory.exists():
            for item in self.working_directory.iterdir():
                if item.is_dir():
                    for subitem in item.iterdir():
                        subitem.unlink()
                    item.rmdir()
                else:
                    item.unlink()
        else:
            self.working_directory.mkdir(parents=True, exist_ok=True)

    def _check_snapshot_file(self):
        return self.path_to_snapshot.exists()

    def _create_fam_runfiles(self, omegas_step, output_folder=""):
        """
        Create a FAM runfile for the given omegas
        """
        d = self.data
        run_folder = self.working_directory.joinpath('tmp')
        run_folder.mkdir(parents=True, exist_ok=True)
        for iteration, omega in enumerate(omegas_step):
            Rew = omega.real
            Imw = omega.imag
            file = run_folder.joinpath(f"fam_run{Rew}+{Imw}j.sh")
            file.parent.mkdir(parents=True, exist_ok=True)

            script = f"""\
#!/usr/bin/env bash
neutrons={d.num_neutron}
protons={d.num_proton}
dx=0.8
size=16
l={d.l}
m={d.m}
param="{d.param}"
pref="{d.prefix}"
logdir_name="{output_folder}"
OUT="V{iteration}"

# Recompute dependent variables if needed
nwn={d.num_neutron_wf}
nwp={d.num_proton_wf}
nbox=$(printf "%.0f" "$(echo "$size / $dx" | bc -l)")

# Directories HARDCODED (REFERS TO TANTALUS)
EXECDIR="{self.TANTALUS_PATH}exec"
PARAMDIR="{self.TANTALUS_PATH}parameterizations"

#1. environment variables
# clean up old log and data files. Logfile relative to workdir
wffile="{self.path_to_meanfield_wf}"
famoutfile="../$logdir_name/fam.$protons.$neutrons.$nwn.$nwp.$dx.out$OUT"
famfile="../$logdir_name/fam_data.$protons.$neutrons.$nwn.$nwp.$dx.fam$OUT"
xyfile="../$logdir_name/xy.$protons.$neutrons.$nwn.$nwp.$dx.xy$OUT"

exe="Tantalus.$pref.exe"              # full name of the mean-field executable
exefam="fam.$pref.exe"                # full name of the fam executable
param="$param"                         # name of the parameterization
workdir="src/work_dir/work.$protons.$neutrons.$size.$nwn.$nwp$OUT"

if [ ! -d "$workdir/" ]; then
  mkdir "$workdir"
fi

cp $EXECDIR/$exe             "$workdir"/
cp $EXECDIR/$exefam          "$workdir"/
cp $PARAMDIR/"$param.param"  "$workdir"/
cd "$workdir"

# Create runtime data
cat << EOF > fam.data
&nucleus
neutrons=$neutrons, protons=$protons
/
&mesh
nx=$nbox, ny=$nbox, nz=$nbox, dx=$dx
/
&func
name_param='${{param}}'
/
&pairing
/
&evolution
maxiter=1000
/
&scfiteration
/
&wfs
nwn = $nwn, nwp = $nwp
/
&IO
InputFilename='$wffile'
OutputFilename='trash'
allowtransform=.true.
famfile='${{famfile}}'
xyfile ='${{xyfile}}' 
/
&MomentParam
/
&Cranking
/
&fam
omega={Rew}
smear={Imw}
l=$l
m=$m
maxiter=10000
fam_precision=1e-6
/
EOF

./$exefam < fam.data > $famoutfile
fam_check=$?
        """
            file.write_text(script, newline="\n")  # , encoding="utf-8")#, newline="\n")
            file.chmod(0o755)

        return run_folder

    def _check_tantalus_installation(self):
        # check if tantalus is installed
        self.TANTALUS_PATH = "$HOME/code/tantalus/"
        return True #todo check properly

    def build_snapshot_basis_static(self, max_workers=4):
        d = self.data
        self._clear_working_directory()

        # static basis creation
        omegas = None
        if self.build_type == 'equidistant_1D':
            omegas = np.linspace(d.w_min, d.w_max, num=d.num_snapshots) + d.smear * 1.j

        if omegas is None:
            raise ValueError("No omegas defined for snapshot basis.")

        # start FAM calculation
        # output folder name
        OUTPUT_NAME = "TMP_SNAPSHOTS"

        fam_runfiles = self._create_fam_runfiles(omegas, output_folder=OUTPUT_NAME)
        RUN_FAM = input("Do you want to launch the FAM calculation now? (y/n): ") == 'y'

        # for fam_runfile in fam_runfiles.iterdir():
        #     ###  launch FAM with this runfile ###
        #     # Convert Windows path to WSL path
        #     wsl_path = f"/mnt/{fam_runfile.drive[0].lower()}{fam_runfile.as_posix()[2:]}"
        #     if RUN_FAM:
        #         print(f"Launching FAM with runfile: {fam_runfile}")
        #         subprocess.run(["bash", wsl_path], check=True)
        #     else:
        #         print(f"FAM runfile created at {fam_runfile}. You can launch it manually later.")

        base_dir = Path(f"src/work_dir/{OUTPUT_NAME}")
        base_dir.mkdir(parents=True, exist_ok=True)

        def launch_fam(fam_runfile: Path):
            # Convert Windows path to WSL path
            wsl_path = f"/mnt/{fam_runfile.drive[0].lower()}{fam_runfile.as_posix()[2:]}"
            if RUN_FAM:
                print(f"Launching FAM with runfile: {fam_runfile}")
                subprocess.run(["bash", wsl_path], check=True)
            else:
                print(f"FAM runfile created at {fam_runfile}. You can launch it manually later.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(launch_fam, fam_runfiles.iterdir())

        print("FAM launch completed.")

        # merge all output folders into one
        # from the folder OUTPUT_NAME read all files, merge them into one, and delete the old ones
        if RUN_FAM:
            self.path_to_snapshot = combine_FAM_output(directory=fr"{self.working_directory.joinpath(OUTPUT_NAME)}",
                               output_dir=self.working_directory.parent.parent.joinpath("_output"))

            self._clear_working_directory()
        pass

    def build_snapshot_basis_iterative(self):




        # make iterative basis. Each iteration, launch a new FAM calculation and automatically continue
        raise NotImplementedError

