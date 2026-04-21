"""
For a given meanfield solution, start to build a ROM basis for the response function.
"""
import re
try:
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
except ImportError:
    print("subprocess and concurrent.futures modules are required for launching FAM runs. Please make sure to have them available.")
from pathlib import Path
import numpy as np
try:
    from functools import cached_property
    from src.Utils_basis import cost_numba, _evaluate_PG_numba_coef
    from src.Utils_parsers import parse_XY_numba, combine_FAM_output
except ImportError:
    print("Numba not installed.")
import platform

class ROM_basis:
    def __init__(self):
        self.omegas = None
        self.snapshots = None # todo consider merging omegas and  snapshots together
        self.F = None


    def is_loaded(self):
        return self.omegas is not None and self.snapshots is not None


    def load(self, path_to_snapshot):
        from src.Utils_parsers import parse_XY_numba
        # load snapshots and omegas from file
        self.snapshots, self.omegas, self.F = parse_XY_numba(path_to_snapshot)
        return self

    @cached_property
    def strength(self):
        S_FAM = np.array([
            np.sum(np.conj(self.F) * self.snapshots[:, 0, :][j] + np.conj(self.F) * self.snapshots[:, 1, :][j])
            for j in range(len(self.omegas))
        ], dtype=np.complex128)
        # return -2 * np.imag(S_FAM) / np.pi
        return 2*S_FAM # 2 is required to match the strength from direct FAM calculation


class data:
    def __init__(self, file, fam_params=None):
        parse = self._parse_mf(file)
        self.num_proton = parse["num_proton"]
        self.num_neutron = parse["num_neutron"]
        self.num_proton_wf = parse["num_proton_wf"]
        self.num_neutron_wf = parse["num_neutron_wf"]
        self.param = parse["param"]
        self.dx = parse["dx"]
        self.size = parse["size"]

        self.param_prefix = {"BSkG2": "BXL",
                             "BSkG4": "BXL",
                             "BSkG5": "BXL-N2LO"}  # dict for parameterization prefixes @ tantalus specific

        self._fam_defaults = {
            'w_min': 0.0,
            'w_max': 20.0,
            'max_num_snapshots': 20, # consider moving this to the builder
            'smear': 1.0,
            'l': 0,
            'm': 0
        }
        self.set_FAM_parameters(fam_params)

    def set_FAM_parameters(self, fam_params):
        # FAM data
        if fam_params is None:
            fam_params = {}
        params = self._fam_defaults.copy()
        params.update(fam_params)
        # Set the attributes
        for key, value in params.items():
            setattr(self, key, value)

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
            "dx": grab(r'dx\s*=\s*([0-9.eE+-]+)', float),
            "size": grab(r'nx\s*=\s*([0-9]+)', float) * grab(r'dx\s*=\s*([0-9.eE+-]+)', float),
        }

    @property
    def prefix(self):
        return self.param_prefix[self.param]

def convert_to_linux(path: str) -> str:
    if platform.system() == "Windows":
        path = path.replace("\\", "/")
        if path[1] == ":":
            drive = path[0].lower()
            rest = path[2:]
            rest = rest.lstrip("/")
            return f"/mnt/{drive}/{rest}"
        else:
            return path
    else:
        return path

class ROM_builder:
    def __init__(self,
                 path_to_meanfield_wf: str,
                 path_to_meanfield_out: str,
                 FAM,
                 tantalus_path="$HOME/code/tantalus/"):
        self._DEBUG_clear_wd = True # if False, the output files are not merged and the working dir is not cleared. Only used for debugging purposes
        self._build_types = ['equidistant_1D', 'greedy', 'greedy_2D','contour']
        self._run_types = ['generate_runfiles', 'default']

        self.data = data(path_to_meanfield_out)
        self.data.set_FAM_parameters(FAM)

        self.build_type = 'equidistant_1D'
        self.path_to_meanfield_wf = convert_to_linux(path_to_meanfield_wf)

        self.basis = ROM_basis() # to be loaded!!
        self.basis.load = self.load # overwrite function to have the same function but ALSO save path

        self.working_directory = self.get_base_dir().joinpath('work_dir')
        # todo consider making a function of working dir so i can change it to linux or back..

        ### check if workdir exists

        # check if Tantalus is installed, if not raise error
        self.set_TANTALUS_path(tantalus_path)

        print(f"Checking Tantalus installation at: {self.TANTALUS_PATH}")
        if not self._check_tantalus_installation():
             raise Exception("Tantalus installation not found. No FAM runs can be launched.")

        # todo check if wf input file is valid

        self.path_to_snapshot = None # to be set
        self.run_type = 'default' # run all by default
        self.tmp_output = "TMP_SNAPSHOTS"
        self.output_dir = self.working_directory.parent.parent.joinpath("_outputs")

        self.greedy_settings = {"cost_threshold": 0.0}
        self.greedy_2D_settings = {"cost_threshold": 0.0, #threshold of the greedy snapshot cost cutoff
                                   "cost_relaxation": 0.1} #selecting snapshots below 0.1 relative cost


    def get_base_dir(self):
        return Path(__file__).resolve().parent

    def set_TANTALUS_path(self, path):
        self.TANTALUS_PATH = path


    def set_build_type(self, build_type):
        if build_type not in self._build_types:
            raise ValueError("Unknown build type. Supported types: " + ", ".join(self._build_types))
        self.build_type = build_type


    def set_run_type(self, run_type):
        if run_type not in self._run_types:
            raise ValueError("Unknown build type. Supported types: " + ", ".join(self._run_types))
        if run_type == 'generate_runfiles' and self.build_type == 'greedy':
            raise ValueError("Cannot only generate runfiles for greedy build type, as it requires iterative runs.")
        self.run_type = run_type


    def set_contour_parameters(self, n=13, R=500, r=0.1):
        self.contour_n = n
        self.contour_R = R
        self.contour_r = r


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
        # make sure the working directory exists
        self.working_directory.mkdir(parents=True, exist_ok=True)


    def _create_fam_runfiles(self, omegas, output_folder="", global_iteration=0):
        """
        Create a FAM runfile for the given omegas
        """
        d = self.data
        # always create a new folder, add number if already exists
        # check if file exists
        run_folder = self.working_directory.joinpath('runfiles')

        run_folder.mkdir(parents=True, exist_ok=True)
        for iteration, omega in enumerate(omegas):
            Rew = omega.real
            Imw = omega.imag
            file = run_folder.joinpath(f"fam_run.{d.num_neutron}.{d.num_proton}.{d.num_neutron_wf}.{d.num_proton_wf}.{d.param}.{d.size}.{Rew}+{Imw}j.sh")
            file.parent.mkdir(parents=True, exist_ok=True)

            script = f"""\
#!/usr/bin/env bash
neutrons={d.num_neutron}
protons={d.num_proton}
dx={d.dx}
size={round(d.size)}
l={d.l}
m={d.m}
param="{d.param}"
pref="{d.prefix}"
logdir_name="{output_folder}"
OUT="V{iteration+global_iteration}"

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
famoutfile="../$logdir_name/fam.$protons.$neutrons.$nwn.$nwp.$l.$m.$param.$size.out$OUT"
famfile="../$logdir_name/fam_data.$protons.$neutrons.$nwn.$nwp.$l.$m.$param.$size.fam$OUT"
xyfile="../$logdir_name/xy.$protons.$neutrons.$nwn.$nwp.$l.$m.$param.$size.xy$OUT"

exe="Tantalus.$pref.exe"              # full name of the mean-field executable
exefam="fam.$pref.exe"                # full name of the fam executable
param="$param"                         # name of the parameterization
workdir="{convert_to_linux(str(self.working_directory))}/work.$protons.$neutrons.$size.$nwn.$nwp$OUT"

if [ ! -d "$workdir/" ]; then
  mkdir "$workdir"
fi

cp $EXECDIR/$exe             "$workdir"/
cp $EXECDIR/$exefam          "$workdir"/
cp $PARAMDIR/"$param.param"  "$workdir"/
cd "$workdir"

if [ ! -d "../$logdir_name/" ]; then
  mkdir "../$logdir_name"
fi

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

        # if only one file is created, return just this file
        if len(omegas) == 1:
            return file
        return run_folder


    # def _check_tantalus_installation(self):
    #     return Path(self.TANTALUS_PATH).expanduser().is_dir()

    def _check_tantalus_installation(self):
        if platform.system() == "Windows":
            # Use WSL to check the Linux path
            try:
                subprocess.run(["wsl", "bash", "-c", f"test -d {self.TANTALUS_PATH}"], check=True)
                return True
            except subprocess.CalledProcessError:
                return False
        else:
            # Linux just check the path directly
            return Path(self.TANTALUS_PATH).is_dir()

    def build_snapshot_basis(self, max_workers=4, build_type=None, run_type=None):
        def launch_fam(fam_runfile):
            if platform.system() == "Windows":
                # Convert Windows path to WSL path
                wsl_path = f"/mnt/{fam_runfile.drive[0].lower()}{fam_runfile.as_posix()[2:]}"
                print(f"Launching FAM with runfile: {fam_runfile}")
                subprocess.run(["bash", wsl_path], check=True)
            else:
                subprocess.run(["bash", str(fam_runfile)], check=True)

        if build_type is not None:
            self.set_build_type(build_type)
        if run_type is not None:
            self.set_run_type(run_type)

        ## clear up wd
        if self._DEBUG_clear_wd:
            self._clear_working_directory()

        d = self.data

        if self.build_type == 'equidistant_1D':
            # static basis creation
            omegas = np.linspace(d.w_min, d.w_max, num=d.max_num_snapshots) + d.smear * 1.j
            if self.basis.is_loaded():
                raise AssertionError("Basis already loaded. Cannot initiate equidistant sampling.")
            fam_runfiles = self._create_fam_runfiles(omegas, output_folder=self.tmp_output)

            if self.run_type == 'default':
                dir = self.working_directory.joinpath(self.tmp_output)
                dir.mkdir(parents=True, exist_ok=True)

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    executor.map(launch_fam, fam_runfiles.iterdir())

                print("FAM launch completed, merging ...")
                self.path_to_snapshot = combine_FAM_output(directory=fr"{dir}",
                                                           output_dir=self.output_dir)

        elif self.build_type == 'contour':
            ### start calculations to allow contour integration!
            if self.basis.is_loaded():
                print("Basis already loaded. Adding samples")
            # check if contour parameters are set
            if not hasattr(self, 'contour_n') or not hasattr(self, 'contour_R') or not hasattr(self, 'contour_r'):
                raise ValueError("Contour parameters not set. Please set contour parameters before building contour basis.")
            samples = [self.contour_R * np.cos(arg) + 1j * self.contour_R * np.sin(arg) for arg in np.linspace(0, np.pi/2, self.contour_n)]

            fam_runfiles = self._create_fam_runfiles(samples, output_folder=self.tmp_output)
            if self.run_type == 'default':
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    executor.map(launch_fam, fam_runfiles.iterdir())
                print("FAM launch completed, merging ...")
                self.path_to_snapshot = combine_FAM_output(directory=fr"{self.working_directory.joinpath(self.tmp_output)}",
                                                           output_dir=self.output_dir,
                                                           master_file=self.path_to_snapshot if self.basis.is_loaded() else None)

        elif self.build_type == 'greedy':
            # iterative basis creation
            W_scan = np.linspace(d.w_min, d.w_max, num=2000) + d.smear * 1.j
            if not self.basis.is_loaded():
                initial_vectors = 2
                # choose two snapshots, say at 0.25 and 0.75 from the spectrum
                w1 = 0.25*(d.w_max - d.w_min) + d.smear * 1.j
                w2 = 0.75*(d.w_max - d.w_min) + d.smear * 1.j

                # initiate two FAM runs
                fam_runfiles = self._create_fam_runfiles([w1,w2], output_folder=self.tmp_output)
                #base_dir = Path(f"src/work_dir/{self.tmp_output}")
                base_dir = self.working_directory.joinpath(self.tmp_output)

                base_dir.mkdir(parents=True, exist_ok=True)

                with ThreadPoolExecutor(max_workers=2) as executor:
                    executor.map(launch_fam, fam_runfiles.iterdir())

                print("Initial FAM launch completed.")
                self.path_to_snapshot = combine_FAM_output(directory=fr"{self.working_directory.joinpath(self.tmp_output)}",
                                                      output_dir=self.output_dir,
                                                      output_name='')
            else:
                initial_vectors = len(self.basis.omegas)
                print(f"Initial basis loaded. Continuing to add {self.data.max_num_snapshots-initial_vectors} vectors")

            snapshots, snapshot_omegas, F = parse_XY_numba(self.path_to_snapshot)

            ### greedy loop
            GREEDY_COST = np.inf
            while GREEDY_COST > self.greedy_settings["cost_threshold"] and len(snapshot_omegas) < d.max_num_snapshots:
                FdF = F.conj().T @ F

                COSTS = np.zeros(len(W_scan))

                X = snapshots[:, 0, :]
                Y = snapshots[:, 1, :]

                diff_XY = X - Y

                MXdF = diff_XY.conj() @ F
                FdMX = MXdF.conj()
                XMMX = X.conj() @ X.T + Y.conj() @ Y.T

                alphas = _evaluate_PG_numba_coef(W_scan, snapshot_omegas, FdF, FdMX, MXdF, XMMX)

                for idx, omega_test in enumerate(W_scan):
                    alpha = alphas[idx]

                    COSTS[idx] = cost_numba(omega=omega_test,
                                      alpha=alpha,
                                      FdF=FdF, MXdF=MXdF, XMMX=XMMX, F_norm=np.linalg.norm(F),
                                      snapshot_omegas=snapshot_omegas)

                # ADD NEW SNAPSHOT WITH MAX COST
                max_cost_idx = np.argmax(COSTS)

                print('Initiating new FAM run for new snapshot: ', W_scan[max_cost_idx])
                # launch new FAM run for new snapshot
                fam_runfile = self._create_fam_runfiles([W_scan[max_cost_idx]],
                                                        output_folder=self.tmp_output,
                                                        global_iteration=len(snapshot_omegas)+1)
                base_dir = Path(f"src/work_dir/{self.tmp_output}")
                base_dir.mkdir(parents=True, exist_ok=True)

                launch_fam(fam_runfile)
                print("FAM launch completed for new snapshot.")

                self.path_to_snapshot = combine_FAM_output(directory=fr"{self.working_directory.joinpath(self.tmp_output)}",
                                                       output_dir=self.output_dir,
                                                       master_file=self.path_to_snapshot if self.basis.is_loaded() else None)

                # todo: change the paraser here to that it can parse just one extra one (last one in the file) and concat here...
                snapshots, snapshot_omegas, F = parse_XY_numba(self.path_to_snapshot) # update

                GREEDY_COST = np.max(COSTS)

        elif self.build_type == 'greedy_2D':
            # Define the W_scan @ the target smearing
            W_scan = np.linspace(d.w_min, d.w_max, num=2000) + d.smear * 1.j

            initial_vectors = 5
            if not self.basis.is_loaded():
                # choose FIVE snapshots, say at 0.25 and 0.75 from the spectrum @ 10 times the target smearing !
                ws = np.linspace( 0.25 * (d.w_max - d.w_min) ,  0.75 * (d.w_max - d.w_min), initial_vectors) + 10 * d.smear * 1.j
                # initiate two FAM runs
                fam_runfiles = self._create_fam_runfiles(ws, output_folder=self.tmp_output)
                self.working_directory.joinpath(self.tmp_output).mkdir(parents=True, exist_ok=True)

                with ThreadPoolExecutor(max_workers=5) as executor:
                    executor.map(launch_fam, fam_runfiles.iterdir())

                print("Initial FAM launch completed.")
                self.path_to_snapshot = combine_FAM_output(
                    directory=fr"{self.working_directory.joinpath(self.tmp_output)}",
                    output_dir=self.output_dir,
                    output_name='')
            else:
                initial_vectors = len(self.basis.omegas)
                print(
                    f"Initial basis loaded. Continuing to add {self.data.max_num_snapshots - initial_vectors} vectors")

            snapshots, snapshot_omegas, F = parse_XY_numba(self.path_to_snapshot)

            ### greedy loop
            GREEDY_COST = np.inf
            while GREEDY_COST > self.greedy_2D_settings["cost_threshold"] and len(snapshot_omegas) < d.max_num_snapshots:
                FdF = F.conj().T @ F

                COSTS = np.zeros(len(W_scan))

                X = snapshots[:, 0, :]
                Y = snapshots[:, 1, :]

                diff_XY = X - Y

                MXdF = diff_XY.conj() @ F
                FdMX = MXdF.conj()
                XMMX = X.conj() @ X.T + Y.conj() @ Y.T

                # we find the "ideal" snapshot @ the target smearing.
                alphas = _evaluate_PG_numba_coef(W_scan, snapshot_omegas, FdF, FdMX, MXdF, XMMX)

                for idx, omega_test in enumerate(W_scan):
                    alpha = alphas[idx]

                    COSTS[idx] = cost_numba(omega=omega_test,
                                            alpha=alpha,
                                            FdF=FdF, MXdF=MXdF, XMMX=XMMX, F_norm=np.linalg.norm(F),
                                            snapshot_omegas=snapshot_omegas)

                max_cost_idx = np.argmax(COSTS)
                #print('Ideal new snapshot: ', W_scan[max_cost_idx])

                # now, scan this omega value along the complex axis, and find the "sub"-optimal omega value
                H_scan = np.real(W_scan[max_cost_idx]) + np.linspace(1, 20,200)*1j*d.smear
                COSTS_H = np.zeros(len(H_scan))
                alphas = _evaluate_PG_numba_coef(H_scan, snapshot_omegas, FdF, FdMX, MXdF, XMMX)
                for idx, omega_test in enumerate(H_scan):
                    alpha = alphas[idx]

                    COSTS_H[idx] = cost_numba(omega=omega_test,
                                            alpha=alpha,
                                            FdF=FdF, MXdF=MXdF, XMMX=XMMX, F_norm=np.linalg.norm(F),
                                            snapshot_omegas=snapshot_omegas)

                # find the maximal cost in H but below a threshold
                COSTS_H = COSTS_H / np.max(COSTS_H)  # normalise wrt "highest" one
                threshold = self.greedy_2D_settings["cost_relaxation"]
                COSTS_H = np.where(COSTS_H > threshold, 0.0, COSTS_H)
                if np.all(COSTS_H == 0.0):
                    print("No new snapshot found above threshold, adding the one with largest smearing")
                    max_cost_idx_H = -1
                else:
                    max_cost_idx_H = np.argmax(COSTS_H)
                SUB_OPTIMAL_OMEGA = H_scan[max_cost_idx_H]
                print("Selected : ", SUB_OPTIMAL_OMEGA)

                # launch new FAM run for new snapshot
                fam_runfile = self._create_fam_runfiles([SUB_OPTIMAL_OMEGA],
                                                        output_folder=self.tmp_output,
                                                        global_iteration=len(snapshot_omegas)+1)
                self.working_directory.joinpath(self.tmp_output).mkdir(parents=True, exist_ok=True)

                launch_fam(fam_runfile)

                print("FAM launch completed for new snapshot.")
                self.path_to_snapshot = combine_FAM_output(
                    directory=fr"{self.working_directory.joinpath(self.tmp_output)}",
                    output_dir=self.output_dir,
                    master_file=self.path_to_snapshot if self.basis.is_loaded() else None)

                snapshots, snapshot_omegas, F = parse_XY_numba(self.path_to_snapshot)  # update

                GREEDY_COST = np.max(COSTS) # save the cost evaluated at target smearing

        else: raise ValueError("Unknown build type: " + self.build_type+". Supported types: "+self._build_types)

        # clean up and load basis
        if not self._DEBUG_clear_wd:
            print("Clearing wd is disabled.")

        elif self.run_type == 'default':
            self._clear_working_directory()
            self.basis.load(self.path_to_snapshot)
            print("Basis loaded.")
        return


    def load(self, path_to_snapshot):
        from src.Utils_parsers import parse_XY_numba
        try:
            # load snapshots and omegas from file
            self.basis.snapshots, self.basis.omegas, self.basis.F = parse_XY_numba(path_to_snapshot)
            self.path_to_snapshot = Path(path_to_snapshot)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find snapshot {path_to_snapshot}.")