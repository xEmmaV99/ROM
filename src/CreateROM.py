"""
For a given meanfield solution, start to build a ROM basis for the response function.
"""
import subprocess
from pathlib import Path
import numpy as np


class CreateROM:
    def __init__(self, path_to_meanfield_wf=None, p=10, n=10, w_min=0.0, w_max=50.0, smear=1.0, snapshots=20):
        self.num_proton = p
        self.num_neutron = n

        self.build_type = 'equidistant_1D'
        self.set_omega_range(w_min, w_max)
        self.path_to_meanfield_wf = path_to_meanfield_wf
        self.num_snapshots = snapshots
        self.l = 0  # multipolarity
        self.m = 0  # magnetic quantum number
        self.smear = smear  # default smearing width in MeV

        ### tantalus specific parameters ###
        self.param = "BSkG2"
        self.param_prefix = {"BSkG2": "BXL"} # dict for parameterization prefixes @ tantalus specific

        BASE_DIR = Path(__file__).resolve().parent
        self.working_directory = BASE_DIR.joinpath('work_dir')
        print(f"Working directory set to: {self.working_directory}")
        if path_to_meanfield_wf is None:
            print("No meanfield path provided. No FAM runs will be launched.")
            self.launch_FAM = False
        else:
            print("Meanfield path provided.")
            if self.check_tantalus_installation():
                print("Tantalus installation found.")
                self.launch_FAM = True
            else:
                print("Tantalus installation not found. No FAM runs will be launched.")
                self.launch_FAM = False

    def check_tantalus_installation(self):
        # check if tantalus is installed
        self.TANTALUS_PATH = "$HOME/code/tantalus/"
        return True

    def set_omega_range(self, w_min, w_max):
        self.w_min = w_min
        self.w_max = w_max
        pass

    def build_snapshot_basis_static(self):
        # static basis creation
        omegas = None
        if self.build_type == 'equidistant_1D':
            omegas = np.linspace(self.w_min, self.w_max, num=self.num_snapshots) + self.smear * 1.j

        if omegas is None:
            raise ValueError("No omegas defined for snapshot basis.")

        if self.launch_FAM: # start FAM calculation
            # output folder name
            OUTPUT_NAME = "TMP_SNAPSHOTS"

            fam_runfiles = self.create_fam_runfiles(omegas, output_folder=OUTPUT_NAME)
            RUN_FAM = input("Do you want to launch the FAM calculation now? (y/n): ") == 'y'
            for fam_runfile in fam_runfiles.iterdir():
                ###  launch FAM with this runfile ###
                # Convert Windows path to WSL path
                wsl_path = f"/mnt/{fam_runfile.drive[0].lower()}{fam_runfile.as_posix()[2:]}"
                if RUN_FAM:
                    subprocess.run(["bash", wsl_path], check=True)
                else:
                    print(f"FAM runfile created at {fam_runfile}. You can launch it manually later.")

            print("FAM launch completed.")

            # merge all output folders into one
            # from the folder OUTPUT_NAME read all files, merge them into one, and delete the old ones
            print(f"Merging FAM output files into folder: {OUTPUT_NAME}")


            #self.clear_working_directory()
            return

        else:
            print("FAM launch skipped due to missing Tantalus installation or meanfield path.")
            print("Defined omegas for snapshot basis:")
            for omega in omegas:
                print(omega)
            return

    def clear_working_directory(self):
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

    def create_fam_runfiles(self, omegas_step, output_folder=""):
        """
        Create a FAM runfile for the given omegas
        """
        run_folder = self.working_directory.joinpath('tmp')
        run_folder.mkdir(parents=True, exist_ok=True)
        for omega in omegas_step:
            Rew = omega.real
            Imw = omega.imag
            file = run_folder.joinpath(f"fam_run{Rew}+{Imw}j.sh")
            file.parent.mkdir(parents=True, exist_ok=True)

            # it is probably easier to create a file for each omega and then merge all together

            script = f"""\
#!/usr/bin/env bash
neutrons={self.num_neutron}
protons={self.num_proton}
dx=0.8
size=16
l={self.l}
m={self.m}
param="{self.param}"
pref="{self.param_prefix[self.param]}"
logdir_name="{output_folder}"
OUT="{Rew}+{Imw}j"

# Recompute dependent variables if needed
nwn=30
nwp=30
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
workdir="src/work_dir/work.$protons.$neutrons.$size.$nwn.$nwp"

if [ ! -d "$workdir/" ]; then
  mkdir "$workdir"
fi
if [ ! -d "src/work_dir/$logdir_name/" ]; then
  mkdir "src/work_dir/$logdir_name"
fi

cp $EXECDIR/$exe             "$workdir"/
cp $EXECDIR/$exefam          "$workdir"/
cp $PARAMDIR/"$param.param"  "$workdir"/
cd "$workdir"

echo "Starting FAM calculation"
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
fam_precision=1e-5
/
EOF

./$exefam < fam.data > $famoutfile
fam_check=$?

cat "$famfile"
        """
            file.write_text(script, newline="\n")#, encoding="utf-8")#, newline="\n")
            file.chmod(0o755)

        return run_folder
