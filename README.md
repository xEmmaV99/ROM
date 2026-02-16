

This repository can be used to create an emulator for QRPA linear response. The emulator applies reduced order modelling to extrapolate from previous FAM output. 


 The repository is structured as follows:
```text
ROM/
├── _input/
├── _output/
├── example_scripts/
│   ├── create_snapshots.py
│   └── load_snapshots.py
├── src/
│   ├── __init__.py
│   ├── Emulator.py
│   ├── ROM_basis.py
│   └── Utils.py
├── .gitignore
└──  README.md
```
As suggested by the ```example_scripts```, there are two ways (online and offline) of using this repo. The online stage is emulator creation while the offline stage is emulator usage.

### Online: Emulator Creation
#### Required: mean-field solution and mean-field output in ```_input```, tantalus installation
Initiate a ```ROM_builder``` object. This requires a meanfield solution and meanfield output. FAM parameters can be passed through a dictionary, or updated after initialisation using ```rom_builder.data.set_FAM_parameters()```. 


The snapshot creation is started using ```build_snapshot_basis``` and specifying the build type (sampling method). This initiates the FAM calculations using tantalus which are required to obtain the basis vectors.

In case the ```build_type == 'greedy'```, one can load previous runs using ```rom_builder.basis.load()```, the greedy search will continue to add basis vectors.


### Offline: Emulator Application
#### Required: Emulator XY matrices in ```_output```
In order to set up the emulator, one has to load a basis with ```basis = ROM_basis()```,  ```basis.load(path)``` and then calling ``` emulator = Emulator(basis)```. The projection method is specified using the ```emulator.projection_method``` field. 

To apply the emulator to target values, simply use ```emulator.evaluate(target_values)```.

