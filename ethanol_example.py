import numpy as np
from calculators import QMLCalculator
import ase
import ase.optimize

def optimize(molecule, calculator, trajectory_basename=None):
    molecule.set_calculator(calculator)
    if trajectory_basename is None:
        dyn = ase.optimize.BFGS(molecule)
    else:
        dyn = ase.optimize.BFGS(molecule, trajectory=f"{trajectory_basename}.traj")
    dyn.run(fmax=0.05, steps=1000)

    if trajectory_basename is None:
        return

    from ase.io import Trajectory, write
    trajs = [t for t in Trajectory(f'{trajectory_basename}.traj')]
    write(f"{trajectory_basename}.xyz", trajs[0])
    for i, t in enumerate(trajs[1:]):
        write(f"{trajectory_basename}.xyz", t, append=True)

def ase_molecule_from_xyz(xyz_path):
    import rmsd
    atom_labels, coordinates = rmsd.get_coordinates_xyz(xyz_path)
    molecule = ase.Atoms(atom_labels, coordinates)
    return molecule

if __name__ == "__main__":
    ase_molecule = ase_molecule_from_xyz("structures/ethanol.xyz")
    optimize(ase_molecule, QMLCalculator(), "qml_optimization")
