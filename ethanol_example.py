import numpy as np
from calculators import QMLCalculator
import rmsd
import ase
import ase.optimize

def optimize(xyz_path):
    calculator = QMLCalculator()
    atom_labels, coordinates = rmsd.get_coordinates_xyz(xyz_path)
    molecule = ase.Atoms(atom_labels, coordinates)
    molecule.set_calculator(calculator)
    dyn = ase.optimize.BFGS(molecule, trajectory="test.traj")
    dyn.run(fmax=0.05, steps=100)
    from ase.io import read, write, Trajectory
    trajs = [t for t in Trajectory('test.traj')]
    write("test.xyz", trajs[0])
    for i, t in enumerate(trajs[1:]):
        write("test.xyz", t, append=True)

if __name__ == "__main__":
    optimize("examples/ethanol.xyz")
