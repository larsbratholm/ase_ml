import numpy as np
from calculators import QMLCalculator, RdkitCalculator
import ase
import ase.optimize

def optimize(molecule, calculator, trajectory_basename=None):
    molecule.set_calculator(calculator)
    if trajectory_basename is None:
        dyn = ase.optimize.BFGS(molecule)
    else:
        import ase.optimize.sciopt
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

def rdkit_molobj_from_xyz(xyz_path):
    from xyz2mol import xyz2mol, read_xyz_file
    atom_labels, _, coordinates = read_xyz_file(xyz_path)
    molobj = xyz2mol(atom_labels, coordinates, allow_charged_fragments=False,
                     use_huckel=True, use_graph=True)
    return molobj[0]


if __name__ == "__main__":
    ase_molecule = ase_molecule_from_xyz("structures/ethanol.xyz")

    #Pre-optimization with mmff
    rdkit_molobj = rdkit_molobj_from_xyz("structures/ethanol.xyz")
    optimize(ase_molecule, RdkitCalculator(rdkit_molobj), "ff_optimization")

    #optimize(ase_molecule, QMLCalculator(), "qml_optimization")
