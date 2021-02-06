""" Modified from https://github.com/andersx/qml-ase/blob/master/calculators.py
"""

import numpy as np
import time
import joblib
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
try:
    from sklearn.linear_model import Lasso
    from qml.representations import generate_fchl_acsf
    from qml.kernels.gradient_kernels import get_local_kernel, get_local_gradient_kernel
    QML_AVAILABLE = True
except:
    QML_AVAILABLE = False


class BaseCalculator(Calculator):
    name = 'BaseCalculator'
    implemented_properties = ['energy', 'forces']
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.e_total = None
        self.forces = None

    def calculation_required(self, atoms, properties):
        if self.atoms != atoms:
            return True
        #elif self.atoms.get_positions() is not atoms.get_positions(): # Works for energy/force checks
        elif not np.array_equal(self.atoms.get_positions(), atoms.get_positions()): # Works only for force checks
            return True
        for prop in properties:
            if prop == 'energy' and self.e_total is None:
                return True
            elif prop == 'forces' and self.forces is None:
                return True
        return False

    def calculate(self, atoms: Atoms = None, properties=('energy', 'forces'),
                  system_changes=all_changes):

        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError(
                'No ASE atoms supplied to calculation, and no ASE atoms supplied with initialisation.')

        self.query(atoms)

        if 'energy' in properties:
            self.results['energy'] = self.e_total

        if 'forces' in properties:
            self.results['forces'] = self.forces

        return

    def get_potential_energy(self, atoms=None, force_consistent=False):
        # Only having calculation_required calls in forces happens to be faster,
        # since we can add more stringent criteria in the method
        #if self.calculation_required(atoms, ["energy"]):
        self.query(atoms)
        return self.e_total

    def get_forces(self, atoms=None):
        if self.calculation_required(atoms, ["forces"]):
            self.query(atoms=atoms)
        return self.forces

class QMLCalculator(BaseCalculator):
    name = 'QMLCalculator'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not QML_AVAILABLE:
            print("QML not available")
            raise SystemExit

        model = joblib.load('models/reactive_fchl.pkl')
        self._set_model(model)

    def _set_model(self, model):

        self.alphas = model['alphas']
        self.repr = model['training_representations']
        self.charges = model['nuclear_charges']
        self.n_atoms = len(self.charges[0])
        self.elements = model['params']['elements']

        ## Offset from training
        #self.scaler = model['scaler']

        # Hyper-parameters
        self.sigma = model['sigma']
        self.params = model['params']

        return

    def _featurizer(self, nuclear_charges):
        """
        Get the counts of each element as features.
        """

        n = len(nuclear_charges)
        m = len(self.elements)
        element_to_index = {v:i for i, v in enumerate(self.elements)}
        features = np.zeros((n,m), dtype=int)

        for i, charge in enumerate(nuclear_charges):
            count_dict = {k:v for k,v in zip(*np.unique(charge, return_counts=True))}
            for key, value in count_dict.items():
                if key not in element_to_index:
                    continue
                j = element_to_index[key]
                features[i, j] = value

        return features

    def query(self, atoms=None, print_time=False):
        if print_time:
            start = time.time()

        # Store latest positions
        self.atoms = atoms

        # kcal/mol to ev (via 627.5 kcal/mol per hartree)
        # kcal/mol/aangstrom to ev / aangstrom
        conv_energy = 0.04336478087649403
        conv_force = 0.04336478087649403

        coordinates = atoms.get_positions()
        nuclear_charges = atoms.get_atomic_numbers()
        n_atoms = coordinates.shape[0]

        rep, drep = generate_fchl_acsf(
            nuclear_charges,
            coordinates,
            gradients=True,
            **self.params)

        # Put data into arrays
        Qs = [nuclear_charges]
        Xs = np.array([rep], order="F")
        dXs = np.array([drep], order="F")

        # Get kernels
        energy_kernel = get_local_kernel(self.repr, Xs, self.charges, Qs, self.sigma)
        force_kernel = get_local_gradient_kernel(self.repr, Xs, dXs, self.charges, Qs, self.sigma)
        kernel = np.concatenate((energy_kernel, force_kernel))

        # Get predictions
        energy_and_forces = np.dot(kernel, self.alphas)

        # Get energy offset
        #features = self._featurizer(Qs)
        #offset = self.scaler.predict(features)[0]

        # Energy prediction
        energy = energy_and_forces[0]# + offset
        self.e_total = energy * conv_energy

        # Force prediction
        forces = energy_and_forces[1:].reshape(-1, 3)
        self.forces = forces * conv_force

        return
