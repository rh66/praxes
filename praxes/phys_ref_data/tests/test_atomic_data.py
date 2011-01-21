import operator as op

import numpy as np
import quantities as pq

from praxes import testing
from ..elam import atomic_data


class TestElements(testing.TestCase):

    def test_symbol(self):
        for key in ('Si', 'O'):
            self.assertEqual(atomic_data[key].symbol, key)

    def test_creation(self):
        with self.assertRaises(KeyError):
            for key in ('Z', 'O2-'):
                atomic_data[key]

    def test_atomic_mass(self):
        for key, val in (('Si', 28.0855 * pq.u), ('O', 15.9994 * pq.u)):
            self.assertAlmostEqual(atomic_data[key].atomic_mass, val)

    def test_molar_mass(self):
        for key, val in (
            ('Si', 28.0855 * pq.g/pq.mol), ('O', 15.9994 * pq.g/pq.mol)
            ):
            self.assertAlmostEqual(atomic_data[key].molar_mass, val)

    def test_atomic_number(self):
        self.assertEqual(atomic_data['H'].atomic_number, 1)

    def test_mass_density(self):
        self.assertEqual(atomic_data['C'].mass_density, 2.26 * pq.g/pq.cm**3)

    def test_photoabsorption(self):
        self.assertAlmostEqual(
            atomic_data['Cu'].photoabsorption_cross_section(10 * pq.keV),
            214.459 * pq.cm**2/pq.g,
            delta=1e-3
            )

    def test_coherent_scattering(self):
        self.assertAlmostEqual(
            atomic_data['Cu'].coherent_scattering_cross_section(10 * pq.keV),
            1.45 * pq.cm**2/pq.g,
            delta=1e-3
            )

    def test_incoherent_scattering(self):
        self.assertAlmostEqual(
            atomic_data['Cu'].incoherent_scattering_cross_section(10 * pq.keV),
            0.077 * pq.cm**2/pq.g,
            delta=1e-3
            )


class TestLevels(testing.TestCase):

    def test_creation(self):
        self.assertEqual(
            atomic_data['U'].xray_levels['K'].iupac_symbol, 'K'
            )

    def test_jump_ratio(self):
        self.assertEqual(
            atomic_data['U'].xray_levels['L2'].jump_ratio, 1.4
            )

    def test_fluorescence_yield(self):
        self.assertEqual(
            atomic_data['U'].xray_levels['L2'].fluorescence_yield, 0.467
            )

    def test_absorption_edge(self):
        self.assertEqual(
            atomic_data['U'].xray_levels['L2'].absorption_edge, 20.948*pq.keV
            )

    def test_element(self):
        self.assertEqual(
            atomic_data['U'].xray_levels['L2'].element.symbol, 'U'
            )


class TestTransitions(testing.TestCase):

    def test_creation(self):
        self.assertEqual(
            atomic_data['U'].xray_levels['K'].\
                transitions['K-L3'].iupac_symbol,
            'K-L3'
            )

    def test_initial_level(self):
        self.assertEqual(
            atomic_data['U'].xray_levels['K'].\
                transitions['K-L3'].initial_level.iupac_symbol,
            'K'
            )

    def test_final_level(self):
        self.assertEqual(
            atomic_data['U'].xray_levels['K'].\
                transitions['K-L3'].final_level.iupac_symbol,
            'L3'
            )

    def test_element(self):
        self.assertEqual(
            atomic_data['U'].xray_levels['K'].\
                transitions['K-L3'].element.symbol,
            'U'
            )

    def test_emission_energy(self):
        self.assertEqual(
            atomic_data['U'].xray_levels['K'].\
                transitions['K-L3'].emission_energy,
            98.440 * pq.keV
            )

    def test_intensity(self):
        self.assertEqual(
            atomic_data['U'].xray_levels['K'].\
                transitions['K-L3'].intensity,
            0.473147
            )

    def test_siegbahn_symbol(self):
        self.assertEqual(
            atomic_data['U'].xray_levels['K'].\
                transitions['K-L3'].siegbahn_symbol,
            'Ka1'
            )
