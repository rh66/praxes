# -*- coding: utf-8 -*-
'''
This is an interface to a database of fundamental X-ray fluorescence
parameters compiled by W.T. Elam, B.D. Ravel and J.R. Sieber, and
published in Radiation Physics and Chemistry, 63 (2), 121 (2002). The
database is published by NIST at
http://www.cstl.nist.gov/acd/839.01/xrfdownload.html

 * densities are given in gm/cm^3

 * photoabsorption, coherent and incoherent scattering are log values

'''

from __future__ import absolute_import

from collections import OrderedDict
import json
import os
import sqlite3
import textwrap

import numpy as np
import quantities as pq

from .interpolate import splint
from . import base
from .decorators import memoize


class ElamSQLiteConnection(sqlite3.Connection):

    def __init__(self):
        super(ElamSQLiteConnection, self).__init__(
            os.path.join(os.path.split(__file__)[0], 'elam.db')
            )

elamdb = ElamSQLiteConnection()


class Transition(object):
    """
    Publicly accessible attributes:
    iupac
    siegbahn
    energy (eV)
    intensity (fraction)

    The following is quoted verbatim from the ElamDB12.txt file:

    Relative emission rates, fits to low-order polynomials, low-Z
    extrapolations by hand and eye data from Salem, Panossian, and
    Krause, Atomic Data and Nuclear Data Tables Vol. 14 No.2 August
    1974, pp92-109. M shell data is from T. P. Schreiber and A. M.
    Wims, X-ray Spectrometry Vol. 11, No. 2, 1982, pp42-45. Small,
    arbitrary intensities assigned to Mgamma and Mzeta lines.
    """

    def _get_data(self, id):
        cursor = elamdb.cursor()
        result = cursor.execute(
            '''select %s from emission_lines
            where element=? and iupac_symbol=?''' % id,
            (self._element, self._iupac)
            ).fetchone()
        cursor.close()
        return result[0]

    @property
    def final_level(self):
        return self._get_data('final_level')

    @property
    def initial_level(self):
        return self._get_data('initial_level')

    @property
    def element(self):
        return self._element

    @property
    @memoize
    def emission_energy(self):
        return self._get_data('energy') * pq.eV

    @property
    @memoize
    def intensity(self):
        return self._get_data('intensity')

    @property
    def iupac(self):
        return self._iupac

    @property
    @memoize
    def siegbahn(self):
        return self._get_data('Siegbahn_symbol')

    def __init__(self, element, iupac):
        self._element = element
        self._iupac = iupac

    @memoize
    def __repr__(self):
        return "<%s(%s, %s)>" % (self.__class__, self.element, self.iupac)

    @memoize
    def __str__(self):
        return textwrap.dedent(
            """\
            %s(%s, %s)
              emission energy: %s
              intensity: %s
              Siegbahn symbol: %s""" % (
                str(self.__class__).split('.')[-1],
                self.element,
                self.iupac,
                self.emission_energy,
                self.intensity,
                self.siegbahn,
                )
            )


class XrayLevel(object):
    """
    Publicly accessible attributes:
    name
    energy (eV)
    fluorescence_yield
    jump_ratio

    The following is quoted verbatim from the ElamDB12.txt file:

    K-shell fluorescence yield below Z=11 from new fits in J. H.
    Hubbell et. al., J. Chem. Phys. Ref. Data, Vol. 23, No. 2, 1994,
    pp339-364. Fluorescence yields and Coster-Kronig transition
    rates for K and L shells Krause, J. Phys. Chem. Ref. Data, Vol.
    8, No. 2, 1979, pp307-327. Values for wK, wL2,and f23 are from
    Table 1. (values for light atoms in condensed matter) (note that
    this produces a large step in f23 values at z=30, see discussion
    in reference section 5.3 L2 Subshell and section 7 last
    paragraph)

    Values of wL1 for Z=85-110 and f12 for Z=72-96 from Krause were
    modified as suggested by W. Jitschin, "Progress in Measurements
    of L-Subshell Fluorescence, Coster-Kronig, and Auger Values", AIP
    Conference Proceedings 215, X-ray and Inner-Shell Processes,
    Knoxville, TN, 1990. T. A. Carlson, M. O. Krause, and S. T.
    Manson, Eds. (American Institute of Physics, 1990).

    Fluorescence yields and Coster-Kronig transition rates for M
    shells Eugene J. McGuire, "Atomic M-Shell Coster-Kronig, Auger,
    and Radiative Rates, and Fluorescence Yields for Ca-Th", Physical
    Review A, Vol. 5, No. 3, March 1972, pp1043-1047.

    Fluorescence yields and Coster-Kronig transition rates for N
    shells Eugene J. McGuire, "Atomic N-shell Coster-Kronig, Auger,
    and Radiative Rates and Fluorescence Yields for 38 <= Z <= 103",
    Physical Review A 9, No. 5, May 1974, pp1840-1851. Values for
    Z=38 to 50 were adjusted according to instructions on page 1845,
    at the end of Section IV.a., and the last sentence of the
    conclusions.
    """

    @property
    @memoize
    def ck_probabilities(self):
        c = elamdb.cursor()
        items = c.execute(
            '''select final_level, transition_probability from Coster_Kronig
            where element=? and initial_level=? order by final_level''',
            (self.element, self.name)
            )
        return OrderedDict(items)

    @property
    @memoize
    def ck_total_probabilities(self):
        c = elamdb.cursor()
        items = c.execute(
            '''select final_level, total_transition_probability
            from Coster_Kronig
            where element=? and initial_level=? order by final_level''',
            (self.element, self.name)
            )
        return OrderedDict(items)

    def _get_data(self, id):
        cursor = elamdb.cursor()
        result = cursor.execute('''select %s from xray_levels
            where element=? and label=?''' % id, (self._element, self._name)
            ).fetchone()
        cursor.close()
        return result[0]

    @property
    def element(self):
        return self._element

    @property
    @memoize
    def absorption_edge(self):
        return self._get_data('absorption_edge') * pq.eV

    @property
    @memoize
    def fluorescence_yield(self):
        return self._get_data('fluorescence_yield')

    @property
    @memoize
    def jump_ratio(self):
        return self._get_data('jump_ratio')

    @property
    @memoize
    def transitions(self):
        c = elamdb.cursor()
        keys = c.execute(
            '''select iupac_symbol from emission_lines
            where element=? and initial_level=? order by iupac_symbol''',
            (self.element, self.name)
            )
        res = OrderedDict()
        for (key, ) in keys:
            res[key] = Transition(self.element, key)
        return res

    @property
    def name(self):
        return self._name

    def __init__(self, element, name):
        self._element = element
        self._name = name

    @memoize
    def __repr__(self):
        return "<%s(%s, %s)>" % (self.__class__, self.element, self.name)

    @memoize
    def __str__(self):
        return textwrap.dedent(
            """\
            %s(%s, %s)
              absorption edge: %s
              flourescence yield: %s
              jump ratio: %s
              Coster Kronig probabilities: %s
              Coster Kronig total probabilities: %s
              transitions: %s""" % (
                str(self.__class__).split('.')[-1],
                self.element,
                self.name,
                self.absorption_edge,
                self.fluorescence_yield,
                self.jump_ratio,
                self.ck_probabilities.items(),
                self.ck_total_probabilities.items(),
                self.transitions.keys()
                )
            )


class AtomicData(base.AtomicData):
    """
    Publicly accessible attributes:

    symbol
    atomic_number
    molar_mass
    mass_density (gm/cm3)

    Densities are theoretical solid densities for all elements,
    regardless of state at STP, unless otherwise specified.
    """

    def _get_data(self, id):
        cursor = elamdb.cursor()
        result = cursor.execute('''select %s from elements
            where element=?''' % id, (self.element, )
            ).fetchone()
        cursor.close()
        return result[0]

    @property
    @memoize
    def atomic_mass(self):
        return (self.molar_mass / pq.constants.N_A).rescale('amu')

    @property
    @memoize
    def _coherent_scatter(self):
        return CoherentScatter(self.element)

    @property
    @memoize
    def xray_levels(self):
        c = elamdb.cursor()
        keys = c.execute(
            '''select label from xray_levels where element=?
            order by absorption_edge desc''',
            (self.element,)
            )
        res = OrderedDict()
        for (key, ) in keys:
            res[key] = XrayLevel(self.element, key)
        return res

    @property
    @memoize
    def _incoherent_scatter(self):
        return IncoherentScatter(self.element)

    @property
    @memoize
    def mass_density(self):
        return self._get_data('density') * pq.g / pq.cm**3

    @property
    @memoize
    def molar_mass(self):
        return self._get_data('molar_mass') * pq.g / pq.mol

    @property
    @memoize
    def _photoabsorption(self):
        return Photoabsorption(self.element)

    def __init__(self, symbol):
        """
        symbol is a string, like 'Ca' or 'Ca2+'
        """
        base.AtomicData.__init__(self, symbol)
        try:
            self._get_data('element')
        except AssertionError:
            raise NotImplementedError(
                'Fluorescence data have not been reported for %s'
                % self.element
                )

    @memoize
    def __repr__(self):
        return "<%s(%s)>" % (self.__class__, self.symbol)

    @memoize
    def __str__(self):
        return textwrap.dedent(
            """\
            %s(%s)
              mass density: %s
              molar mass: %s
              x-ray levels: %s""" % (
                str(self.__class__).split('.')[-1],
                self.symbol,
                self.mass_density,
                self.molar_mass,
                self.xray_levels.keys()
                )
            )

    def photoabsorption_cross_section(self, energy):
        """
        Return the photoabsorption cross section for 100 < E < 8e5 eV
        """
        res = self._photoabsorption(energy) * self.atomic_mass
        res.units = 'cm**2'
        return res

    def coherent_scattering_cross_section(self, energy):
        """
        Return the coherent scattering cross section for 100 < E < 8e5 eV
        """
        res = self._coherent_scatter(energy) * self.atomic_mass
        res.units = 'cm**2'
        return res

    def incoherent_scattering_cross_section(self, energy):
        """
        Return the incoherent scattering cross section for 100 < E < 8e5 eV
        """
        res = self._incoherent_scatter(energy) * self.atomic_mass
        res.units = 'cm**2'
        return res


class SplineInterpolable(object):

    @property
    def element(self):
        return self._element

    @property
    @memoize
    def _log_independant_value(self):
        return self._get_data('log_energy')

    def __init__(self, element):
        self._element = element

    def __call__(self, energy):
        log_energy = np.log(energy.rescale('eV').magnitude)
        log_xa = self._log_independant_value
        log_ya = self._log_dependant_value
        log_yspline = self._log_dependant_value_spline

        log_res = splint(log_xa, log_ya, log_yspline, log_energy)
        return np.exp(log_res) * pq.cm**2 / pq.g


class Photoabsorption(SplineInterpolable):

    def _get_data(self, id):
        cursor = elamdb.cursor()
        result = cursor.execute('''select %s from photoabsorption
            where element=?''' % id, (self.element, )
            ).fetchone()
        cursor.close()
        return np.array(json.loads(result[0]))

    @property
    @memoize
    def _log_dependant_value(self):
        return self._get_data('log_photoabsorption')

    @property
    @memoize
    def _log_dependant_value_spline(self):
        return self._get_data('log_photoabsorption_spline')


class Scatter(SplineInterpolable):

    def _get_data(self, id):
        cursor = elamdb.cursor()
        result = cursor.execute('''select %s from scattering
            where element=?''' % id, (self.element, )
            ).fetchone()
        cursor.close()
        return np.array(json.loads(result[0]))


class CoherentScatter(Scatter):

    @property
    @memoize
    def _log_dependant_value(self):
        return self._get_data('log_coherent_scatter')

    @property
    @memoize
    def _log_dependant_value_spline(self):
        return self._get_data('log_coherent_scatter_spline')


class IncoherentScatter(Scatter):

    @property
    @memoize
    def _log_dependant_value(self):
        return self._get_data('log_incoherent_scatter')

    @property
    @memoize
    def _log_dependant_value_spline(self):
        return self._get_data('log_incoherent_scatter_spline')