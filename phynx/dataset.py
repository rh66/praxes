"""
"""

from __future__ import absolute_import, with_statement

import copy
import posixpath

import h5py
import numpy as np

from .base import _PhynxProperties
from .exceptions import H5Error
from .registry import registry
from .utils import memoize, simple_eval, sync


class Dataset(_PhynxProperties, h5py.Dataset):

    """
    """

    def _get_acquired(self):
        return self.attrs.get('acquired', self.npoints)
    def _set_acquired(self, value):
        self.attrs['acquired'] = int(value)
    acquired = property(_get_acquired, _set_acquired)

    @property
    def entry(self):
        try:
            target = self.file['/'.join(self.parent.name.split('/')[:2])]
            assert isinstance(target, registry['Entry'])
            return target
        except AssertionError:
            return None

    @property
    @sync
    def map(self):
        res = np.zeros(self.acquisition_shape, self.dtype)
        res.flat[:len(self)] = self.value.flatten()
        return res

    @property
    def measurement(self):
        try:
            return self.entry.measurement
        except AttributeError:
            return None

    def __init__(
        self, parent_object, name, shape=None, dtype=None, data=None,
        chunks=None, compression='gzip', shuffle=None, fletcher32=None,
        maxshape=None, compression_opts=None, **kwargs
    ):
        if data is None and shape is None:
            h5py.Dataset.__init__(self, parent_object, name)
            _PhynxProperties.__init__(self, parent_object)
        else:
            h5py.Dataset.__init__(
                self, parent_object, name, shape=shape, dtype=dtype,
                data=data, chunks=chunks, compression=compression,
                shuffle=shuffle, fletcher32=fletcher32, maxshape=maxshape,
                compression_opts=compression_opts
            )
            _PhynxProperties.__init__(self, parent_object)

            self.attrs['class'] = self.__class__.__name__

        for key, val in kwargs.iteritems():
            if not np.isscalar(val):
                val = str(val)
            self.attrs[key] = val

    def __getitem__(self, args):
        if isinstance(args, int):
            # this is a speedup to workaround an hdf5 indexing bug
            res = super(Dataset, self).__getitem__(slice(args, args+1))
            res.shape = res.shape[1:]
            return res
        else:
            return super(Dataset, self).__getitem__(args)

    @sync
    def __repr__(self):
        try:
            return '<%s dataset "%s": shape %s, type "%s" (%d attrs)>'%(
                self.__class__.__name__,
                self.name,
                self.shape,
                self.dtype.str,
                len(self.attrs)
            )
        except Exception:
            return "<Closed %s dataset>" % self.__class__.__name__

    @sync
    def mean(self, indices=None):
        if indices is None:
            indices = range(self.acquired)
        elif len(indices):
            indices = [i for i in indices if i < self.acquired]

        res = np.zeros(self.shape[1:], 'f')
        nitems = 0
        for i in indices:
            if not self.measurement.masked[i]:
                nitems += 1
                res += self[i]
        if nitems:
            return res / nitems
        return res


class Axis(Dataset):

    """
    """

    @property
    def axis(self):
        return self.attrs.get('axis', 0)

    @property
    def primary(self):
        return self.attrs.get('primary', 0)

    @property
    def range(self):
        try:
            return simple_eval(self.attrs['range'])
        except H5Error:
            return (self.value[[0, -1]])

    @sync
    def __cmp__(self, other):
        try:
            assert isinstance(other, Axis)
            return cmp(self.primary, other.primary)
        except AssertionError:
            raise AssertionError(
                'Cannot compare Axis and %s'%other.__class__.__name__
            )


class Signal(Dataset):

    """
    """

    def _get_efficiency(self):
        return self.attrs.get('efficiency', 1)
    def _set_efficiency(self, value):
        self.attrs['efficiency'] = float(value)
    efficiency = property(_get_efficiency, _set_efficiency)

    @property
    def signal(self):
        return self.attrs.get('signal', 0)

    @property
    @memoize
    def corrected_value(self):
        return CorrectedDataProxy(self)

    @sync
    def __cmp__(self, other):
        try:
            assert isinstance(other, Signal)
            ss = self.signal if self.signal else 999
            os = other.signal if other.signal else 999
            return cmp(ss, os)
        except AssertionError:
            raise AssertionError(
                'Cannot compare Signal and %s'%other.__class__.__name__
            )


class ImageData(Signal):

    """
    """


class DeadTime(Signal):

    """
    The native format of the dead time data needs to be specified. This can be
    done when creating a new DeadTime dataset by passing a dead_time_format
    keyword argument with one of the following values:

    * 'percent' - the percent of the real time that the detector is not live
    * '%' - same as 'percent'
    * 'fraction' - the fraction of the real time that the detector is not live
    * 'normalization' - data is corrected by dividing by the dead time value
    * 'correction' - data is corrected by muliplying by the dead time value

    Alternatively, the native format can be specified after the fact by setting
    the format property to one of the values listed above.
    """

    @property
    @memoize
    def correction(self):
        return DeadTimeProxy(self, 'correction')

    @property
    @memoize
    def percent(self):
        return DeadTimeProxy(self, 'percent')

    @property
    @memoize
    def fraction(self):
        return DeadTimeProxy(self, 'fraction')

    @property
    @memoize
    def normalization(self):
        return DeadTimeProxy(self, 'normalization')

    @property
    @memoize
    def format(self):
        return self.attrs['dead_time_format']

    def __init__(self, *args, **kwargs):
        format = kwargs.pop('dead_time_format', None)
        super(DeadTime, self).__init__(*args, **kwargs)

        if format:
            valid = ('percent', '%', 'fraction', 'normalization', 'correction')
            if format not in valid:
                raise ValueError(
                    'dead time format must one of: %r, got %s'
                    % (', '.join(valid), format)
                    )
            self.attrs['dead_time_format'] = format


class DataProxy(object):

    @property
    def acquired(self):
        return self._dset.acquired

    @property
    def map(self):
        res = np.zeros(self._dset.acquisition_shape, self._dset.dtype)
        res.flat[:len(self)] = self[:].flatten()
        return res

    @property
    def measurement(self):
        return self._dset.measurement

    @property
    def npoints(self):
        return self._dset.npoints

    @property
    def plock(self):
        return self._dset.plock

    @property
    def shape(self):
        return self._dset.shape

    def __init__(self, dataset):
        with dataset.plock:
            self._dset = dataset

    @sync
    def __getitem__(self, args):
        raise NotImplementedError(
            '__getitem__ must be implemented by $s' % self.__class__.__name__
        )

    def __len__(self):
        return len(self._dset)

    @sync
    def mean(self, indices=None):
        if indices is None:
            indices = range(self.acquired)
        elif len(indices):
            indices = [i for i in indices if i < self.acquired]

        res = np.zeros(self.shape[1:], 'f')
        nitems = 0
        for i in indices:
            if not self.measurement.masked[i]:
                nitems += 1
                res += self[i]
        if nitems:
            return res / nitems
        return res


class CorrectedDataProxy(DataProxy):

    @sync
    def __getitem__(self, key):
        data = self._dset[key]

        try:
            data /= self._dset.efficiency
        except AttributeError:
            pass

        # normalization may be something like ring current or monitor counts
        try:
            norm = self._dset.parent['normalization'][key]
            if norm.shape and len(norm.shape) < len(data.shape):
                newshape = [1]*len(data.shape)
                newshape[:len(norm.shape)] = norm.shape
                norm.shape = newshape
            data /= norm
        except H5Error:
            # fails if normalization is not defined
            pass

        # detector deadtime correction
        try:
            dtc = self._dset.parent['dead_time'].correction[key]
            if isinstance(dtc, np.ndarray) \
                    and len(dtc.shape) < len(data.shape):
                newshape = [1]*len(data.shape)
                newshape[:len(dtc.shape)] = dtc.shape
                dtn.shape = newshape
            data *= dtc
        except H5Error:
            # fails if dead_time_correction is not defined
            pass

        return data


class DeadTimeProxy(DataProxy):

    @property
    @memoize
    def format(self):
        return self._format

    def __init__(self, dataset, format):
        with dataset.plock:
            super(DeadTimeProxy, self).__init__(dataset)

            assert format in (
                'percent', '%', 'fraction', 'normalization', 'correction'
            )
            self._format = format

    @sync
    def __getitem__(self, args):
        if self._dset.format == 'fraction':
            fraction = self._dset.__getitem__(args)
        elif self._dset.format in ('percent', '%'):
            fraction = self._dset.__getitem__(args) / 100.0
        elif self._dset.format == 'correction':
            fraction = self._dset.__getitem__(args) - 1
        elif self._dset.format == 'normalization':
            fraction = 1.0 / self._dset.__getitem__(args) - 1
        else:
            raise ValueError(
                'Unrecognized dead time format: %s' % self._dset.format
            )

        if self.format == 'fraction':
            return fraction
        elif self.format in ('percent', '%'):
            return 100 * fraction
        elif self.format == 'correction':
            return 1 / (1 - fraction)
        elif self.format == 'normalization':
            return 1 - fraction
        else:
            raise ValueError(
                'Unrecognized dead time format: %s' % self.format
            )
