"""
"""

from __future__ import absolute_import

import copy
import logging
import os

import numpy as np
from PyQt4 import QtCore
from SpecClient import SpecScan, SpecCommand, SpecConnectionsManager, \
    SpecEventsDispatcher, SpecWaitObject
import h5py

from . import TEST_SPEC
from xpaxs.io import phynx


logger = logging.getLogger(__file__)


class QtSpecScanA(SpecScan.SpecScanA, QtCore.QObject):

    def __init__(self, specVersion, parent=None):
        QtCore.QObject.__init__(self, parent)
        SpecScan.SpecScanA.__init__(self, specVersion)

        self._resume = SpecCommand.SpecCommandA('scan_on', specVersion)
        self._scan_aborted = SpecCommand.SpecCommandA('_SC_NEWSCAN = 0', specVersion)

        self._scanData = None
        self._lastPoint = None

    def __call__(self, cmd):
        if self.connection.isSpecConnected():
            self.connection.send_msg_cmd(cmd)

    def abort(self):
        if self.isScanning():
            self.connection.abort()
            self._scan_aborted()
            try:
                self._scanData.npoints = self._lastPoint
            except (AttributeError, h5py.h5.H5Error, TypeError):
                pass
            self.scanAborted()

    def connected(self):
        pass

    def disconnected(self):
        pass

    def newScan(self, scanParameters):
        logger.debug('newScan: %s', scanParameters)

        tree = scanParameters['phynx']
        info = tree.pop('info')

        import xpaxs
        fileInterface = xpaxs.application.getService('FileInterface')

        if fileInterface:
            specFile = info['file_name']
            h5File = fileInterface.getH5FileFromKey(specFile)
            # It is possible for a scan number to appear multiple times in a
            # spec file. Booo!
            name = str(info['acquisition_id'])
            acq_order = ''
            i = 0
            while (name + acq_order) in h5File:
                i += 1
                acq_order = '.%d'%i
            name = name + acq_order

            # create the entry and measurement groups
            entry = h5File.create_group(name, type='Entry', **info)
            measurement = entry.create_group(
                'measurement', type='Measurement'
            )
            # create all the groups under measurement, defined by clientutils:
            keys = sorted(tree.keys())
            for k in keys:
                t, kwargs = tree.pop(k)
                if 'shape' in kwargs and 'dtype' in kwargs:
                    # these are empty datasets, lets start small and grow
                    kwargs['maxshape'] = kwargs['shape']
                    kwargs['shape'] = (1, ) + kwargs['shape'][1:]
                phynx.registry[t](measurement, k, **kwargs)

            # make a few links:
            if 'masked' in measurement['scalar_data']:
                for k, val in measurement.mcas:
                    val['masked'] = measurement['scalar_data/masked']

            self._scanData = entry

            ScanView = xpaxs.application.getService('ScanView')
            if ScanView:
                view = ScanView(entry)

            self.connect(
                self,
                QtCore.SIGNAL('beginProcessing'),
                view.processData
            )

        self.emit(QtCore.SIGNAL("newScanLength"), info['npoints'])

    def newScanData(self, scanData):
        logger.debug( 'scanData: %s', scanData)

        i = int(scanData.pop('i'))

        if self._scanData:
            with self._scanData.plock:
                m = self._scanData['measurement']
                for k, val in scanData.iteritems():
                    try:
                        m[k][i] = val
                    except ValueError:
                        m[k].resize(i+1, axis=0)
                        m[k][i] = val

        self._lastPoint = i
        if i == 0:
            self.emit(QtCore.SIGNAL("beginProcessing"))
        self.emit(QtCore.SIGNAL("newScanData"), i)


    def newScanPoint(self, i, x, y, scanData):
        scanData['i'] = i
        scanData['x'] = x
        scanData['y'] = y
        logger.debug( "newScanPoint: %s", i)
        self.emit(QtCore.SIGNAL("newScanPoint"), i)

    def pause(self):
        logger.info('Scan Paused')
        self.connection.abort()

    def resume(self):
        logger.info('Scan Resumed')
        self._resume()

    def scanAborted(self):
        logger.info('Scan Aborted')
        self.emit(QtCore.SIGNAL("scanAborted()"))

    def scanFinished(self):
        logger.info( 'scan finished')
        self._scanData = None
        self.emit(QtCore.SIGNAL("scanFinished()"))

    def scanStarted(self):
        logger.info( 'scan started')
        self.emit(QtCore.SIGNAL("scanStarted()"))
