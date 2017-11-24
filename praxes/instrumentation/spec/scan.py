"""
"""

from __future__ import absolute_import, with_statement

import copy
#import logging
import os

import numpy as np
from PyQt4 import QtCore
from SpecClient import SpecScan, SpecCommand, SpecConnectionsManager, \
    SpecEventsDispatcher, SpecWaitObject
import h5py

from . import TEST_SPEC

import inspect
import ast
import time

from praxes.io.phynx.migration.spec import h5deadtime

#logger = logging.getLogger(__file__)


class QtSpecScanA(SpecScan.SpecScanA, QtCore.QObject):

    scanLength = QtCore.pyqtSignal(int)
    beginProcessing = QtCore.pyqtSignal()
    scanData = QtCore.pyqtSignal(int)
    scanPoint = QtCore.pyqtSignal(int)
    aborted = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()
    paused = QtCore.pyqtSignal()
    resumed = QtCore.pyqtSignal()
    started = QtCore.pyqtSignal()

    def __init__(self, specVersion, parent=None):
        QtCore.QObject.__init__(self, parent)
        SpecScan.SpecScanA.__init__(self, specVersion)

        self._scanData = None
        self._lastPoint = -1

    def __call__(self, cmd):
        if self.connection.isSpecConnected():
            self.connection.send_msg_cmd(cmd)

    def connected(self):
        pass

    def disconnected(self):
        pass

    def newScan(self, scanParameters):
#        logger.debug('newScan: %s', scanParameters)
        self._lastPoint = -1
        tree = scanParameters.pop('phynx', None)
        if tree is None:
            return

        from praxes.io.phynx import registry
        info = tree.pop('info')

        import praxes
        fileInterface = praxes.application.getService('FileInterface')
	
        specFile = info['source_file']
        h5File = fileInterface.getH5FileFromKey(specFile) 
        if h5File is None:
            return
	
	print "in spec.py, h5File is not None ..."
        # It is possible for a scan number to appear multiple times in a
        # spec file. Booo!
        with h5File:
            name = str(info['acquisition_id'])
            acq_order = ''
            i = 0
            while (name + acq_order) in h5File:
                i += 1
                acq_order = '.%d'%i
            name = name + acq_order

            # create the entry and measurement groups
            self._scanData = h5File.create_group(name, 'Entry', **info)
            measurement = self._scanData.create_measurement(**tree)
            # print "in /instrumentation/spec/scan.py, type(self._scanData): ", type(self._scanData)
	    # <class 'praxes.io.phynx.entry.Entry'>
	
	specFile = self._scanData.attrs['source_file']
	scanId = self._scanData.attrs['acquisition_id']
	self._hdf5dirName = [specFile+'_scan'+str(scanId), 'scan'+str(scanId)]
        ScanView = praxes.application.getService('ScanView')
        view = ScanView(self._scanData)

        # make sure only one view gets this signal:
        try:
            self.beginProcessing.disconnect()
        except TypeError:
            # no connections to disconnect
            pass
        if view:
            self.beginProcessing.connect(view.processData)

        self.scanLength.emit(info['npoints'])


    def newFlyScanData(self, scanData):
	# handle real-time flyscan data 
	scalarnames = scanData['scalar_names']
	scalar_names = scalarnames.split()
	scalarvalues = scanData['scalar_values']
	coln = len(scalarvalues)
	[hdf5dirName, scanNamePre] = self._hdf5dirName
	shape = ast.literal_eval(self._scanData.attrs['acquisition_shape'])
	measurementKeys = self._scanData['measurement'].keys()
	vortexes = []
	for key in measurementKeys:
	    if (key.startswith('vortex')):
		vortexes.append(key)
	monitor = self._scanData['measurement/'+vortexes[0]].attrs['monitor']
#	numRead = self._scanData['measurement/scalar_data']['i']
	if len(shape)>1:
	    nrow = (self._lastPoint + 1)/shape[1]
	else:
	    nrow = 0
	scanFileName = scanNamePre+'_'+str(nrow)+'.hdf5'
	hdf5name = os.path.join(hdf5dirName, scanFileName)
	st = os.stat(hdf5name)
	hdf5time = max(st.st_atime, st.st_ctime, st.st_mtime)
	timePassed = time.time() - hdf5time
	while (timePassed <3.0):
	    time.sleep(1)
	    st = os.stat(hdf5name)
	    hdf5time = max(st.st_atime, st.st_ctime, st.st_mtime)
	    timePassed = time.time() - hdf5time	    
	hdf5file = h5py.File(hdf5name,'r')
	measuredData = hdf5file['/entry/instrument/detector/data'].value
#	[mcapts,nvortex,chlen]=hdf5file['/entry/instrument/detector/data'].shape
	[mcapts,nvortex,chlen]=measuredData.shape
    	countSum = np.zeros((nvortex, mcapts))
	for imca in range(nvortex):
	    hdf5location = '/entry/instrument/detector/NDAttributes/CHAN'+str(imca+1)+'ROI1'
	    countSum[imca] = hdf5file[hdf5location][:]
	hdf5file.close()

	for icoln in range(mcapts):
	    newData = {}
	    for iscalar in range(len(scalar_names)):
		ttt = 'scalar_data/'+scalar_names[iscalar]
		newData[ttt] = scalarvalues[icoln][iscalar]
	    newData['scalar_data/i'] = self._lastPoint + 1
	    t_interval = newData['scalar_data/mcs0']/1.0e6
	    for ivortex in range(len(vortexes)):
		ocr = countSum[ivortex, icoln]/t_interval
		deadtime = h5deadtime(ocr)    
		ttt = vortexes[ivortex]+'/'+'counts'
		newData[ttt] = measuredData[icoln, ivortex, :] 
		ttt = vortexes[ivortex]+'/'+'dead_time'
		newData[ttt] = deadtime
		ttt = vortexes[ivortex]+'/'+monitor
		ttts = 'scalar_data/'+monitor
		newData[ttt] = newData[ttts]
		ttt = vortexes[ivortex]+'/'+'total_counts'
		newData[ttt] = countSum[ivortex, icoln]
	    self._scanData.update_measurement(**newData)
	    self._lastPoint += 1
	    if self._lastPoint == 0:
            	self.beginProcessing.emit()
	    self.scanData.emit(self._lastPoint)
	    

    def newScanData(self, scanData):
#        logger.debug( 'scanData: %s', scanData)
        if self._scanData is None:
            return
	aquisition = self._scanData.attrs['acquisition_command']
	if aquisition.startswith("fly"):
	    self.newFlyScanData(scanData)
	else:
            self._scanData.update_measurement(**scanData)
            self._lastPoint += 1
            if self._lastPoint == 0:
            	self.beginProcessing.emit()
            self.scanData.emit(self._lastPoint)

	

    def newScanPoint(self, i, x, y, scanData):
        scanData['i'] = i
        scanData['x'] = x
        scanData['y'] = y
#        logger.debug( "newScanPoint: %s", i)
        self.scanPoint.emit(i)

    def scanAborted(self):
#        logger.info('Scan Aborted')
        if self._scanData is not None:
            with self._scanData:
                self._scanData.npoints = self._lastPoint + 1
            self.aborted.emit()
            self.scanFinished()

    def scanFinished(self):
#        logger.info( 'scan finished')
        self._scanData = None
        self.finished.emit()

    def scanPaused(self):
        self.paused.emit()

    def scanResumed(self):
        self.resumed.emit()

    def scanStarted(self):
#        logger.info('scan started')
        self.started.emit()
