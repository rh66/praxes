"""
"""

from __future__ import absolute_import, with_statement

import copy
#import gc
#import logging
import posixpath
import Queue
import sys

from PyQt4 import QtCore, QtGui
from PyMca5.PyMcaGui.physics.xrf.FitParam import FitParamDialog
import numpy as np

from praxes.frontend.analysiswindow import AnalysisWindow
from .ui import resources
from .elementsview import ElementsView
from .results import XRFMapResultProxy
from praxes.io import phynx

import ast
from PyQt4.QtGui import QMessageBox

#logger = logging.getLogger(__file__)


class McaAnalysisWindow(AnalysisWindow):

    """
    """

    @property
    def n_points(self):
        return self._n_points

    # TODO: this should eventually take an MCA entry
    def __init__(self, scan_data, parent=None):
        super(McaAnalysisWindow, self).__init__(parent)

        self.analysisThread = None

        with scan_data:
            if isinstance(scan_data, phynx.Entry):
                self.scan_data = scan_data.entry.measurement
            elif isinstance(scan_data, phynx.Measurement):
                self.scan_data = scan_data
            elif isinstance(scan_data, phynx.MultiChannelAnalyzer):
                self.scan_data = scan_data
            else:
                with scan_data:
                    raise TypeError(
                        'H5 node type %s not recognized by McaAnalysisWindow'
                        % scan_data.__class__.__name__
                    )
            self._n_points = scan_data.entry.npoints
            self._dirty = False
            self._results = XRFMapResultProxy(self.scan_data)

            pymcaConfig = self.scan_data.pymca_config
            self._setupUi(resources['mcaanalysiswindow.ui'])

            title = '%s: %s: %s'%(
                posixpath.split(scan_data.file.file_name)[-1],
                posixpath.split(getattr(scan_data.entry, 'name', ''))[-1],
                posixpath.split(self.scan_data.name)[-1]
            )
            self.setWindowTitle(title)

            self.elementsView = ElementsView(self.scan_data, self)
            self.splitter.addWidget(self.elementsView)

            self.xrfBandComboBox.addItems(self.availableElements)
            try:
                self.deadTimeReport.setText(
                    str(self.scan_data.mcas.values()[0]['dead_time'].format)
                    )
            except KeyError:
                self.deadTimeReport.setText('Not found')

            self._setupMcaDockWindows()
            self._setupJobStats()

            plotOptions = self.elementsView.plotOptions
            self.optionsWidgetVLayout.insertWidget(1, plotOptions)

            self.elementsView.pickEvent.connect(self.processAverageSpectrum)
            # TODO: remove the window from the list of open windows when we close
    #           self.scanView.scanClosed.connect(self.scanClosed)


            self.fitParamDlg = FitParamDialog(parent=self)

            if pymcaConfig:
                self.fitParamDlg.setParameters(pymcaConfig)
                self.spectrumAnalysis.configure(pymcaConfig)
            else:
                self.configurePymca()

            try:
                mca = self.scan_data.entry.measurement.mcas.values()[0]
                eff = mca.monitor.efficiency
                self.monitorEfficiency.setText(str(eff))
                self.monitorEfficiency.setEnabled(True)
            except AttributeError:
                pass

            self.progressBar = QtGui.QProgressBar(self)
            self.progressBar.setMaximumHeight(17)
            self.progressBar.hide()
            self.progressBar.addAction(self.actionAbort)
            self.progressBar.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)

            self.progress_queue = Queue.Queue()
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.elementMapUpdated)

            self.elementsView.updateFigure()

            self._restoreSettings()

    @property
    def availableElements(self):
        with self.scan_data:
            try:
                return sorted(
                    self.scan_data['element_maps'].fits.keys()
                    )
            except KeyError:
                return []

    @property
    def deadTimePercent(self):
        return str(self.deadTimeComboBox.currentText())

    @property
    def mapType(self):
        return str(self.mapTypeComboBox.currentText()).lower().replace(' ', '_')

    @property
    def normalization(self):
        return str(self.normalizationComboBox.currentText())

    @property
    def xrfBand(self):
        return str(self.xrfBandComboBox.currentText())

    @property
    def pymcaConfig(self):
        return self.fitParamDlg.getParameters()

    def _setupMcaDockWindows(self):
        from .mcaspectrum import McaSpectrum
        from PyMca5.PyMcaGui.physics.xrf.ConcentrationsWidget import Concentrations

        self.concentrationsAnalysisDock = \
            self._createDockWindow('ConcentrationAnalysisDock')
        self.concentrationsAnalysis = Concentrations()
        self._setupDockWindow(
            self.concentrationsAnalysisDock,
            QtCore.Qt.BottomDockWidgetArea,
            self.concentrationsAnalysis,
            'Concentrations Analysis'
        )

        self.spectrumAnalysisDock = \
            self._createDockWindow('SpectrumAnalysisDock')
        self.spectrumAnalysis = McaSpectrum(self.concentrationsAnalysis)
        self._setupDockWindow(
            self.spectrumAnalysisDock,
            QtCore.Qt.BottomDockWidgetArea,
            self.spectrumAnalysis,
            'Spectrum Analysis'
        )

    def _setupJobStats(self):
        from praxes.dispatch.jobstats import JobStats

        self.jobStats = JobStats()
        self.jobStatsDock = self._createDockWindow('JobStatsDock')
        self._setupDockWindow(self.jobStatsDock,
                               QtCore.Qt.RightDockWidgetArea,
                               self.jobStats, 'Analysis Server Stats')

    @QtCore.pyqtSignature("bool")
    def on_actionAbort_triggered(self):
        try:
            self.analysisThread.stop()
        except AttributeError:
            pass

    @QtCore.pyqtSignature("bool")
    def on_actionAnalyzeSpectra_triggered(self):
        self.processData()

    @QtCore.pyqtSignature("bool")
    def on_actionConfigurePymca_triggered(self):
        self.configurePymca()

    @QtCore.pyqtSignature("bool")
    def on_actionCalibration_triggered(self):
        self.processAverageSpectrum()

    @QtCore.pyqtSignature("QString")
    def on_mapTypeComboBox_currentIndexChanged(self):
        self.elementsView.updateFigure(self.getElementMap())
        self.cntChannels.setChecked(False)    

    @QtCore.pyqtSignature("")
    def on_monitorEfficiency_editingFinished(self):
        with self.scan_data:
            try:
                value = float(self.monitorEfficiency.text())
                assert (0 < value <= 1)
                for mca in self.scan_data.entry.measurement.mcas.values():
                    mca.monitor.efficiency = value
            except (ValueError, AssertionError):
                mca = self.scan_data.entry.measurement.mcas.values()[0]
                self.monitorEfficiency.setText(
                    str(mca.monitor.efficiency)
                    )

    @QtCore.pyqtSignature("QString")
    def on_xrfBandComboBox_currentIndexChanged(self):
        self.elementsView.updateFigure(self.getElementMap())
        self.cntChannels.setChecked(False)    # added by RH for scattering plot
        
    @QtCore.pyqtSignature("bool")
    def on_cntChannels_toggled(self):
        t = str(self.ChannelsEdit.text()).split(",")
        if (len(t)==2 and self.cntChannels.isChecked()):
            try:
                ch1=int(t[0])
                ch2=int(t[1])
            except:
                print "Channels inputs should be: number1, number2 (number2>number1)"
                
            if ch2>ch1:
                self.elementsView.updateFigure(self.channelMap(ch1, ch2))   
                
                
    def channelMap(self, ch1, ch2):
        channelmap = np.zeros(self.scan_data.entry.acquisition_shape, dtype='f')
        for measurement_key in self.scan_data.keys():  # go through 'mca1', mca2', etc
            if 'counts' in self.scan_data[measurement_key].keys():
	        liveTime = 1.0-self.scan_data[measurement_key]['dead_time'][:]/100.0
                channelmap.flat[:len(liveTime)] = ( channelmap.flat[:len(liveTime)] + 
                        np.sum(self.scan_data[measurement_key]['counts'][:len(liveTime),ch1:ch2], axis=1)/liveTime)
        return channelmap
        
        
        

    def closeEvent(self, event):
        if self.analysisThread:
            self.showNormal()
            self.raise_()
            warning = '''Data analysis is not yet complete.
            Are you sure you want to close?'''
            res = QtGui.QMessageBox.question(self, 'closing...', warning,
                                             QtGui.QMessageBox.Yes,
                                             QtGui.QMessageBox.No)
            if res == QtGui.QMessageBox.Yes:
                if self.analysisThread:
                    self.analysisThread.stop()
                    self.analysisThread.wait()
                    #QtGui.qApp.processEvents()
            else:
                event.ignore()
                return

        AnalysisWindow.closeEvent(self, event)
        event.accept()

    def configurePymca(self):
        if self.fitParamDlg.exec_():
            #QtGui.qApp.processEvents()
            self.statusbar.showMessage('Reconfiguring PyMca ...')
            configDict = self.fitParamDlg.getParameters()
            self.spectrumAnalysis.configure(configDict)
            with self.scan_data:
                self.scan_data.pymca_config = configDict
            self.statusbar.clearMessage()

    def elementMapUpdated(self):
        if self._dirty:
            self._dirty = False
            self.elementsView.updateFigure(self.getElementMap())
        #QtGui.qApp.processEvents()

    def getElementMap(self, mapType=None, element=None):
        if element is None: element = self.xrfBand
        if mapType is None: mapType = self.mapType

        if mapType and element:
            try:
		temp=self._results.get(element, mapType)
		return temp
#                return self._results.get(element, mapType)
            except KeyError:
                return np.zeros(self.scan_data.entry.acquisition_shape)
        else:
            with self.scan_data:
                return np.zeros(
                    self.scan_data.entry.acquisition_shape, dtype='f'
                    )

    def initializeElementMaps(self, elements):
        self._results = XRFMapResultProxy(self.scan_data, elements)

    def processAverageSpectrum(self, indices=None):
        with self.scan_data:
            if indices is None:
                indices = np.arange(self.scan_data.entry.acquired)
            n_indices = len(indices)
            [n_indices2, slen] = self.scan_data['vortex1/counts'].shape
            n_indices = np.min(np.array([n_indices, n_indices2]))
            # indices = np.array(range(n_indices))
            if n_indices:
                masked = self.scan_data.entry.measurement.masked[...][indices]
                indices = indices[np.logical_not(masked)]
                n_indices = len(indices)

            if not n_indices:
                return

            self.statusbar.showMessage('Averaging spectra ...')

            try:
                # looking at individual element
                monitor = self.scan_data.monitor.corrected_value
                mon0 = monitor[indices[0]]
                channels = self.scan_data.channels
                counts = channels.astype('float32') * 0
                dataset = self.scan_data['counts'].corrected_value
                for i in indices:
                    counts += dataset[i] / n_indices / (monitor[i]/mon0)
            except AttributeError:
                # looking at multiple elements
                mcas = self.scan_data.mcas.values()
                monitor = mcas[0].monitor.corrected_value
                mon0 = monitor[indices[0]]
                channels = mcas[0].channels
                counts = channels.astype('float32') * 0
                for mca in mcas:
                    dataset = mca['counts'].corrected_value
                    for i in indices:
                        counts += dataset[i] / n_indices / (monitor[i]/mon0)

            self.spectrumAnalysis.setData(x=channels, y=counts)
            self.statusbar.showMessage('Performing Fit ...')
            self.spectrumAnalysis.mcafit.config['concentrations']['flux'] = mon0
            self.spectrumAnalysis.mcafit.config['concentrations']['time'] = 1
            fitresult = self.spectrumAnalysis.fit()

            self.fitParamDlg.setFitResult(fitresult['result'])

            self.statusbar.clearMessage()

        self.setMenuToolsActionsEnabled(True)

    def processComplete(self):
        self.progressBar.hide()
        self.progressBar.reset()
        self.statusbar.removeWidget(self.progressBar)
        self.statusbar.clearMessage()
        self.timer.stop()

        self.analysisThread = None

        self.setMenuToolsActionsEnabled(True)
        self.elementMapUpdated()
        self._results.flush()


    def processData(self):
        from .mptaskmanager import XfsTaskManager

        self.setMenuToolsActionsEnabled(False)

        self._resetPeaks()

        settings = QtCore.QSettings()
        settings.beginGroup('JobServers')
        n_local_processes, ok = settings.value(
            'LocalProcesses', QtCore.QVariant(1)
            ).toInt()

        thread = XfsTaskManager(
            self.scan_data,
            copy.deepcopy(self.pymcaConfig),
            self._results,
            n_local_processes=n_local_processes
            )

        thread.progress_report.connect(self.update)
        thread.finished.connect(self.processComplete)
        self.actionAbort.triggered.connect(thread.stop) #thread.stop

        self.statusbar.showMessage('Analyzing spectra ...')
        self.statusbar.addPermanentWidget(self.progressBar)
        self.progressBar.show()

        self.analysisThread = thread

        thread.start()
        self.timer.start(1000)

    def update(self, report):
        self._dirty = True

        n_processed = report.pop('n_processed')
        with self.scan_data:
            n_points = self.scan_data.entry.npoints
        progress = int((100.0 * n_processed) / n_points)
        self.progressBar.setValue(progress)
#        self.jobStats.updateTable(item)

    def _resetPeaks(self):
        peaks = []

        for el, edges in self.pymcaConfig['peaks'].items():
            for edge in edges:
                name = '_'.join([el, edge])
                peaks.append(name)

        peaks.sort()
        self.initializeElementMaps(peaks)

        self.xrfBandComboBox.clear()
        self.xrfBandComboBox.addItems(peaks)

    def setMenuToolsActionsEnabled(self, enabled=True):
        self.actionAnalyzeSpectra.setEnabled(enabled)
        self.actionConfigurePymca.setEnabled(enabled)
        self.actionCalibration.setEnabled(enabled)
        self.actionEnhance_Spectra.setEnabled(enabled)

    @QtCore.pyqtSignature("bool")
    def on_actionEnhance_Spectra_triggered(self):        
        qwt = QtGui.QWidget()
        reply = QMessageBox.question(qwt, 'Message', 'Enhance spectrum',QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            self.enhance_save()        
        
    def enhance_save(self):
        scanshape = ast.literal_eval(self.scan_data.entry.attrs['acquisition_shape'])
        cfgstr = self.scan_data.entry.measurement.attrs['pymca_config']
        oldh5_name = self.scan_data.file.attrs['file']
        scan_id = self.scan_data.entry
        old_name = '.'.join(oldh5_name.split(".")[:-1]) 
        if old_name[-1] == 'E':
            newh5_name = old_name + 'E.h5'
        else:
            newh5_name = old_name + '_' + str(scan_id.name) + '_E.h5'
        oldh5 = self.scan_data.file
        from praxes.io.phynx import open
        newh5 = open(newh5_name, 'a')
        newh5.attrs['file'] = newh5_name
        self.h5transfer(oldh5, scan_id, newh5)
        cfg = ast.literal_eval(cfgstr)  
        chmax = cfg['fit']['xmax']+10  
        mcas = self.scan_data.mcas.values()
        mcounts = [mca['counts'].corrected_value for mca in mcas]
        monitorName = self.scan_data.entry.measurement['scalar_data'].attrs['monitor']
        monitors = mcas[0][str(monitorName)][:]
        ctshape = mcounts[0].shape
        chmax = min(chmax, ctshape[1])
        mcapts = ctshape[0]
        colnum = scanshape[1]
        rownum = int(mcapts/colnum)
        counts2 = np.zeros((rownum, colnum, chmax), dtype=float)
        monitor2 = np.zeros((rownum, colnum))
        for mca_pts in range(mcapts):
            irow = int(mca_pts/colnum)
            icol = mca_pts - irow*colnum
            cts = [counts[mca_pts] for counts in mcounts]
            ctsum = np.sum(cts,0)
            counts2[irow, icol,:] = ctsum[:chmax]  #np.sum(cts, 0)[0:chmax]
            monitor2[irow, icol] = monitors[mca_pts]
        if irow>2:
            for i_row in range(irow-1):
                #print 'i_row+2: ', i_row+2
                counts2[i_row+1,:,:] = (0.5*counts2[i_row+1,:,:]+
                       0.25*counts2[i_row,:,:]+0.25*counts2[i_row+2,:,:])
                monitor2[i_row+1,:] = 0.5*monitor2[i_row+1,:]+0.25*monitor2[i_row,:]+0.25*monitor2[i_row+2,:]
        if icol>2:
            for i_col in range(icol-1):
                counts2[:,i_col+1,:] = 0.5*counts2[:,i_col+1,:]+0.25*counts2[:,i_col,:]+0.25*counts2[:,i_col+2,:]
                monitor2[:,i_col+1] = 0.5*monitor2[:,i_col+1]+0.25*monitor2[:,i_col]+0.25*monitor2[:,i_col+2] 
        
        newh5[str(scan_id.name)].measurement['vortex1']['counts'][0:mcapts,0:chmax] = counts2.reshape(-1,counts2.shape[-1])
        newh5[str(scan_id.name)].measurement['vortex1/'+monitorName][0:mcapts]=monitor2.ravel()
        newh5[str(scan_id.name)].measurement['vortex1/dead_time'][0:mcapts]=0.0
        newh5.close()
        

    def h5transfer(self, oldh5, scan_id, newh5):
        mcas = oldh5[str(scan_id.name)].measurement.mcas.values()
        if len(mcas)>1:
            attrs = oldh5[str(scan_id.name)].attrs
            entry = newh5.create_group(scan_id.name, type='Entry', **attrs)
            attrs = oldh5[str(scan_id.name)].measurement.attrs
            measurement = entry.create_group('measurement', type='Measurement', **attrs)
            for key, value in oldh5[str(scan_id.name)].measurement.items():
                if str(key).startswith('vortex'):
                    pass
                else:
                    dest = '/'+str(scan_id.name)+'/measurement'
                    src = dest + '/'+key
                    oldh5.copy(src, newh5[dest])
            src = dest + '/' + 'vortex1'
            oldh5.copy(src, newh5[dest])
        else:
            oldh5.copy(str(scan_id.name), newh5)
                
        


    @classmethod
    def offersService(cls, h5Node):
        if isinstance(
            h5Node, (phynx.Entry, phynx.Measurement, phynx.MultiChannelAnalyzer)
            ):
            return len(h5Node.entry.measurement.mcas) > 0
        return False


#if __name__ == "__main__":
#    import sys
#    app = QtGui.QApplication(sys.argv)
#    app.setOrganizationName('Praxes')
#    form = McaAnalysisWindow()
#    form.show()
#    sys.exit(app.exec_())
