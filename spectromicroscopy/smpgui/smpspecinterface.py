"""
"""

#---------------------------------------------------------------------------
# Stdlib imports
#---------------------------------------------------------------------------

import sys
import time
import weakref

#---------------------------------------------------------------------------
# Extlib imports
#---------------------------------------------------------------------------

from PyQt4 import QtCore, QtGui

#---------------------------------------------------------------------------
# SMP imports
#---------------------------------------------------------------------------

from spectromicroscopy import smpConfig
from spectromicroscopy import configutils
from spectromicroscopy.smpgui.ui_smpspecinterface import Ui_SmpSpecInterface
from spectromicroscopy.smpgui import configuresmp, pymcafitparams, \
    scananalysis, scancontrols
from spectromicroscopy.smpcore import specrunner, qtspecscan, qtspecvariable
from SpecClient import SpecClientError

#---------------------------------------------------------------------------
# Normal code begins
#---------------------------------------------------------------------------


class SmpSpecInterface(Ui_SmpSpecInterface, QtGui.QWidget):
    """Establishes a Experimenbt controls 
    Generates Control and Feedback instances
   Addes Scan atributes to specRunner instance 
    """
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.setupUi(self)
        
        self.parent = parent
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        
        self.connectToSpec()
        
        pymcaConfigFile = configutils.getDefaultPymcaConfigFile()
        self.pymcaConfig = configutils.getPymcaConfig(pymcaConfigFile)
        self.pymcaConfigWidget = pymcafitparams.PyMcaFitParams(self)
        self.pymcaConfigWidget.setParameters(self.pymcaConfig)
        self.tabWidget.insertTab(1, self.pymcaConfigWidget,
                                 'PyMca Configuration')

        self.scanControls = scancontrols.ScanControls(self)
        self.gridlayout.addWidget(self.scanControls, 0,0)
        self.gridlayout.addWidget(self.tabWidget, 0,1,1,1)

        self.getDefaults()
        self.configureSkipmode()
        
        self.connectSignals()

    def connectSignals(self):
        self.connect(self.specRunner.scan, 
                     QtCore.SIGNAL("newMesh(PyQt_PyObject)"),
                     self.newScanAnalysis2D)
        self.connect(self.specRunner.scan,
                     QtCore.SIGNAL("newTseries(PyQt_PyObject)"),
                     self.newScanAnalysis1D)
        self.connect(self.specRunner.scan,
                     QtCore.SIGNAL("newAscan(PyQt_PyObject)"),
                     self.newScanAnalysis1D)
        self.connect(self.specRunner.scan,
                     QtCore.SIGNAL("newA2scan(PyQt_PyObject)"),
                     self.newScanAnalysis1D)
        self.connect(self.specRunner.scan,
                     QtCore.SIGNAL("newA3scan(PyQt_PyObject)"),
                     self.newScanAnalysis1D)
        self.connect(self.specRunner.scan,
                     QtCore.SIGNAL("newScan(PyQt_PyObject)"),
                     self.setTabLabel)
        self.connect(self.pymcaConfigWidget,
                     QtCore.SIGNAL("configChanged(PyQt_PyObject)"),
                     self.changedPyMcaConfig)
        self.connect(self.deadtimeCorrCheckBox,
                     QtCore.SIGNAL("clicked(bool)"),
                     self.deadtimeCorrEnabled)
        self.connect(self.skipmodeCheckBox,
                     QtCore.SIGNAL("clicked(bool)"),
                     self.skipmodeEnabled)
        self.connect(self.skipmodeCounterComboBox,
                     QtCore.SIGNAL("currentIndexChanged(QString)"),
                     self.skipmodeCounterChanged)
        self.connect(self.skipmodeThreshSpinBox,
                     QtCore.SIGNAL("valueChanged(double)"),
                     self.skipmodeThresholdChanged)
        self.connect(self.skipmodePrecountSpinBox,
                     QtCore.SIGNAL("valueChanged(double)"),
                     self.skipmodePrecountChanged)
        self.connect(self.specRunner.scan, 
                     QtCore.SIGNAL("scanStarted()"),
                     self.disableScanOptions)
        self.connect(self.specRunner.scan, 
                     QtCore.SIGNAL("scanFinished()"),
                     self.enableScanOptions)
        self.connect(self.specRunner.scan, 
                     QtCore.SIGNAL("scanAborted()"),
                     self.enableScanOptions)

    def changedPyMcaConfig(self):
        self.pymcaConfig = self.pymcaConfigWidget.getParameters()
        self.emit(QtCore.SIGNAL("changedPyMcaConfig(PyQt_PyObject)"),
                  self.pymcaConfig)

    def closeEvent(self, event):
        self.specRunner.skipmode(0)
        self.specRunner.close()
        event.accept()

    def connectToSpec(self):
        specVersion = self.getSpecVersion()
        try:
            self.window().statusBar.showMessage('Connecting')
            QtGui.qApp.processEvents()
            self.specRunner = specrunner.SpecRunner(specVersion, timeout=500)
            self.specRunner.scan = \
                qtspecscan.QtSpecScanA(self.specRunner.specVersion)
            self.specRunner.runMacro('smp_mca.mac')
            self.window().statusBar.clearMessage()
        except SpecClientError.SpecClientTimeoutError:
            self.connectionError(specVersion)
            raise SpecClientError.SpecClientTimeoutError

    def connectionError(self, specVersion):
        error = QtGui.QErrorMessage()
        server, port = specVersion.split(':')
        error.showMessage('''\
        SMP was unabel to connect to the "%s" spec instance at "%s". Please \
        make sure you have started spec in server mode (for example "spec \
        -S").'''%(port, server))
        error.exec_()

    def deadtimeCorrEnabled(self, enabled):
        smpConfig.deadtimeCorrection.enabled = enabled

    def disableScanOptions(self):
        self.tabWidget.setEnabled(False)

    def enableScanOptions(self):
        self.tabWidget.setEnabled(True)

    def configureSkipmode(self):
        enabled = 0
        if smpConfig.skipmode.enabled: enabled = 1
        self.specRunner.skipmode(enabled,
                                 smpConfig.skipmode.precount,
                                 str(smpConfig.skipmode.counter),
                                 smpConfig.skipmode.threshold)

    def skipmodeEnabled(self, enabled):
        smpConfig.skipmode.enabled = enabled
        self.configureSkipmode()

    def skipmodeCounterChanged(self, counter):
        smpConfig.skipmode.counter = '%s'%counter
        self.configureSkipmode()

    def skipmodeThresholdChanged(self, threshold):
         smpConfig.skipmode.threshold = threshold
         self.configureSkipmode()

    def skipmodePrecountChanged(self, precount):
        smpConfig.skipmode.precount = precount
        self.configureSkipmode()

    def getDefaults(self):
        self.deadtimeCorrCheckBox.setChecked(smpConfig.deadtimeCorrection.enabled)
        self.skipmodeCheckBox.setChecked(smpConfig.skipmode.enabled)
        self.skipmodeThreshSpinBox.setValue(smpConfig.skipmode.threshold)
        self.skipmodePrecountSpinBox.setValue(smpConfig.skipmode.precount)
        
        counters = self.specRunner.getCountersMne()
        self.skipmodeCounterComboBox.addItems(counters)
        try:
            ind = counters.index(smpConfig.skipmode.counter)
            self.skipmodeCounterComboBox.setCurrentIndex(ind)
        except ValueError:
            smpConfig.skipmode.counter = counters[0]

    def getSpecVersion(self):
        return ':'.join([smpConfig.session.server,
                         smpConfig.session.port])

    def newScanAnalysis(self, newAnalysis):
        self.parent.mainTab.addTab(newAnalysis, '')
        self.parent.mainTab.setCurrentWidget(newAnalysis)

    def newScanAnalysis1D(self, scanParams):
        self.newScanAnalysis(scananalysis.ScanAnalysis1D(self, scanParams))

    def newScanAnalysis2D(self, scanParams):
        self.newScanAnalysis(scananalysis.ScanAnalysis2D(self, scanParams))

    def setTabLabel(self, scanParams):
        temp = scanParams['datafile']
        temp = temp.rstrip('.dat').rstrip('.txt').rstrip('.mca')
        label = '_'.join([temp, scanParams['title'].replace(' ', '')])
        i = self.parent.mainTab.currentIndex()
        self.parent.mainTab.setTabText(i, label)


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    myapp = SmpProjectInterface()
    myapp.show()
    sys.exit(app.exec_())
