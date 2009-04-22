"""
"""
from __future__ import with_statement

import logging
import time

import numpy as np
import pp
from PyMca import ClassMcaTheory
from PyMca.ConcentrationsTool import ConcentrationsTool
from PyQt4 import QtCore
import numpy as np
np.seterr(all='ignore')

from xpaxs.dispatch.pptaskmanager import PPTaskManager


logger = logging.getLogger(__file__)
DEBUG = False

def analyzeSpectrum(index, spectrum, tconf, advancedFit, mfTool):
    start = time.time()
    advancedFit.config['fit']['use_limit'] = 1
    # TODO: get the channels from the controller
    advancedFit.setdata(y=spectrum)
    advancedFit.estimate()
    estimate = time.time()
    if ('concentrations' in advancedFit.config) and \
            (advancedFit._fluoRates is None):
        fitresult, result = advancedFit.startfit(digest=1)
    else:
        fitresult = advancedFit.startfit(digest=0)
        result = advancedFit.imagingDigestResult()
    result['index'] = index
    fit = time.time()

    if mfTool:
        temp = {}
        temp['fitresult'] = fitresult
        temp['result'] = result
        temp['result']['config'] = advancedFit.config
        tconf.update(advancedFit.configure()['concentrations'])
        conc = mfTool.processFitResult(config=tconf, fitresult=temp,
                                       elementsfrommatrix=False,
                                       fluorates=advancedFit._fluoRates)
        result['concentrations'] = conc
    fitconc = time.time()
    report = {'estimate':estimate-start,
              'fit': fit-estimate,
              'fitconc': fitconc-fit}

    return {
        'index': index,
        'result': result,
        'advancedFit': advancedFit,
        'report': report
    }


class XfsPPTaskManager(PPTaskManager):

    def __init__(self, scan, enumerator, config, parent=None):
        super(XfsPPTaskManager, self).__init__(scan, enumerator, parent)

        with self.lock:
            self._config = config
            try:
                self._intensity = config['concentrations']['flux']
            except IndexError:
                self._intensity = None

            self._advancedFit = ClassMcaTheory.McaTheory(config=config)
            self._advancedFit.enableOptimizedLinearFit()
            self._mfTool = None
            if 'concentrations' in config:
                self._mfTool = ConcentrationsTool(config)
                self._tconf = self._mfTool.configure()

    def submitJob(self, index, data):
#        print 'submitting job', index
        with self.lock:
            args = (
                index, data, self._tconf, self._advancedFit,
                self._mfTool
            )
            self._jobServer.submit(
                analyzeSpectrum,
                args,
                modules=("time", ),
                callback=self.queueResults
            )
#        print 'job %d submitted' % index

    def updateElementMap(self, element, mapType, index, val):
#        print 'updating', element, mapType, index
        with self._scan.plock:
#            print 'updateElementMap acquired the lock'
            try:
                entry = '%s_%s'%(element, mapType)
                print 'updating element map for', entry
                self._scan['element_maps'][entry][index] = val
#                self._scan['element_maps'][entry][index] = np.random.rand(1)
                print entry, 'updated'
            except ValueError:
                print "index %d out of range for %s", index, entry
            except H5Error:
                print "%s not found in element_maps", entry
#        print element, mapType, index, 'updated'

    def updateRecords(self):
#        print 'Updating records'
        with self.lock:
            while self._results:
                data = self._results.pop(0)
#                print 'I have data'
                # this lock shouldn't be necessary
                with self._scan.plock:
#                    print 'I acquired the data lock'
                    index = data['index']

                    with self.lock:
#                        print 'I acquired the thread lock'
                        self._advancedFit = data['advancedFit']
                        self._totalProcessed += 1

                    result = data['result']
                    for group in result['groups']:
                        g = group.replace(' ', '_')

                        fitArea = result[group]['fitarea']
                        if fitArea: sigmaArea = result[group]['sigmaarea']/fitArea
                        else: sigmaArea = np.nan

                        self.updateElementMap(g, 'fit', index, fitArea)
                        self.updateElementMap(g, 'fit_error', index, sigmaArea)

                    if 'concentrations' in result:
                        massFractions = result['concentrations']['mass fraction']
                        for key, val in massFractions.iteritems():
                            k = key.replace(' ', '_')
                            self.updateElementMap(k, 'mass_fraction', index, val)
                    self.dirty = True
#        print 'records updated'
