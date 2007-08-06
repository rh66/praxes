#!/usr/bin/env python
import codecs
import os
import sys
os.system("pyuic4 XpMaster.ui>XpMaster.py")
DEBUG=2

#GUI
from PyQt4 import QtCore, QtGui
from XpMaster import Ui_XpMaster
from SpecRunner import SpecRunner
#Number Crunching
import numpy as np
import MA
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as ToolBar
from matplotlib.figure import Figure
import time
from tempfile import TemporaryFile, NamedTemporaryFile
from PyMca import ClassMcaTheory , ConcentrationsTool 
from external.configobj import ConfigObj

ElementsInfo = [
   ["H",   1,    1,1,   "hydrogen",   1.00800,     1008.00   ],
   ["He",  2,   18,1,   "helium",     4.00300,     0.118500  ],
   ["Li",  3,    1,2,   "lithium",    6.94000,     534.000   ],
   ["Be",  4,    2,2,   "beryllium",  9.01200,     1848.00   ],
   ["B",   5,   13,2,   "boron",      10.8110,     2340.00   ],
   ["C",   6,   14,2,   "carbon",     12.0100,     1580.00   ],
   ["N",   7,   15,2,   "nitrogen",   14.0080,     1.25      ],
   ["O",   8,   16,2,   "oxygen",     16.0000,     1.429     ],
   ["F",   9,   17,2,   "fluorine",   19.0000,     1108.00   ],
   ["Ne",  10,  18,2,   "neon",       20.1830,     0.9       ],
   ["Na",  11,   1,3,   "sodium",     22.9970,     970.000   ],
   ["Mg",  12,   2,3,   "magnesium",  24.3200,     1740.00   ],
   ["Al",  13,  13,3,   "aluminium",  26.9700,     2720.00   ],
   ["Si",  14,  14,3,   "silicon",    28.0860,     2330.00   ],
   ["P",   15,  15,3,   "phosphorus", 30.9750,     1820.00   ],
   ["S",   16,  16,3,   "sulphur",    32.0660,     2000.00   ],
   ["Cl",  17,  17,3,   "chlorine",   35.4570,     1560.00   ],
   ["Ar",  18,  18,3,   "argon",      39.9440,     1.78400   ],
   ["K",   19,   1,4,   "potassium",  39.1020,     862.000   ],
   ["Ca",  20,   2,4,   "calcium",    40.0800,     1550.00   ],
   ["Sc",  21,   3,4,   "scandium",   44.9600,     2992.00   ],
   ["Ti",  22,   4,4,   "titanium",   47.9000,     4540.00   ],
   ["V",   23,   5,4,   "vanadium",   50.9420,     6110.00   ],
   ["Cr",  24,   6,4,   "chromium",   51.9960,     7190.00   ],
   ["Mn",  25,   7,4,   "manganese",  54.9400,     7420.00   ],
   ["Fe",  26,   8,4,   "iron",       55.8500,     7860.00   ],
   ["Co",  27,   9,4,   "cobalt",     58.9330,     8900.00   ],
   ["Ni",  28,  10,4,   "nickel",     58.6900,     8900.00   ],
   ["Cu",  29,  11,4,   "copper",     63.5400,     8940.00   ],
   ["Zn",  30,  12,4,   "zinc",       65.3800,     7140.00   ],
   ["Ga",  31,  13,4,   "gallium",    69.7200,     5903.00   ],
   ["Ge",  32,  14,4,   "germanium",  72.5900,     5323.00   ],
   ["As",  33,  15,4,   "arsenic",    74.9200,     5.73000   ],
   ["Se",  34,  16,4,   "selenium",   78.9600,     4790.00   ],
   ["Br",  35,  17,4,   "bromine",    79.9200,     3120.00   ],
   ["Kr",  36,  18,4,   "krypton",    83.8000,     3.74000   ],
   ["Rb",  37,   1,5,   "rubidium",   85.4800,     1532.00   ],
   ["Sr",  38,   2,5,   "strontium",  87.6200,     2540.00   ],
   ["Y",   39,   3,5,   "yttrium",    88.9050,     4405.00   ],
   ["Zr",  40,   4,5,   "zirconium",  91.2200,     6530.00   ],
   ["Nb",  41,   5,5,   "niobium",    92.9060,     8570.00   ],
   ["Mo",  42,   6,5,   "molybdenum", 95.9500,     10220.00  ],
   ["Tc",  43,   7,5,   "technetium", 99.0000,     11500.0   ],
   ["Ru",  44,   8,5,   "ruthenium",  101.0700,    12410.0   ],
   ["Rh",  45,   9,5,   "rhodium",    102.9100,    12440.0    ],
   ["Pd",  46,  10,5,   "palladium",  106.400,     12160.0   ],
   ["Ag",  47,  11,5,   "silver",     107.880,     10500.00  ],
   ["Cd",  48,  12,5,   "cadmium",    112.410,     8650.00   ],
   ["In",  49,  13,5,   "indium",     114.820,     7280.00   ],
   ["Sn",  50,  14,5,   "tin",        118.690,     5310.00   ],
   ["Sb",  51,  15,5,   "antimony",   121.760,     6691.00   ],
   ["Te",  52,  16,5,   "tellurium",  127.600,     6240.00   ],
   ["I",   53,  17,5,   "iodine",     126.910,     4940.00   ],
   ["Xe",  54,  18,5,   "xenon",      131.300,     5.90000   ],
   ["Cs",  55,   1,6,   "caesium",    132.910,     1873.00   ],
   ["Ba",  56,   2,6,   "barium",     137.360,     3500.00   ],
   ["La",  57,   3,6,   "lanthanum",  138.920,     6150.00   ],
   ["Ce",  58,   4,9,   "cerium",     140.130,     6670.00   ],
   ["Pr",  59,   5,9,   "praseodymium",140.920,    6769.00   ],
   ["Nd",  60,   6,9,   "neodymium",  144.270,     6960.00   ],
   ["Pm",  61,   7,9,   "promethium", 147.000,     6782.00   ],
   ["Sm",  62,   8,9,   "samarium",   150.350,     7536.00   ],
   ["Eu",  63,   9,9,   "europium",   152.000,     5259.00   ],
   ["Gd",  64,  10,9,   "gadolinium", 157.260,     7950.00   ],
   ["Tb",  65,  11,9,   "terbium",    158.930,     8272.00   ],
   ["Dy",  66,  12,9,   "dysprosium", 162.510,     8536.00   ],
   ["Ho",  67,  13,9,   "holmium",    164.940,     8803.00   ],
   ["Er",  68,  14,9,   "erbium",     167.270,     9051.00   ],
   ["Tm",  69,  15,9,   "thulium",    168.940,     9332.00   ],
   ["Yb",  70,  16,9,   "ytterbium",  173.040,     6977.00   ],
   ["Lu",  71,  17,9,   "lutetium",   174.990,     9842.00   ],
   ["Hf",  72,   4,6,   "hafnium",    178.500,     13300.0   ],
   ["Ta",  73,   5,6,   "tantalum",   180.950,     16600.0   ],
   ["W",   74,   6,6,   "tungsten",   183.920,     19300.0   ],
   ["Re",  75,   7,6,   "rhenium",    186.200,     21020.0   ],
   ["Os",  76,   8,6,   "osmium",     190.200,     22500.0   ],
   ["Ir",  77,   9,6,   "iridium",    192.200,     22420.0   ],
   ["Pt",  78,  10,6,   "platinum",   195.090,     21370.0   ],
   ["Au",  79,  11,6,   "gold",       197.200,     19370.0   ],
   ["Hg",  80,  12,6,   "mercury",    200.610,     13546.0   ],
   ["Tl",  81,  13,6,   "thallium",   204.390,     11860.0   ],
   ["Pb",  82,  14,6,   "lead",       207.210,     11340.0   ],
   ["Bi",  83,  15,6,   "bismuth",    209.000,     9800.00   ],
   ["Po",  84,  16,6,   "polonium",   209.000,     0         ],
   ["At",  85,  17,6,   "astatine",   210.000,     0         ],
   ["Rn",  86,  18,6,   "radon",      222.000,     9.73000   ],
   ["Fr",  87,   1,7,   "francium",   223.000,     0         ],
   ["Ra",  88,   2,7,   "radium",     226.000,     0         ],
   ["Ac",  89,   3,7,   "actinium",   227.000,     0         ],
   ["Th",  90,   4,10,  "thorium",    232.000,     11700.0   ],
   ["Pa",  91,   5,10,  "proactinium",231.03588,   0         ],
   ["U",   92,   6,10,  "uranium",    238.070,     19050.0   ],
   ["Np",  93,   7,10,  "neptunium",  237.000,     0         ],
   ["Pu",  94,   8,10,  "plutonium",  239.100,     19700.0   ],
   ["Am",  95,   9,10,  "americium",  243,         0         ],
   ["Cm",  96,  10,10,  "curium",     247,         0         ],
   ["Bk",  97,  11,10,  "berkelium",  247,         0         ],
   ["Cf",  98,  12,10,  "californium",251,         0         ],
   ["Es",  99,  13,10,  "einsteinium",252,         0         ],
   ["Fm",  100,  14,10, "fermium",    257,         0         ],
   ["Md",  101,  15,10, "mendelevium",258,         0         ],
   ["No",  102,  16,10, "nobelium",   259,         0         ],
   ["Lr",  103,  17,10, "lawrencium", 262,         0         ],
   ["Rf",  104,   4,7,  "rutherfordium",261,       0         ],
   ["Db",  105,   5,7,  "dubnium",    262,         0         ],
   ["Sg",  106,   6,7,  "seaborgium", 266,         0         ],
   ["Bh",  107,   7,7,  "bohrium",    264,         0         ],
   ["Hs",  108,   8,7,  "hassium",    269,         0         ],
   ["Mt",  109,   9,7,  "meitnerium", 268,         0         ],]
ElementList= [ elt[0] for elt in ElementsInfo ]
Shells = ['K','L1','L2','L3',
                 'M1','M2','M3','M4','M5',
                 'N1','N2','N3','N4','N5','N6','N7',
                 'O1','O2','O3','O4','O5','O6','O7',
                 'P1','P2','P3','P4','P5']
VisualTypes=["imshow","line plot"]
Scans=["select","tseries","mesh"]

class MyXP(Ui_XpMaster,QtGui.QMainWindow):
    """Establishes a Experimenbt controls    """
    def __init__(self,parent=None):
        self.DEBUG=DEBUG
        QtGui.QWidget.__init__(self, parent)
        self.parent=parent
        self.setupUi(self)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        self.config()
        self.setup=0
        if parent:
            Bar=parent.Bar
        else:
            Bar=self.Bar
        Bar.addAction("PyMca Config file",self.set_config_file)
        Bar.addAction("Configure SMP",self.config_smp)
        for item in Scans:
            self.ScanBox.addItem(item)
        self.xprun=SpecRunner(self,DEBUG,self.__server,self.__port)
        self.xprun.exc("NPTS=0")
        self.buffer=TemporaryFile( 'w+b')
        self.ElementSelect.setLineEdit(self.ElementText)
        QtCore.QObject.connect(self.ScanBox,QtCore.SIGNAL("currentIndexChanged(int)"),self.ScanControls)
        QtCore.QObject.connect(self.Run,QtCore.SIGNAL("clicked()"),self.run_scan)
        QtCore.QObject.connect(self.ElementSelect,QtCore.SIGNAL("currentIndexChanged(int)"),self.set_element)
        QtCore.QObject.connect(self.ScaleBox,QtCore.SIGNAL("currentIndexChanged(int)"),self.change_limits)#self.set_scale)
        QtCore.QObject.connect(self.MinValSpin,QtCore.SIGNAL("editingFinished()"),self.change_limits)
        QtCore.QObject.connect(self.MaxValSpin,QtCore.SIGNAL("editingFinished()"),self.change_limits)
        QtCore.QObject.connect(self.Estop,QtCore.SIGNAL("clicked()"),self.emergancy_stop)
        QtCore.QObject.connect(self.Mover,QtCore.SIGNAL("clicked()"),self.Move)
        QtCore.QObject.connect(self.AutoRanger,QtCore.SIGNAL("clicked()"),self.auto_set)
        for peak in self.__peaks:
            self.ElementSelect.addItem(peak)
        self.Spin_Slide_Motor=[]
        self.Image_Element=''

    def ScanControls(self):
        """establishes the Motor controls for selected scan"""
        if self.Spin_Slide_Motor:
            for object in self.Spin_Slide_Motor:
                object.hide()
        self.Spin_Slide_Motor=[]
        if self.ScanBox.currentText()==Scans[1]:
            self.X= Spinner_Slide_Motor(self.MotorBox,self.xprun,"X",0)
            self.X.show()
            self.Spin_Slide_Motor.append(self.X)
        elif self.ScanBox.currentText()==Scans[2]:
            self.X= Spinner_Slide_Motor(self.MotorBox,self.xprun,"X",0)
            self.Y= Spinner_Slide_Motor(self.MotorBox,self.xprun,"Z",1)
            self.X.show()
            self.Y.show()
            self.Spin_Slide_Motor.append(self.X)
            self.Spin_Slide_Motor.append(self.Y)
    
    def Move(self):
        for i in len(self.Spin_Slide_Motor):
            self.Spin_Slide_Motor[i].Move()
        
    def emergancy_stop(self):
        self.xprun.EmergencyStop() 
    
    def config_smp(self):
        print self.__server,self.__port
        editor=Customizer(self,self.configfile)
        editor.exec_()
        config=ConfigObj(self.configfile)
        self.__server=config["setup"]["server"]
        self.__port=config["setup"]["port"]
        print "***",self.__server,self.__port
    
    def set_config_file(self):
        try:
            fd = QtGui.QFileDialog(self)
            self.filename = "%s"%fd.getOpenFileName()
            config=ConfigObj(self.filename)
            self.__peaks=config["peaks"]
            self.ElementSelect.clear()
            for peak in self.__peaks:
                self.ElementSelect.addItem(peak)
        except:
            print "come on now"

    
    def config(self):
        user=os.path.expanduser("~")
        filepath="%s/.spectromicroscopy/"%(user)
        configfilename =os.path.join(filepath,"smp.conf")
        self.configfile=configfilename
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
        if os.path.isfile(configfilename):
            config = ConfigObj(configfilename)
        else:
            s = codecs.open(configfilename,'w','utf-8')
            s.close()
            config=ConfigObj()
            config.filename=configfilename
            config.write()
        if "setup" not in config.keys():
            config["setup"]={}
            config.write()
        if "server" not in config["setup"].keys():
            config["setup"]["server"]="f3.chess.cornell.edu"
            config.write()
        if "port" not in config["setup"].keys():
            config["setup"]["port"]="xrf"
            config.write()
        if "DEBUG" not in config["setup"].keys():
            config["setup"]["DEBUG"]=0
            config.write()
        self.__server=config["setup"]["server"]
        self.__port=config["setup"]["port"]
        self.DEBUG=config["setup"]["DEBUG"]
        self.filename=os.path.join(filepath,"pymca.cfg")
        reader=ConfigObj(self.filename)
        self.__peaks=[]
        try:
            elements=reader["peaks"]
            for key in elements.keys():
                self.__peaks.append("%s %s"%(key,elements[key]))
        except KeyError:
            pass

    
    def set_element(self):
        self.Image_Element="%s"%self.ElementSelect.currentText()
        self.change_limits()
    
    def auto_set(self):
        self.MinValSpin.setValue(0)
        self.MaxValSpin.setValue(0)
        self.change_limits()
    
    def change_limits(self):
        if self.setup!=0:
            parrent=self.ImageFrame.widget(0)
            self.Range_Max=self.MaxValSpin.value()
            self.Range_Min=self.MinValSpin.value()
            scale="%s"%self.ScaleBox.currentText()
            self.image=MyCanvas(self.ScanBox.currentText(),self.__images[self.Image_Element],
                                            self.x_index,self.y_index,self.Range_Min,self.Range_Max,parrent,self.energy,self.__totaled,scale)
            self.image.setGeometry(QtCore.QRect(169,-1,771,900))
            self.image.setMinimumSize(100,100)
            self.image.setObjectName("Graph")
            self.image.show()
    
    def run_scan(self):
        """
        sets up scans
        Pauses scan if running
        resumes scan if paused
        
        Inorder to run a scan it:
       1) establishes required values
       2)connects MCA_DATAas spec variable
       3)begins a timmer to gather data
        """
        if "%s"%self.Run.text()=="Scan":
            self.setup=0
            self.processed=[]
            indexs=[[1,1,1],[1,1,1],[1,1,1]]
            for i in range(len(self.Spin_Slide_Motor)):
                for j in range(3):
                    indexs[i][j]=self.Spin_Slide_Motor[i].get_settings()[j]
            count_time=self.Counter.value()
            self.Range_Max=self.MaxValSpin.value()
            self.Range_Min=self.MinValSpin.value()
            if "%s"%self.ScanBox.currentText()=="tseries":
                self.max=indexs[0][2]*indexs[1][2]
                self.xprun.set_cmd("tseries %s %s"%(self.max,count_time))
                self.x_index=indexs[0][2]
                self.y_index=1
            else:
                self.x_index=indexs[0][2]+1
                self.y_index=indexs[1][2]+1
                self.max=self.x_index*self.y_index
                x = self.X.get_motor_name()
                y = self.Y.get_motor_name()
                (xmin, xmax, xstep)=indexs[0]
                (ymin, ymax, ystep)=indexs[1]
                part_one=" %s %s %s %s"%(x,xmin,xmax,xstep)
                part_two=" %s %s %s %s"%(y,ymin, ymax, ystep)
                self.xprun.set_cmd("mesh"+part_one+part_two+" %s"%(count_time))
            self.xprun.exc("NPTS=0")
            self.processed=[]
            self.theory=ClassMcaTheory.McaTheory(self.filename)
            self.theory.enableOptimizedLinearFit()
            self.data = np.memmap(self.buffer.name,dtype=float,mode='w+',shape=(self.max,2048))
            self.xprun.exc("MCA_DATA=0")
            self.xprun.set_var('MCA_DATA',"Sync")
            self.__images = {}
            self.__sigmas = {}
            config=ConfigObj(self.filename)
            gain=float(config["detector"]["gain"])
            offset=float(config["detector"]["zero"])
            self.__totaled=np.zeros(2048, np.float_)
            converstion=0.01##TODO: read this value from config files
            self.energy = gain*np.arange(2048, dtype=np.float_)+offset
            self.timer = QtCore.QTimer(self)
            QtCore.QObject.connect(self.timer, QtCore.SIGNAL("timeout()"), self.data_collect)
            self.timer.start(20)
            self.xprun.run_cmd()
            self.Run.setText("Pause")
        elif "%s"%self.Run.text()=="Pause":
            print "Pause command"
            self.xprun.exc("")
            self.Run.setText("Resume")
        elif "%s"%self.Run.text()=="Resume":
            self.xprun.exc("scan_on")
            self.Run.setText("Pause")
    
    def data_collect(self):
        """gathers data from spec and processes it """
        max_index=self.max
        self.xprun.update()
        (value,index,actual)=self.xprun.get_values()
        if actual:
            typed=type(value[0])
            print "<<%s>> %s"%(index,typed)
            for i in range(len(value[0])):
                if len(value[0])>1:
                    self.data[index-1,i]=value[0][i][1]
                else:
                    self.data[index-1,i]=value[0][i]
            self.theory.setdata(range(2048),self.data[index-1],None)
            self.theory.estimate()
            fitresult, result = self.theory.startfit(digest=1)
            self.processed.append((fitresult,result))
            self.__peaks = []
            self.__nrows = len(range(0,max_index))
            for group in result['groups']:
                self.__peaks.append(group)
                if not self.setup:
                    self.__images[group] = np.zeros((self.__nrows,1), np.float_)
                    self.__sigmas[group] = np.zeros((self.__nrows,1), np.float_)
            self.__images['chisq']  = np.zeros((self.__nrows,1), np.float_) - 1.
            self.__images['chisq'][index-1, 0] = result['chisq']
            for peak in self.__peaks:
                if not self.setup:
                    self.__images[peak][index-1, 0] = result[peak]['fitarea']
                    self.__sigmas[peak][index-1,0] = result[peak]['sigmaarea']
                else:
                    self.__images[peak][index-1, 0] += result[peak]['fitarea']
                    self.__sigmas[peak][index-1,0] += result[peak]['sigmaarea']
            if self.Image_Element not in self.__peaks:self.Image_Element=self.__peaks[0]
            print self.__images[self.Image_Element]
            scale="%s"%self.ScaleBox.currentText()
            self.__totaled[1]+=self.data[index-1]
            parrent=self.ImageFrame.widget(0)
            self.image=MyCanvas(self.ScanBox.currentText(),self.__images[self.Image_Element],
                                        self.x_index,self.y_index,self.Range_Min,self.Range_Max,parrent,self.energy,self.__totaled,scale)
            self.image.setGeometry(QtCore.QRect(169,-1,771,800))
            self.image.setMinimumSize(100,100)
            self.image.setObjectName("Graph")
            self.image.show()
            self.ToolBar=ToolBar(self.image,parrent)
            self.ToolBar.setGeometry(QtCore.QRect(169,800,771,50))
            self.ToolBar.show()
            self.ToolBar.draw()
            self.ToolBar.update()
            self.setup=1
            print self.Image_Element
        if index==self.max:
            self.timer.stop()
            self.setup=2
            self.xprun.exc("MCA_DATA=0")
            self.Run.setText("Scan")
        

class MyCanvas(FigureCanvas):
    """This is a QWidget as well as a FigureCanvasAgg"""
    def __init__(self, visual,matrix, x, y, vmin, vmax,parent=None,energy_matrix=None,  plot_matrix=None,scale="linear", width=5, height=4, dpi=100):
        self.plot_matrix=plot_matrix
        self.energy_matrix=energy_matrix
        self.scale=scale
        self.visual=visual
        self.x=x
        self.y=y
        self.min=vmin
        if vmax ==0:
            self.max=None
            self.min=None
        elif vmin>vmax:
            self.min=vmax
            self.max=vmin
        else:
            self.max=vmax
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_axes([0.1,0.1,.6,.6])
        self.axes2=self.fig.add_axes([0.1,0.75,.8,.2])
        self.axes3=self.fig.add_axes([0.75,0.15,0.075,0.5])
        self.axes2.hold(False)
        self.axes.hold(False)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.matrix=matrix.reshape(self.x,self.y)
        self.compute_initial_figure()


    def new_data(self,matrix,scale):
        self.scale=scale
        self.matrix=matrix.reshape(self.x,self.y)
        if self.visual==Scans[2]:
            self.image.set_data(self.matrix)
            self.image.changed()
        elif self.visual==Scans[1]:
            matrix=self.matrix.flatten()
            self.axes.plot(matrix,"r-")
            self.axes.axis([0,len(matrix)-1,self.min,self.max])
        if self.scale=="linear":
            self.axes2.axis([0, len(self.plot_matrix), MA.minimum(self.plot_matrix), MA.maximum(self.plot_matrix)])
            self.axes2.plot(self.energy_matrix,self.plot_matrix,"b-")
        else:
            self.axes2.axis([0, 2048, 1,1000])
            self.axes2.semilogy(self.energy_matrix,self.plot_matrix,"b-")
        self.axes2.set_xlabel("Energy")
        self.axes2.set_ylabel("Count")
        self.draw()

    def sizeHint(self):
        w, h = self.get_width_height()
        return QtCore.QSize(w, h)

    def minimumSizeHint(self):
        return QtCore.QSize(10, 10)

    def compute_initial_figure(self):
        if self.plot_matrix!=None:
            if self.scale=="linear":
                self.axes2.axis([0, len(self.plot_matrix), MA.minimum(self.plot_matrix), MA.maximum(self.plot_matrix)])
                self.axes2.plot(self.energy_matrix,self.plot_matrix,"b-")
            else:
                self.axes2.axis([0, 2048, 1,1000])
                self.axes2.semilogy(self.energy_matrix,self.plot_matrix,"b-")
            self.axes2.set_xlabel("Energy")
            self.axes2.set_ylabel("Count")
        if self.visual==Scans[2]:
            self.image=self.axes.imshow(self.matrix,vmin=self.min, vmax=self.max,interpolation="nearest",origin="lower")
            self.cbar=self.fig.colorbar(self.image,self.axes3)
            matrix=self.matrix.flatten()
        elif self.visual==Scans[1]:
            matrix=self.matrix.flatten()
            self.axes.plot(matrix,"r-")
            self.axes.axis([0,len(matrix)-1,self.min,self.max])
        self.draw()
    


class Spinner_Slide_Motor(QtGui.QFrame):
    def __init__(self,parent,xprun,axis,number):
        QtGui.QFrame.__init__(self,parent)
        self.xprun=xprun
        self.setGeometry(QtCore.QRect(60+91*number,310,91,321))
        self.setFrameShape(QtGui.QFrame.StyledPanel)
        self.setFrameShadow(QtGui.QFrame.Raised)
        self.setObjectName("MotorWidget")

        self.Axis = QtGui.QLabel(self)
        self.Axis.setGeometry(QtCore.QRect(40,10,16,16))
        self.Axis.setObjectName("Axis")
        self.Axis.setText(axis)

        self.MotorName = QtGui.QLineEdit(self)
        self.MotorName.setGeometry(QtCore.QRect(20,30,51,24))
        self.MotorName.setObjectName("MotorName")
        if number==0:
            self.MotorName.setText("samx")
        elif number==1:
            self.MotorName.setText("samz")
        elif number==2:
            self.MotorName.setText("samy")

        self.ScanMin = QtGui.QDoubleSpinBox(self)
        self.ScanMin.setGeometry(QtCore.QRect(10,60,62,23))
        self.ScanMin.setObjectName("ScanMin")

        self.ScanMax = QtGui.QDoubleSpinBox(self)
        self.ScanMax.setGeometry(QtCore.QRect(10,90,62,23))
        self.ScanMax.setObjectName("ScanMax")

        self.ScanSteps = QtGui.QSpinBox(self)
        self.ScanSteps.setGeometry(QtCore.QRect(20,120,51,23))
        self.ScanSteps.setObjectName("ScanSteps")

        self.MotorSlide = QtGui.QSlider(self)
        self.MotorSlide.setGeometry(QtCore.QRect(30,160,25,121))
        self.MotorSlide.setSingleStep(1)
        self.MotorSlide.setOrientation(QtCore.Qt.Vertical)
        self.MotorSlide.setTickPosition(QtGui.QSlider.TicksBelow)
        self.MotorSlide.setTickInterval(10)
        self.MotorSlide.setObjectName("MotorSlide")

        self.MotorSpin = QtGui.QDoubleSpinBox(self)
        self.MotorSpin.setGeometry(QtCore.QRect(10,290,62,23))
        self.MotorSpin.setDecimals(4)
        self.MotorSpin.setSingleStep(0.01)
        self.MotorSpin.setObjectName("MotorSpin")
        
        self.Spin=self.MotorSpin
        self.Slide=self.MotorSlide
        self.Min=self.ScanMin
        self.Max=self.ScanMax
        self.Step=self.ScanSteps
        self.motor_get()
        QtCore.QObject.connect(self.Slide,QtCore.SIGNAL("sliderReleased()"),
                               self.slide)
        QtCore.QObject.connect(self.Spin,
                               QtCore.SIGNAL("editingFinished()"),self.spin)
        QtCore.QObject.connect(self.MotorName, QtCore.SIGNAL("editingFinished()"),self.motor_get)
        self.get_params()
        self.setup()
    def motor_get(self):
        name="%s"%self.MotorName.text()
        self.xprun.readmotors([name],"Async")
        self.Motor=self.xprun.get_motor(name)
    def get_params(self):
        self.values=[]
        params=['position','offset','sign',"low_limit","high_limit","step_size"]
        for param in params:
            try: 
                self.values.append(self.Motor.getParameter(param))
            except:
                self.values.append(0)
            
            
    def setup(self):
        values=self.values
        min =values[0]-1#values[2]*values[3]+values[1]
        max=values[0]#values[2]*values[4]+values[1]
        motor_step_size=values[5]
        self.Step.setRange(-100000,100000)
        self.Step.setValue(2)##1)
        self.Max.setRange(min,max)
        self.Min.setRange(min,max)
        self.Max.setValue(max)
        self.Min.setValue(min)
        self.Spin.setRange(min,max)
        self.Slide.setRange(min*motor_step_size,max*motor_step_size)
        self.Spin.setValue(self.values[0])
        self.Slide.setTickInterval(20*motor_step_size)
        self.spin()
    def slide(self):
        self.Spin.setValue(self.Slide.value()/self.values[5])
    def spin(self):
        self.Slide.setValue(self.Spin.value()*self.values[5])
    def Move(self):
        self.Motor.move(self.Spin.value())
    def get_settings(self):
        return (self.Min.value(),self.Max.value(),self.Step.value())
    def get_motor_name(self):
        return self.Motor.specName
        
        

class Customizer(QtGui.QDialog):
    def __init__(self,parent,filename):
        self.parent=parent
        QtGui.QDialog.__init__(self,parent)
        self.setObjectName("Customizer")
        self.resize(QtCore.QSize(QtCore.QRect(0,0,461,421).size()).expandedTo(self.minimumSizeHint()))

        self.buttonBox = QtGui.QDialogButtonBox(self)
        self.buttonBox.setGeometry(QtCore.QRect(0,390,461,32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.NoButton|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")

        self.tabWidget = QtGui.QTabWidget(self)
        self.tabWidget.setGeometry(QtCore.QRect(0,0,461,391))
        self.tabWidget.setObjectName("tabWidget")

        self.SetupTab = QtGui.QWidget()
        self.SetupTab.setObjectName("SetupTab")

        self.frame = QtGui.QFrame(self.SetupTab)
        self.frame.setGeometry(QtCore.QRect(0,0,201,371))
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName("frame")

        self.label = QtGui.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(10,40,41,16))
        self.label.setObjectName("label")

        self.PortLabel = QtGui.QLabel(self.frame)
        self.PortLabel.setGeometry(QtCore.QRect(20,80,23,16))
        self.PortLabel.setObjectName("PortLabel")

        self.PortEdit = QtGui.QLineEdit(self.frame)
        self.PortEdit.setGeometry(QtCore.QRect(60,80,121,24))
        self.PortEdit.setObjectName("PortEdit")

        self.ServerEdit = QtGui.QLineEdit(self.frame)
        self.ServerEdit.setGeometry(QtCore.QRect(60,40,121,24))
        self.ServerEdit.setObjectName("ServerEdit")
        self.tabWidget.addTab(self.SetupTab,"")

        self.Other = QtGui.QWidget()
        self.Other.setObjectName("Other")
        self.tabWidget.addTab(self.Other,"")
        self.label.setText("Server")
        self.PortLabel.setText("Port")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.SetupTab), "Setup")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Other), "Other")
        self.tabWidget.setCurrentIndex(0)
        QtCore.QObject.connect(self.buttonBox,QtCore.SIGNAL("accepted()"),self.accept)
        QtCore.QObject.connect(self.buttonBox,QtCore.SIGNAL("rejected()"),self.reject)
        QtCore.QObject.connect(self.ServerEdit,QtCore.SIGNAL("editingFinished()"),self.set_server)
        QtCore.QObject.connect(self.PortEdit,QtCore.SIGNAL("editingFinished()"),self.set_port)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.set=0
        self.server=''
        self.port=''
        self.result=""
        self.config=ConfigObj(filename)


    def set_server(self):
        self.server="%s"%self.ServerEdit.text()
        self.config["setup"]["server"]=self.server
    
    def set_port(self):
        self.port="%s"%self.PortEdit.text()
        self.config["setup"]["port"]=self.port
    
    
    def accept(self):
        self.config.write()
        QtGui.QDialog.accept(self)
    def rejected (self):
        self.set=1
        QtGui.QDialog.accept(self)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = MyXP()
    myapp.show()
    sys.exit(app.exec_())
