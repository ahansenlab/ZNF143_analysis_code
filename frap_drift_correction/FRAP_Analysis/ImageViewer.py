"""
ImageViewer.py - Simple image viewer for FRAP data
Originally Written by: Alec Heckert for package quot
https://github.com/alecheckert/quot
Modified by Domenic Narducci for FRAP
"""

import sys

# Image Reader
from .ImageReader import FRAPImage

# Model fitter
from .ModelFitter import BasicExponential, PureDiffusion, OneReaction, \
    TwoReaction, FullOneReactionAverage, FullTwoReactionAverage

# Core GUI utilities
from PySide2.QtCore import Qt, QLocale
from PySide2.QtWidgets import QWidget, QGridLayout, \
    QPushButton, QDialog, QLabel, QLineEdit, QShortcut,\
    QApplication, QListWidget, QListWidgetItem, QHBoxLayout, \
    QVBoxLayout, QComboBox, QAction, QMenuBar, QFileDialog, \
    QCheckBox
from PySide2.QtGui import QKeySequence, QFont
from PySide2.QtGui import Qt as QtGui_Qt
from .guiUtils import IntSlider, SingleImageWindow, LabeledQComboBox, \
    coerce_type, set_dark_app, LabeledCircleROI

# pyqtgraph utilities for showing images
import pyqtgraph
from pyqtgraph import ImageView, CircleROI, PlotWidget

# numpy
import numpy as np

# change list widget item comparison
def _new_lt_(self, other):
    return int(self.text()) < int(other.text())

QListWidgetItem.__lt__ = _new_lt_

MODELS = ["None", "Basic Exponential", "Pure Circular Diffusion",
          "Single Reaction Dominant", "Double Reaction Dominant",
          "Full Single Reaction (Avg)", "Full Double Reaction (Avg)"]

class ImageViewer(QWidget):
    """
    Show a single frame from a movie with a slider
    to change the frame. This essentially harnesses
    pyqtgraph.ImageView for a simple image viewer that is
    occasionally easier than Fiji.
    init
    ----
    """
    def __init__(self, path, parent=None):
        super(ImageViewer, self).__init__(parent=parent)
        self.path = path
        self.initData()
        self.initUI()
        self.createMenu()

        # Resize main window
        self.win.resize(1000, 900)

        # Show the main window
        self.win.show()

    def initData(self):
        """
        Try to read the image data at the target path.
        """
        self.ImageReader = FRAPImage(self.path)

    def initUI(self):
        """
        Initialize the user interface.
        """
        # Main window
        self.win = QWidget()
        self.win.setWindowTitle(self.path)
        layout = QGridLayout(self.win)

        # Image Viewer
        self.initImageViewer(layout)

        # Keyframes
        self.initKeyframeBox(layout)

        # Pre-processing
        self.initProcessingBox(layout)

        # Modeling
        self.initModelingBox(layout)

        # Update the frame
        self.load_frame(0, reset=True)

        # ROI intensity plotting
        self.IntensityPlot = PlotWidget(parent=self.win)
        self.IntensityMarker = self.make_intensity_plot()
        self.IntensityMarker.setZValue(10)
        self.ModelPlot = self.IntensityPlot.plot([],[])
        layout.addWidget(self.IntensityPlot, 3, 0, 1, 1)
        layout.setColumnStretch(0, 2)


    def initImageViewer(self, layout):
        """
        Initializes the image viewer for the UI
        :param layout: layout to initialize image viewer into
        :return:
        """
        # ImageView
        self.ImageView = ImageView(parent=self.win)
        layout.addWidget(self.ImageView, 0, 0, 2, -1)

        # ROI
        coords = self.ImageReader.get_roi_coords()
        radii = self.ImageReader.get_roi_radii()
        coords = (coords[0] - radii[0], coords[1] - radii[1])

        self.ROI = LabeledCircleROI(pos=coords,
                             radius=radii[0], label="Bleach")

        self.BG_ROI = LabeledCircleROI(pos=(10, 10), radius=5.0, label="Background")
        self.NUC_ROI = LabeledCircleROI(pos=(30, 30), radius=5.0, label="Nucleus")

        self.ImageView.getView().addItem(self.ROI)

        # Frame slider
        self.frame_slider = IntSlider(minimum=0, interval=1,
                                      maximum=self.ImageReader.tdim - 1, init_value=0,
                                      name='Frame', parent=self.win)
        layout.addWidget(self.frame_slider, 2, 0, 1, 2, alignment=Qt.AlignTop)
        self.frame_slider.assign_callback(self.frame_slider_callback)

        # Use the right/left keys to tab through frames
        self.left_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Left), self.win)
        self.right_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Right), self.win)
        self.left_shortcut.activated.connect(self.prev_frame)
        self.right_shortcut.activated.connect(self.next_frame)

    def initKeyframeBox(self, layout):
        """
        Initializes the keyframe box
        :param layout: layout to initialize the keyframe box into
        :return:
        """
        # Make button hbox
        keyframe_hbox = QHBoxLayout()

        # Make list hbox
        list_vbox = QVBoxLayout()
        list_header_hbox = QHBoxLayout()
        list_hbox = QHBoxLayout()
        list_footer_hbox = QHBoxLayout()

        # Add export keyframes button
        self.BleachButton = QPushButton("Set bleach frame", self.win)
        keyframe_hbox.addWidget(self.BleachButton)
        self.BleachButton.clicked.connect(self.bleach_button_callback)
        layout.addLayout(keyframe_hbox, 2, 2, alignment=Qt.AlignRight)
        list_vbox.addLayout(list_header_hbox)
        list_vbox.addLayout(list_hbox)
        list_vbox.addLayout(list_footer_hbox)
        layout.addLayout(list_vbox, 3, 2, alignment=Qt.AlignRight)

        # Make labels
        self.bleachKeyLabel = QLabel()
        self.bleachKeyLabel.setText("Bleach ROI")

        self.nucleusKeyLabel = QLabel()
        self.nucleusKeyLabel.setText("Nucleus ROI")

        self.bgKeyLabel = QLabel()
        self.bgKeyLabel.setText("Background ROI")

        list_header_hbox.addWidget(self.bleachKeyLabel)
        list_header_hbox.addWidget(self.nucleusKeyLabel)
        list_header_hbox.addWidget(self.bgKeyLabel)

        # Add keyframe list
        list_width = 105
        self.KeyframeList = QListWidget(parent=self.win)
        keyframes = self.ImageReader.get_keyframes()["Bleach"].keys()
        self.KeyframeList.addItems([str(key) for key in keyframes])
        self.KeyframeList.setFixedWidth(list_width)

        # Add nucleus list
        self.NucleusList = QListWidget(parent=self.win)
        keyframes_nuc = self.ImageReader.get_nucleus_keyframes().keys()
        self.NucleusList.addItems([str(key) for key in keyframes_nuc])
        self.NucleusList.setFixedWidth(list_width)

        # Add bg list
        self.BgList = QListWidget(parent=self.win)
        keyframes_bg = self.ImageReader.get_bg_keyframes().keys()
        self.BgList.addItems([str(key) for key in keyframes_bg])
        self.BgList.setFixedWidth(list_width)

        # Add keyframe buttons

        # bleach ROI
        self.KeyframeButton = QPushButton("Add keyframe", self.win)
        list_footer_hbox.addWidget(self.KeyframeButton)
        self.KeyframeButton.clicked.connect(self.keyframe_callback)

        # nonbleach ROI
        self.NonbleachKeyframeButton = QPushButton("Add keyframe", self.win)
        list_footer_hbox.addWidget(self.NonbleachKeyframeButton)
        self.NonbleachKeyframeButton.clicked.connect(self.nonbleach_keyframe_callback)

        # bg ROI
        self.BgKeyframeButton = QPushButton("Add keyframe", self.win)
        list_footer_hbox.addWidget(self.BgKeyframeButton)
        self.BgKeyframeButton.clicked.connect(self.bg_keyframe_callback)

        # TODO: Figure out how to sort by number instead of alphabetically
        self.KeyframeList.setSortingEnabled(False)
        self.NucleusList.setSortingEnabled(False)
        self.BgList.setSortingEnabled(False)
        # self.KeyframeList.sortItems(Qt.AscendingOrder)
        list_hbox.addWidget(self.KeyframeList)
        list_hbox.addWidget(self.NucleusList)
        list_hbox.addWidget(self.BgList)

        # make bleach frame label
        self.startFrameLabel = QLabel()
        self.startFrameLabel.setWordWrap(True)
        self.startFrameLabel.setText("Bleach Frame: {}".format(self.ImageReader.get_start_frame()))
        keyframe_hbox.addWidget(self.startFrameLabel)

        # Use the delete key to remove the selected keyframe
        self.delete_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Backspace), self.win)
        self.delete_shortcut.activated.connect(self.delete_keyframe)

    def createMenu(self):
        """
        Creates menu with options
        :return:
        """
        # init menu bar options
        self.mainMenu = QMenuBar(parent=self.win)
        self.fileMenu = self.mainMenu.addMenu("File")

        # create actions
        openAction = QAction("Open File...", self.win)
        openAction.setShortcut("Ctrl+O")
        openAction.triggered.connect(self.openImage)
        loadFrapAction = QAction("Load FRAP File...", self.win)
        loadFrapAction.triggered.connect(self.load_frap_file)
        saveAction = QAction("Save Data...", self.win)
        saveAction.setShortcut("Ctrl+S")
        saveAction.triggered.connect(self.saveData)
        exportCsvAction = QAction("Export as csv...", self.win)
        exportCsvAction.triggered.connect(self.exportAsCsv)

        # attach actions
        self.fileMenu.addAction(openAction)
        self.fileMenu.addAction(saveAction)
        self.fileMenu.addAction(exportCsvAction)

        # make ROI menu
        self.roiMenu = self.mainMenu.addMenu("ROI")
        roi_one = QAction("ROI 1", self.win)
        roi_two = QAction("ROI 2", self.win)
        roi_three = QAction("ROI 3", self.win)
        roi_four = QAction("ROI 4", self.win)
        roi_nuc = QAction("Nucleus ROI", self.win)
        roi_bg = QAction("Background ROI", self.win)

        roi_one.triggered.connect(self.setROIOne)
        roi_two.triggered.connect(self.setROITwo)
        roi_three.triggered.connect(self.setROIThree)
        roi_four.triggered.connect(self.setROIFour)
        roi_nuc.triggered.connect(self.setROINuc)
        roi_bg.triggered.connect(self.setROIBg)

        self.roiMenu.addAction(roi_one)
        self.roiMenu.addAction(roi_two)
        self.roiMenu.addAction(roi_three)
        self.roiMenu.addAction(roi_four)
        self.roiMenu.addAction(roi_nuc)
        self.roiMenu.addAction(roi_bg)

    def initProcessingBox(self, layout):
        """
        Initializes the processing box to customize processing features
        :param layout: layout to initialize the processing box into
        :return:
        """

        # make processing hbox
        processing_vbox = QVBoxLayout()

        # make processing header
        processingLabel = QLabel()
        processingLabel.setWordWrap(True)
        processingLabel.setText("Processing")
        processingLabel.setFont(QFont('Arial', 16))
        processing_vbox.addWidget(processingLabel)

        # make processing sub-sections
        processing_hbox = QHBoxLayout()

        ### PHOTOBLEACHING ###
        # make photobleaching subsection
        photobleaching_vbox = QVBoxLayout()
        photobleachingLabel = QLabel()
        photobleachingLabel.setWordWrap(True)
        photobleachingLabel.setText("Photobleaching Correction")
        photobleachingLabel.setFont(QFont('Arial', 14))
        photobleaching_vbox.addWidget(photobleachingLabel)

        # subtract background option
        self.backgroundCheckbox = QCheckBox("Subtract background")
        self.backgroundCheckbox.setChecked(True)
        photobleaching_vbox.addWidget(self.backgroundCheckbox)

        # show correction factors option
        self.corrFactorsCheckbox = QCheckBox("Show correction factors")
        self.corrFactorsCheckbox.setChecked(False)
        self.corrFactorsCheckbox.clicked.connect(self.corr_factors_callback)
        photobleaching_vbox.addWidget(self.corrFactorsCheckbox)

        # make photobleaching button
        self.photobleachingButton = QPushButton("Correct photobleaching")
        photobleaching_vbox.addWidget(self.photobleachingButton)
        self.photobleachingButton.clicked.connect(self.photobleaching_callback)
        photobleaching_vbox.addWidget(self.photobleachingButton)

        # make photobleaching label
        self.photobleachingLabel = QLabel()
        self.photobleachingLabel.setWordWrap(True)
        self.photobleachingLabel.setText("Photobleaching t1/2 = ")
        photobleaching_vbox.addWidget(self.photobleachingLabel)

        # make photobleaching variance label
        self.photobleachingVarLabel = QLabel()
        self.photobleachingVarLabel.setWordWrap(True)
        self.photobleachingVarLabel.setText("Fit Variance = ")
        photobleaching_vbox.addWidget(self.photobleachingVarLabel)

        # add option to see nuclear segmentation results
        self.segmentationCheckbox = QCheckBox("Show segmentation results")
        self.segmentationCheckbox.setChecked(False)
        self.segmentationCheckbox.clicked.connect(self.segmentation_callback)
        photobleaching_vbox.addWidget(self.segmentationCheckbox)

        # kernel size slider
        self.kernel_slider = IntSlider(minimum=0, interval=1,
                                      maximum=50, init_value=10,
                                      name='Kernel size', parent=self.win)
        photobleaching_vbox.addWidget(self.kernel_slider)
        self.kernel_slider.slider.sliderReleased.connect(self.kernel_slider_callback)

        # threshold slider
        self.threshold_slider = IntSlider(minimum=0, interval=1,
                                      maximum=200, init_value=0,
                                      name='Threshold', parent=self.win)
        photobleaching_vbox.addWidget(self.threshold_slider)
        self.threshold_slider.slider.sliderReleased.connect(self.threshold_slider_callback)

        roi_otsu_hbox = QHBoxLayout()
        # roi option
        self.useRoiCheckbox = QCheckBox("Use roi?")
        self.useRoiCheckbox.setChecked(False)
        self.useRoiCheckbox.clicked.connect(self.roi_nuc_callback)
        roi_otsu_hbox.addWidget(self.useRoiCheckbox)

        # otsu option
        self.otsuCheckbox = QCheckBox("Use otsu?")
        self.otsuCheckbox.setChecked(True)
        self.otsuCheckbox.clicked.connect(self.otsu_callback)
        roi_otsu_hbox.addWidget(self.otsuCheckbox)

        photobleaching_vbox.addLayout(roi_otsu_hbox)

        ### NORMALIZATION ###
        # make normalization subsection
        normalization_vbox = QVBoxLayout()
        normalizationLabel = QLabel()
        normalizationLabel.setWordWrap(True)
        normalizationLabel.setText("Normalization")
        normalizationLabel.setFont(QFont('Arial', 14))
        normalization_vbox.addWidget(normalizationLabel)

        # make normalization combobox
        self.NormalBox = QComboBox()
        self.NormalBox.addItem('Fullscale')
        self.NormalBox.addItem('Double')
        normalization_vbox.addWidget(self.NormalBox)

        # steady state option
        self.steadyStateCheckbox = QCheckBox("Use pre-bleach as steady-state")
        self.steadyStateCheckbox.setChecked(True)
        normalization_vbox.addWidget(self.steadyStateCheckbox)

        # make photobleaching button
        self.normalizationButton = QPushButton("Normalize")
        normalization_vbox.addWidget(self.normalizationButton)
        self.normalizationButton.clicked.connect(self.normalization_callback)
        normalization_vbox.addWidget(self.normalizationButton)

        ### QC ###
        # make QC subsection
        qcLabel = QLabel()
        qcLabel.setWordWrap(True)
        qcLabel.setText("QC Metrics")
        qcLabel.setFont(QFont('Arial', 14))
        normalization_vbox.addWidget(qcLabel)

        # show qc measurements
        # nuclear radius, roi radius, gap ratio, bleaching depth, radius uniform
        # make nuclear radius label
        self.nuclearRadiusLabel = QLabel()
        self.nuclearRadiusLabel.setWordWrap(True)
        self.nuclearRadiusLabel.setText("Nuclear Radius = ")
        normalization_vbox.addWidget(self.nuclearRadiusLabel)

        # make roi radius label
        self.roiRadiusLabel = QLabel()
        self.roiRadiusLabel.setWordWrap(True)
        self.roiRadiusLabel.setText("Roi Radius = ")
        normalization_vbox.addWidget(self.roiRadiusLabel)

        # make gap ratio label
        self.gapRatioLabel = QLabel()
        self.gapRatioLabel.setWordWrap(True)
        self.gapRatioLabel.setText("Gap Ratio = ")
        normalization_vbox.addWidget(self.gapRatioLabel)

        # make bleaching depth label
        self.bleachDepthLabel = QLabel()
        self.bleachDepthLabel.setWordWrap(True)
        self.bleachDepthLabel.setText("Bleaching Depth = ")
        normalization_vbox.addWidget(self.bleachDepthLabel)

        # make radius uniform label
        self.radiusUniformLabel = QLabel()
        self.radiusUniformLabel.setWordWrap(True)
        self.radiusUniformLabel.setText("Radius Uniform = ")
        normalization_vbox.addWidget(self.radiusUniformLabel)

        # make QC button
        self.QCButton = QPushButton("Recalculate QC Metrics")
        normalization_vbox.addWidget(self.QCButton)
        self.QCButton.clicked.connect(self.qc_callback)
        normalization_vbox.addWidget(self.QCButton)

        photobleaching_vbox.addStretch()
        normalization_vbox.addStretch()
        processing_hbox.addLayout(photobleaching_vbox)
        processing_hbox.addLayout(normalization_vbox)
        processing_vbox.addLayout(processing_hbox)
        processing_vbox.addStretch()
        layout.addLayout(processing_vbox, 4, 0, 1, 1)

    def initModelingBox(self, layout):
        """
        Initializes the modeling box to customize processing features
        :param layout: layout to initialize the modeling box into
        :return:
        """

        # make processing hbox
        modeling_vbox = QVBoxLayout()

        # make processing header
        modelingLabel = QLabel()
        modelingLabel.setWordWrap(True)
        modelingLabel.setText("Modeling")
        modelingLabel.setFont(QFont('Arial', 16))

        modeling_vbox.addWidget(modelingLabel)

        # Model fit combo box
        self.ModelBox = QComboBox(self.win)
        for model in MODELS:
            self.ModelBox.addItem(model)
        modeling_vbox.addWidget(self.ModelBox)

        # Fit hbox
        fit_hbox = QHBoxLayout()

        # Fit label
        self.ModelParamLabel = QLabel()
        self.ModelParamLabel.setWordWrap(True)
        fit_hbox.addWidget(self.ModelParamLabel)

        # Fit button
        self.ModelFitButton = QPushButton("Fit Model", self.win)
        self.ModelFitButton.clicked.connect(self.fit_model)
        fit_hbox.addWidget(self.ModelFitButton)
        modeling_vbox.addLayout(fit_hbox)
        modeling_vbox.addStretch()
        layout.addLayout(modeling_vbox, 4, 2, 1, 1)

    def openImage(self):
        """
        Opens a file browser
        :return:
        """
        filename = QFileDialog.getOpenFileName(parent=self.win, caption="Open file",
                                               filter="Image files (*.czi);;Acquisition block (*.czmbi)")
        self.path = filename[0]
        self.refresh()

    def load_frap_file(self):
        """
        Opens a file browser and opens a frap file
        """
        filename = QFileDialog.getOpenFileName(parent=self.win, caption="Load FRAP file",
                                               filter="FRAP files (*.frap)")
        self.path = filename[0]
        self.refresh()

    def saveData(self):
        """
        Opens a file browser to save data
        :return:
        """
        filename = QFileDialog.getSaveFileName(parent=self.win, caption="Save file",
                                               filter="(*.frap)")
        self.ImageReader.save_data(filename[0])

    def exportAsCsv(self):
        """
        Opens a file browser to export as csv
        :return:
        """
        filename = QFileDialog.getSaveFileName(parent=self.win, caption="Save file",
                                               filter="(*.csv)")
        self.ImageReader.export_to_csv(filename[0])

    def setROIOne(self):
        """
        Changes ROI to ROI one. 
        TODO: FIX The way this is implemented
        """
        self.ImageReader.set_roi(0)
        self.refresh(reinit_image=False)

    def setROITwo(self):
        """
        Changes ROI to ROI one. 
        TODO: FIX The way this is implemented
        """
        self.ImageReader.set_roi(1)
        self.refresh(reinit_image=False)

    def setROIThree(self):
        """
        Changes ROI to ROI one. 
        TODO: FIX The way this is implemented
        """
        self.ImageReader.set_roi(2)
        self.refresh(reinit_image=False)

    def setROIFour(self):
        """
        Changes ROI to ROI one. 
        TODO: FIX The way this is implemented
        """
        self.ImageReader.set_roi(3)
        self.refresh(reinit_image=False)

    def setROINuc(self):
        """
        Changes ROI to ROI one.
        TODO: FIX The way this is implemented
        """
        self.ImageReader.set_roi(3)
        self.refresh(reinit_image=False)

    def setROIBg(self):
        """
        Changes ROI to ROI one.
        TODO: FIX The way this is implemented
        """
        self.ImageReader.set_roi(3)
        self.refresh(reinit_image=False)

    def refresh(self, reinit_image=True):
        """
        Refreshes imageviewer, keyframes, model plot, and slider
        :return:
        """
        self.win.setWindowTitle(self.path)
        if reinit_image:
            self.initData()
        self.load_frame(0, reset=True)
        keyframes = self.ImageReader.get_keyframes()
        self.KeyframeList.clear()
        self.NucleusList.clear()
        self.BgList.clear()
        self.KeyframeList.addItems([str(key) for key in keyframes["Bleach"].keys()])
        self.NucleusList.addItems([str(key) for key in keyframes["Nucleus"].keys()])
        self.BgList.addItems([str(key) for key in keyframes["Background"].keys()])
        # update plot
        self.update_plot()

        # delete label
        self.ModelParamLabel.setText("")

        # reset int slider
        self.frame_slider.setMax(self.ImageReader.tdim - 1)

        # reset bleach frame
        self.startFrameLabel.setText("Bleach Frame: {}".format(self.ImageReader.get_start_frame()))

    def update_plot(self):
        """
        Updates the plot
        :return:
        """

        # update plot
        self.IntensityMarker = self.make_intensity_plot()
        self.ModelPlot.clear()
        self.ModelPlot = self.IntensityPlot.plot([], [])

    def make_intensity_plot(self):
        """
        Sets all parameters to initialize ROI intensity plot
        :return: Returns plot object for current frame marker
        """

        time = self.ImageReader.get_frame_metadata()[:, 1]
        time -= time[self.ImageReader.get_start_frame()]

        intensity = self.ImageReader.get_mean_intensity_data()
        min_y_range, max_y_range = min(intensity), max(intensity)
        roi_pen = pyqtgraph.mkPen('w', width=3)
        self.IntensityPlot.clear()
        self.IntensityPlot.addLegend(offset=(-1, -50))
        self.ROIIntensityPlot = self.IntensityPlot.plot(name="ROI")
        self.ROIIntensityPlot.setData(time, intensity, pen=roi_pen)
        bleach_pen = pyqtgraph.mkPen('y', width=3)
        corr_pen = pyqtgraph.mkPen('r', width=1)

        if self.useRoiCheckbox.isChecked():
            raw_nuc_intensity, nuc_intensity = self.ImageReader.get_nucleus_intensity_data()
            raw_bg_intensity, bg_intensity = self.ImageReader.get_bg_intensity_data()
            if self.corrFactorsCheckbox.isChecked():
                self.rawNucIntensityPlot = self.IntensityPlot.plot(name="Raw nucleus")
                self.rawNucIntensityPlot.setData(time, raw_nuc_intensity, pen=bleach_pen)
                self.rawBgIntensityPlot = self.IntensityPlot.plot(name="Raw background")
                self.rawBgIntensityPlot.setData(time, raw_bg_intensity, pen=corr_pen)
                min_y_range = min(min_y_range, min(raw_nuc_intensity), min(raw_bg_intensity))
                max_y_range = max(max_y_range, max(raw_nuc_intensity), max(raw_bg_intensity))

            else:
                self.NucIntensityPlot = self.IntensityPlot.plot(name="Nucleus")
                self.NucIntensityPlot.setData(time, nuc_intensity, pen=bleach_pen)
                self.BgIntensityPlot = self.IntensityPlot.plot(name="Background")
                self.BgIntensityPlot.setData(time, bg_intensity, pen=corr_pen)
                min_y_range = min(min_y_range, min(nuc_intensity), min(bg_intensity))
                max_y_range = max(max_y_range, max(nuc_intensity), max(bg_intensity))

        else:
            if self.corrFactorsCheckbox.isChecked():
                nonbleach_intensity = self.ImageReader.get_nonbleach_intensities()
                corr_factors = self.ImageReader.get_correction_factors()
                self.CFIntensityPlot = self.IntensityPlot.plot(name="Correction factors")
                self.CFIntensityPlot.setData(time, corr_factors * nonbleach_intensity[0], pen=corr_pen)

            else:
                nonbleach_intensity = self.ImageReader.get_corrected_nonbleach_intensities()
            min_y_range = min(min_y_range, min(nonbleach_intensity))
            max_y_range = max(max_y_range, max(nonbleach_intensity))
            self.NonbleachIntensityPlot = self.IntensityPlot.plot(name="Nuclear segmentation")
            self.NonbleachIntensityPlot.setData(time, nonbleach_intensity, pen=bleach_pen)

        self.IntensityPlot.setLabel('left', 'Mean Intensity')
        self.IntensityPlot.setLabel('bottom', 'Time', units='s')
        self.IntensityPlot.setXRange(min(time), max(time)+ 50)
        self.IntensityPlot.setYRange(min_y_range, max_y_range)

        # current frame marker
        current_frame = self.frame_slider.value()
        intensityMarker = self.IntensityPlot.plot([time[current_frame]],
                                       [intensity[current_frame]], pen = None, symbol = '+')
        return intensityMarker

    def load_frame(self, frame_index, reset=False):
        """
        Change the current frame.
        args
        ----
            frame_index     :   int
            reset           :   bool, reset the LUTs and ROI
        """
        if self.segmentationCheckbox.isChecked():
            self.image = self.ImageReader.get_segment_frame(frame_index)
        else:
            self.image = self.ImageReader.get_frame(frame_index)
        self.ImageView.setImage(self.image, autoRange=reset, autoLevels=reset,
            autoHistogramRange=reset)
        self.load_rois(frame_index, set_rois=False)

    def load_rois(self, frame_index, set_rois=True):
        """
        Load ROIs in the current frame
        """
        if set_rois:
            self.ImageReader.set_roi_radii(self.ROI.size()[0]/2)
            self.ImageReader.set_roi_radii(self.NUC_ROI.size()[0]/2, "Nucleus")
            self.ImageReader.set_roi_radii(self.BG_ROI.size()[0]/2, "Background")
        radii = self.ImageReader.get_roi_radii()
        self.ROI.setPos(self.ImageReader.get_viewer_coords()[frame_index])
        self.ROI.setSize([2.0 * r for r in radii])

        nuc_radii = self.ImageReader.get_roi_radii("Nucleus")
        self.NUC_ROI.setPos(self.ImageReader.get_viewer_coords("Nucleus")[frame_index])
        self.NUC_ROI.setSize([2.0 * r for r in nuc_radii])

        bg_radii = self.ImageReader.get_roi_radii("Background")
        self.BG_ROI.setPos(self.ImageReader.get_viewer_coords("Background")[frame_index])
        self.BG_ROI.setSize([2.0 * r for r in bg_radii])

    def load_plot_marker(self, frame_index):
        """
        Changes the location of the marker on the plot.
        :param frame_index: (int) frame index
        """
        tpt, ipt = self.ImageReader.get_time_intensity_pt(frame_index)
        self.IntensityMarker.setData([tpt],[ipt])

    def next_frame(self):
        """
        Go to the frame after the current one.
        """
        next_idx = int(self.frame_slider.value())
        if next_idx < self.frame_slider.maximum:
            next_idx += 1
        self.frame_slider.setValue(next_idx)

    def prev_frame(self):
        """
        Go the frame before the current one.
        """
        prev_idx = int(self.frame_slider.value())
        if prev_idx > self.frame_slider.minimum:
            prev_idx -= 1
        self.frame_slider.setValue(prev_idx)

    def delete_keyframe(self):
        """
        Delete the selected keyframe
        """
        for key_list, key in zip([self.KeyframeList, self.NucleusList, self.BgList], ["Bleach", "Nucleus", "Background"]):
            if key_list.currentItem() is not None:
                currentIdx = int(key_list.currentItem().text())
                if currentIdx != 0 and currentIdx != self.ImageReader.get_tdim()-1:
                    # delete keyframe
                    self.ImageReader.del_keyframe(currentIdx, key)
                    key_list.takeItem(key_list.currentRow())

                    # update plot
                    self.update_plot()

                    # update image
                    self.load_frame(self.frame_slider.value())



    def frame_slider_callback(self):
        """
        Change the current frame.
        """
        self.load_frame(self.frame_slider.value())
        self.load_plot_marker(self.frame_slider.value())

    def kernel_slider_callback(self):
        """
        Changes segmentation kernel
        """
        self.ImageReader.set_kernel_size(self.kernel_slider.value())
        self.load_frame(self.frame_slider.value(), reset=True)
        # update plot
        self.update_plot()

    def threshold_slider_callback(self):
        """
        Changes segmentation threshold.
        """
        if not self.otsuCheckbox.isChecked():
            self.ImageReader.set_threshold(self.threshold_slider.value() / 200.0)
            self.load_frame(self.frame_slider.value(), reset=True)
        # update plot
        self.update_plot()

    def otsu_callback(self):
        """
        Changes segmentation threshold.
        """
        if self.otsuCheckbox.isChecked():
            self.ImageReader.set_threshold("otsu")
        else:
            self.ImageReader.set_threshold(self.threshold_slider.value() / 100.0)
        self.load_frame(self.frame_slider.value(), reset=True)
        # update plot
        self.update_plot()
        
    def roi_nuc_callback(self):
        """
        Use an ROI for the nucleus intensity
        """
        self.ImageReader.set_roi_flag(self.useRoiCheckbox.isChecked())
        if self.useRoiCheckbox.isChecked():
            # self.ROI = CircleROI(pos=coords, radius=radii[0], resizable=False, rotatable=False)
            self.ImageView.getView().addItem(self.BG_ROI)
            self.ImageView.getView().addItem(self.NUC_ROI)
        else:
            self.ImageView.getView().removeItem(self.BG_ROI)
            self.ImageView.getView().removeItem(self.NUC_ROI)
            # self.ImageReader.set_threshold(self.threshold_slider.value() / 100.0)
        self.load_frame(self.frame_slider.value(), reset=True)
        # update plot
        self.update_plot()

    def keyframe_callback(self):
        """
        Add a keyframe for the ROI at the current frame and ROI position
        """
        # update keyframes in image reader
        self.ImageReader.set_keyframe(self.frame_slider.value(),
                                      tuple(self.ROI.pos()))
        self.ImageReader.set_roi_radii(self.ROI.size()[0]/2)

        # update keyframe list
        keyframe_items = [int(self.KeyframeList.item(i).text()) \
                          for i in range(self.KeyframeList.count())]
        if self.frame_slider.value() not in keyframe_items:
            self.KeyframeList.addItem(str(self.frame_slider.value()))
            #self.KeyframeList.sortItems(Qt.AscendingOrder)

        # detach model
        self.ImageReader.detach_model()

        # update plot
        self.update_plot()

        # if showing segmentation, recalculate
        if self.segmentationCheckbox.isChecked():
            self.ImageReader.refresh_segmentation()
            self.load_frame(self.frame_slider.value(), reset=True)

        # delete label
        self.ModelParamLabel.setText("")

    def nonbleach_keyframe_callback(self):
        """
        Add a keyframe for the ROI at the current frame and ROI position
        """
        # update keyframes in image reader
        self.ImageReader.set_keyframe(self.frame_slider.value(), tuple(self.NUC_ROI.pos()), "Nucleus")
        self.ImageReader.set_roi_radii(self.NUC_ROI.size()[0]/2, "Nucleus")

        # update keyframe list
        keyframe_items = [int(self.NucleusList.item(i).text()) \
                          for i in range(self.NucleusList.count())]
        if self.frame_slider.value() not in keyframe_items:
            self.NucleusList.addItem(str(self.frame_slider.value()))
            #self.KeyframeList.sortItems(Qt.AscendingOrder)

        # detach model
        self.ImageReader.detach_model()

        # update plot
        self.update_plot()

        # delete label
        self.ModelParamLabel.setText("")

    def bg_keyframe_callback(self):
        """"""
        """
        Add a keyframe for the ROI at the current frame and ROI position
        """
        # update keyframes in image reader
        self.ImageReader.set_keyframe(self.frame_slider.value(), tuple(self.BG_ROI.pos()), "Background")
        self.ImageReader.set_roi_radii(self.BG_ROI.size()[0]/2, "Background")

        # update keyframe list
        keyframe_items = [int(self.BgList.item(i).text()) \
                          for i in range(self.BgList.count())]
        if self.frame_slider.value() not in keyframe_items:
            self.BgList.addItem(str(self.frame_slider.value()))
            # self.KeyframeList.sortItems(Qt.AscendingOrder)

        # detach model
        self.ImageReader.detach_model()

        # update plot
        self.update_plot()

        # delete label
        self.ModelParamLabel.setText("")

    def bleach_button_callback(self):
        """
        Set's the start frame for bleaching
        """
        bleach_frame = self.frame_slider.value()
        self.ImageReader.set_start_frame(bleach_frame)
        self.startFrameLabel.setText("Bleach Frame: {}".format(self.ImageReader.get_start_frame()))
        # update plot
        self.update_plot()


    def photobleaching_callback(self):
        """
        Corrects photobleaching/handles pressing of button
        :return:
        """
        self.ImageReader.correct_photobleaching(self.backgroundCheckbox.isChecked())

        # reload frames
        self.load_frame(0, reset=True)

        # update plot
        self.update_plot()

        # delete label
        self.ModelParamLabel.setText("")

        # get photobleaching params tau and tau variance
        tau, tau_var = self.ImageReader.get_photobleaching_params()
        t_half = 0.693147 / tau[0]
        self.photobleachingLabel.setText("Photobleaching t1/2 = {:.3g} [1/s]".format(t_half))
        self.photobleachingVarLabel.setText("Fit Variance = {:.3g}".format(*tau_var[0]))

    def qc_callback(self):
        """
        Calculates QC metrics
        """
        nuc_radius, roi_radius, gap_ratio, bleach_depth, radius_uniform = self.ImageReader.recalculate_qc_metrics(self.steadyStateCheckbox.isChecked(),
                                                                                                                  self.backgroundCheckbox.isChecked())
        self.nuclearRadiusLabel.setText("Nuclear Radius = {:.3g} [um]".format(nuc_radius))
        self.roiRadiusLabel.setText("Roi Radius = {:.3g} [um]".format(roi_radius))
        self.gapRatioLabel.setText("Gap Ratio = {:.3g}".format(gap_ratio))
        self.bleachDepthLabel.setText("Bleaching Depth = {:.3g}".format(bleach_depth))
        self.radiusUniformLabel.setText("Radius Uniform = {:.3g} [um]".format(radius_uniform))

    def segmentation_callback(self):
        """
        Shows nuclear segmentation results upon box click.
        :return:
        """
        if self.segmentationCheckbox.isChecked():
            self.ImageReader.refresh_segmentation()
        self.load_frame(self.frame_slider.value(), reset=True)

    def corr_factors_callback(self):
        """
        Shows nuclear segmentation results upon box click.
        :return:
        """
        # update plot
        self.update_plot()

    def normalization_callback(self):
        """
        Normalizes the FRAP curve
        :return:
        """
        current_selection = str(self.NormalBox.currentText())
        prebleach_ss = self.steadyStateCheckbox.isChecked()
        self.ImageReader.normalize_frap_curve(current_selection, prebleach_ss)

        # reload frames
        # self.load_frame(0, reset=True)
        print(self.ImageReader.mean_intensity_data)

        # update plot
        self.update_plot()

        # delete label
        self.ModelParamLabel.setText("")

    def fit_model(self):
        """
        Attaches model, fits it and displays it
        """
        SELECTIONS = {"Basic Exponential":BasicExponential,
                      "Pure Circular Diffusion":PureDiffusion,
                      "Single Reaction Dominant":OneReaction,
                      "Double Reaction Dominant":TwoReaction,
                      "Full Single Reaction (Avg)":FullOneReactionAverage,
                      "Full Double Reaction (Avg)":FullTwoReactionAverage}
        LABELS = {"Basic Exponential":"A = {}\nTau = {}",
                  "Pure Circular Diffusion":"t_d = {}\nD = {}",
                  "Single Reaction Dominant":"k_off = {}\nk_on_* = {}",
                  "Double Reaction Dominant":"k_off1 = {}\nk_off2 = {}\nk_on1_* = {}\n k_on2_* = {}",
                  "Full Single Reaction (Avg)":"k_on = {}\nk_off = {}\nD_f = {}",
                  "Full Double Reaction (Avg)":"k_on_1 = {}\nk_on_2 = {}\nk_off_1 = {}\nk_off_2 = {}\nD_f = {}"}
        current_selection = str(self.ModelBox.currentText())
        if current_selection == "None":
            self.ImageReader.detach_model()
            self.ModelPlot.clear()
            self.ModelPlot = self.IntensityPlot.plot([],[])
            self.ModelParamLabel.setText("")
        else:
            model_pen = pyqtgraph.mkPen('c', width=4)
            self.ImageReader.attach_model(SELECTIONS.get(current_selection))
            self.ModelPlot.clear()
            self.ModelPlot = self.IntensityPlot.plot(*self.ImageReader.get_model_data(),
                                                     pen=model_pen)
            label_text = LABELS.get(current_selection)
            param_vals = self.ImageReader.get_model_params()[0]
            param_vals = [round(val, 4) for val in param_vals]
            self.ModelParamLabel.setText(label_text.format(*param_vals))


    def B_max_int_callback(self):
        """
        Make a maximum intensity projection.
        """
        ex = ChooseProjectionDialog(self.ImageReader.tdim, parent=self)
        if ex.exec_() == QDialog.Accepted:
            method, start_frame, stop_frame = ex.return_val

            # Perform the projection
            result = getattr(self.ImageReader, method)(start=int(start_frame),
                stop=int(stop_frame))

            # Make a standalone window showing the projection
            ex = SingleImageWindow(result, title=method, parent=self)
            ex.show()

class ChooseProjectionDialog(QDialog):
    def __init__(self, n_frames, parent=None):
        super(ChooseProjectionDialog, self).__init__(parent=parent)
        self.n_frames = n_frames
        self.initUI()

    def initUI(self):
        layout = QGridLayout(self)
        self.setWindowTitle("Select projection")

        # Menu to select type of projection
        proj_types = ['max_int_proj', 'sum_proj']
        self.M_proj = LabeledQComboBox(proj_types, "Projection type",
            init_value="max_int_proj", parent=self)
        layout.addWidget(self.M_proj, 0, 0, 1, 2, alignment=Qt.AlignRight)

        # Entry boxes to choose start and stop frames
        label_0 = QLabel(self)
        label_1 = QLabel(self)
        label_0.setText("Start frame")
        label_1.setText("Stop frame")
        layout.addWidget(label_0, 1, 0, alignment=Qt.AlignRight)
        layout.addWidget(label_1, 2, 0, alignment=Qt.AlignRight)

        self.EB_0 = QLineEdit(self)
        self.EB_1 = QLineEdit(self)
        self.EB_0.setText(str(0))
        self.EB_1.setText(str(self.n_frames))
        layout.addWidget(self.EB_0, 1, 1, alignment=Qt.AlignLeft)
        layout.addWidget(self.EB_1, 2, 1, alignment=Qt.AlignLeft)

        # Accept button
        self.B_accept = QPushButton("Accept", parent=self)
        self.B_accept.clicked.connect(self.B_accept_callback)
        layout.addWidget(self.B_accept, 3, 0, alignment=Qt.AlignRight)

    def B_accept_callback(self):
        """
        Accept the current projection settings and return to the
        client widget.
        """
        try:
            self.return_val = [
                self.M_proj.currentText(),
                coerce_type(self.EB_0.text(), int),
                coerce_type(self.EB_1.text(), int),
            ]
            self.accept()
        except ValueError:
            print("Frame values must be integers")


def launch(path):
    pyqtgraph.Qt.QT_LIB = "PySide2"
    app = QApplication([])
    set_dark_app(app)
    instance = ImageViewer(path)
    sys.exit(app.exec_())
