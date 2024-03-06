"""
guiUtils.py -- helper utilities for the quot GUI
Written by: Alec Heckert for package quot
https://github.com/alecheckert/quot

"""

# Numeric
import numpy as np

# Main GUI utilities
from PySide2.QtCore import Qt, QSize
from PySide2.QtGui import QPalette, QColor, QPainterPath, QFont
from PySide2.QtWidgets import QSlider, QWidget, \
    QGridLayout, QVBoxLayout, QLabel, QComboBox

# pyqtgraph utilities
from pyqtgraph import ImageView, CircleROI, TextItem


def set_dark_app(qApp):
    """
    Set the color scheme of a PySide2 QApplication to
    a darker default, inherited by all children.

    Modifies the QApplication in place.

    """
    qApp.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    qApp.setPalette(palette)
    qApp.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")

class LabeledCircleROI(CircleROI):
    """
    A circle ROI with a label derived from the pyqtgraph class.

    init
    ----
    pos     :   tuple, position of the roi
    radius  :   radius of circle
    label   :   text label for ROI
    """


    def __init__(self, pos, radius, label):
        self.TextLabel = TextItem(text=label, color='w', anchor=(0, 0))
        super().__init__(pos=pos, radius=radius, resizable=True, rotatable=False)
        font = QFont()
        font.setPointSize(font.pointSize() * 0.75)
        self.TextLabel.setParentItem(self)
        self.TextLabel.setFont(font)
        self.TextLabel.setPos(radius, radius)
        self.radius = radius

    # def setSize(self, size, **kwargs):
    #     super().setSize(size, **kwargs)
    #     self.radius = self.size()[0] / 2
    #     self.TextLabel.setPos(self.radius, self.radius)
    #     print(self.pos())


class IntSlider(QWidget):
    """
    An integer-valued slider widget with increment 1, to
    be added to a larger GUI. Included is (a) a label for
    the minimum value the slider can assume, (b) a label
    for the maximum value the slider can assume, (c) the
    current slider value, and (d) a name for the slider.

    init
    ----
        parent          :   root QWidget
        minimum         :   int, minimum value for the slider
        maximum         :   int, maximum value for the slider
        interval        :   int, the interval between ticks
        init_value      :   int, initial value
        name            :   str

    """
    def __init__(self, minimum=0, maximum=10, interval=1,
        init_value=0, name=None, min_width=225, parent=None):
        super(IntSlider, self).__init__(parent=parent)

        self.minimum = int(minimum)
        self.maximum = int(maximum)
        self.interval = int(interval)
        self.min_width = min_width
        self.init_value = int(init_value)
        if name is None:
            name = ''
        self.name = name

        # If the interval is not 1, figure out whether
        # the maximum needs to be decreased for an integral
        # number of intervals
        self.slider_values = self._set_slider_values(self.minimum,
            self.maximum, self.interval)
        self.maximum = self.slider_values.max()

        self.initUI()

    def _set_slider_values(self, minimum, maximum, interval):
        """
        Configure the values of the slider, useful when the
        interval is not unity.

        """
        if interval != 1:
            n_intervals = (maximum-minimum)//interval + 1
            slider_values = minimum + interval * \
                np.arange(n_intervals).astype(np.int64)
            slider_values = slider_values[slider_values<=maximum]
        else:
            slider_values = np.arange(minimum, maximum+1).astype(np.int64)

        return slider_values

    def initUI(self):
        """
        Initialize the user interface.

        """
        # Main window
        self.win = QWidget(self)
        self.layout = QGridLayout()
        self.win.setLayout(self.layout)

        # Current value / title label
        self.L_title = QLabel(self.win)
        self.layout.addWidget(self.L_title, 0, 1)

        # Label for minimum and maximum
        self.L_min = QLabel(self.win)
        self.L_max = QLabel(self.win)
        self.L_min.setText(str(self.minimum))
        self.L_max.setText(str(self.maximum))
        self.layout.addWidget(self.L_min, 1, 0, alignment=Qt.AlignLeft)
        self.layout.addWidget(self.L_max, 1, 2, alignment=Qt.AlignRight)

        # Base QSlider
        self.slider = QSlider(Qt.Horizontal, self.win)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.slider_values)-1)
        self.layout.addWidget(self.slider, 1, 1)
        self.slider.valueChanged.connect(self._set_label_current)

        # Set a minimum width for this slider
        self.slider.setMinimumWidth(self.min_width)

        # Set the initial value
        midx = (np.argmin(np.abs(self.slider_values - self.init_value)))
        self.slider.setValue(midx)

        # Update the current label
        self._set_label_current()

    def setMax(self, value):
        """
        Sets the maximum value of the slider and refreshes it
        :param value: Maximum slider value
        :return:
        """
        self.slider.setMaximum(value)
        self.maximum = value
        self.slider_values = self._set_slider_values(self.minimum,
            self.maximum, self.interval)
        self.maximum = self.slider_values.max()
        self.L_max.setText(str(self.maximum))


    def setValue(self, value):
        """
        Set the current value of the slider. (This is in terms of the
        units that are shown to the user, rather than the units of the
        underlying QSlider object.)

        """
        if value < self.minimum or value > self.maximum:
            raise RuntimeError("guiUtils.IntSlider.setValue: value {} is outside " \
                "acceptable range ({}, {})".format(value, self.minimum, self.maximum))
        midx = np.argmin(np.abs(self.slider_values - value))
        self.slider.setValue(midx)

    def sizeHint(self):
        """
        Recommended size of this QWidget to the
        Qt overlords.

        """
        return QSize(125, 50)

    def _set_label_current(self):
        """
        Update the slider title with the current value of
        self.slider.

        """
        self.L_title.setText("%s: %d" % (self.name, self.value()))

    def value(self):
        """
        Return the current value of the slider as an integer.

        """
        return self.slider_values[self.slider.value()]

    def assign_callback(self, func):
        """
        Trigger a function to be called when the slider is changed.
        If several functions are assigned by sequential uses of this
        method, then all of the functions are executed when the slider
        is changed.

        args
        ----
            func        :   function, no arguments

        """
        self.slider.valueChanged.connect(func)

    def hide(self):
        """
        Hide this IntSlider.

        """
        self.win.hide()

    def show(self):
        """
        Show this IntSlider.

        """
        self.win.show()

    def isVisible(self):
        """
        Returns False if the IntSlider is currently hidden.

        """
        return self.win.isVisible()

    def toggle_vis(self):
        """
        Toggle the visibility of this IntSlider between hidden
        and shown.

        """
        if self.isVisible():
            self.hide()
        else:
            self.show()

    def configure(self, **kwargs):
        """
        Change some or all of the slider attributes. Accepted
        kwargs are:

            minimum     :   int, the minimum value of the slider
            maximum     :   int, the maximum value of the slider
            interval    :   int, the slider interval
            name        :   str, the slider label
            init_value  :   int, the initial value

        """
        keys = kwargs.keys()

        # Reconfigure the slider values
        if 'minimum' in keys:
            minimum = int(kwargs.get('minimum'))
        else:
            minimum = self.minimum
        if 'maximum' in keys:
            maximum = int(kwargs.get('maximum'))
        else:
            maximum = self.maximum
        if 'interval' in keys:
            interval = int(kwargs.get('interval'))
        else:
            interval = self.interval

        self.slider_values = self._set_slider_values(minimum,
            maximum, interval)
        self.maximum = self.slider_values.max()
        self.minimum = minimum
        self.interval = interval

        # Update the QSlider
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.slider_values)-1)

        # Set name
        if 'name' in keys:
            self.name = kwargs.get('name')

        # Set initial value
        if 'init_value' in keys:
            midx = (np.argmin(np.abs(self.slider_values - \
                kwargs.get('init_value'))))
            self.slider.setValue(midx)

        # Update labels
        self.L_min.setText(str(self.minimum))
        self.L_max.setText(str(self.maximum))
        self._set_label_current()

class SingleImageWindow(QWidget):
    """
    A standalone window containing a single ImageView
    showing a static image.

    init
    ----
        image       :   2D ndarray (YX)
        title       :   str, window title
        parent      :   root QWidget; must be passed to
                        be visible

    """
    def __init__(self, image, title=None, parent=None):
        super(SingleImageWindow, self).__init__(parent=parent)
        self.image = image
        self.title = title
        self.initUI()

    def initUI(self):
        self.ImageView = ImageView()
        self.ImageView.setImage(self.image)
        if not (self.title is None):
            self.ImageView.setWindowTitle(self.title)
        self.ImageView.show()

class LabeledQComboBox(QWidget):
    """
    A QComboBox with a QLabel above it. Useful to indicate
    the title of a variable represented by this QComboBox.

    init
    ----
        parent          :   root QWidget
        options         :   list of str, values for the QComboBox
        label           :   str, the title above the box
        init_value      :   str, starting value

    """
    def __init__(self, options, label, init_value=None, parent=None):
        super(LabeledQComboBox, self).__init__(parent)
        self.options = options
        self.label_text = label
        self.init_value = init_value
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)
        self.QComboBox = QComboBox(parent=self)
        self.QComboBox.addItems(self.options)
        self.label_widget = QLabel(parent=self)
        self.label_widget.setText(self.label_text)
        self.layout.addWidget(self.label_widget, 0, alignment=Qt.AlignTop)
        self.layout.addWidget(self.QComboBox, 1, alignment=Qt.AlignTop)

        if not self.init_value is None:
            self.QComboBox.setCurrentText(self.init_value)

    def sizeHint(self):
        """
        Recommended size of this QWidget to the
        Qt overlords.

        """
        return QSize(200, 70)

    def currentText(self):
        return self.QComboBox.currentText()

    def setCurrentText(self, *args, **kwargs):
        self.QComboBox.setCurrentText(*args, **kwargs)

    def setLabel(self, text):
        self.label_widget.setText(str(text))

    def assign_callback(self, func):
        self.QComboBox.activated.connect(func)

def coerce_type(arg, type_):
    """
    Try to coerce a string into the type class *type_*.
    When this fails, a ValueError will be raised.

    args
    ----
        arg     :   str
        type_   :   a type class, like int or float

    returns
    -------
        type_, if arg can be coerced, or *arg* if
            coercion fails

    """
    if type_ is int:
        return int(arg)
    elif type_ is float or type_ is np.float64:
        return float(arg)
    elif type_ is bool:
        return bool(arg)
    elif type_ is str:
        return str(arg)