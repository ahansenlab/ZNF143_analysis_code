"""
frapFile.py - frap file class
"""
from xml.etree import ElementTree as ETree

class FRAPFile:
    """
    This is a class for saving and loading custom xml files with extension
    .frap
    """
    def __init__(self, frap_img):
        """
        Initializes an instance of a frap-file class
        :param frap_img: object of class ImageReader
        """
        super().__init__()
        self.frap_img = frap_img
        self.name = frap_img.name
        self.path = frap_img.path
        self.FI = ETree.Element('FRAPImage')
        self.FI = ETree.SubElement(self.FI, 'FRAPImage')
        self.data = ETree.SubElement(self.FI, 'data')

        # Data
        self.time_points = ETree.SubElement(self.data, 'time_data')
        for tp in self.frap_img.frame_data[:, 1]:
            time = ETree.SubElement(self.time_points, 'time')
            time.text = str(tp)

        self.raw_intensity_data = ETree.SubElement(self.data, 'raw_intensity_data')
        for dp in self.frap_img.raw_mean_intensity_data:
            intensity = ETree.SubElement(self.raw_intensity_data, 'intensity')
            intensity.text = str(dp)

        self.intensity_data = ETree.SubElement(self.data, 'intensity_data')
        self.set_intensity_data()

        # Nonbleached intensity data data
        self.nonbleach_data = ETree.SubElement(self.data, 'nonbleach_data')
        self.nonbleach_intensities = ETree.SubElement(self.nonbleach_data, 'nonbleach_intensities')
        self.corrected_nonbleach_intensities = ETree.SubElement(self.nonbleach_data, 'corrected_nonbleach_intensities')
        self.set_nonbleach_data()

        # Keyframes
        self.keyframe_info = ETree.SubElement(self.FI, 'kf_info')

        self.roi_coords = ETree.SubElement(self.keyframe_info, 'roi_coords')
        x_coord = ETree.SubElement(self.roi_coords, 'x_coord')
        x_coord.text = str(self.frap_img.roi_coords[0])
        y_coord = ETree.SubElement(self.roi_coords, 'y_coord')
        y_coord.text = str(self.frap_img.roi_coords[1])

        self.roi_radii = ETree.SubElement(self.keyframe_info, 'roi_radii')
        x_rad = ETree.SubElement(self.roi_radii, 'x_rad')
        x_rad.text = str(self.roi_radii[0])
        y_rad = ETree.SubElement(self.roi_radii, 'y_rad')
        y_rad.text = str(self.roi_radii[1])

        self.keyframe_data = ETree.SubElement(self.keyframe_info, 'kf_data')
        self.roi_viewer_coords = ETree.SubElement(self.keyframe_info, 'roi_viewer_coords')
        self.set_keyframe_data()

        # Processing
        self.processing_info = ETree.SubElement(self.FI, 'processing_info')
        self.photobleaching_params = ETree.SubElement(self.processing_info, 'photobleaching_params')
        self.normalization_method = ETree.SubElement(self.processing_info, 'normalization_method')
        self.prebleach_ss = ETree.SubElement(self.processing_info, 'prebleach_ss')
        self.set_processing_info()

        # Other useful metrics
        self.other_metrics = ETree.SubElement(self.FI, 'other_metrics')
        self.nuclear_radius = ETree.SubElement(self.other_metrics, 'nuclear_radius')
        self.nuclear_radius.text = str(self.frap_img.nuclear_radius)
        self.gap_ratio = ETree.SubElement(self.other_metrics, 'gap_ratio')
        self.bleaching_depth = ETree.SubElement(self.other_metrics, 'bleaching_depth')
        self.bleach_profile_popt = ETree.SubElement(self.other_metrics, 'bleach_profile_popt')
        for p in self.frap_img.bleach_profile_popt:
            param = ETree.SubElement(self.bleach_profile_popt, 'param')
            param.text = str(p)
        self.bleach_profile_pcov = ETree.SubElement(self.other_metrics, 'bleach_profile_pcov')
        self.bleach_profile_pcov.text = str(self.frap_img.bleach_profile_pcov)
        self.bleach_distances = ETree.SubElement(self.other_metrics, 'bleach_distances')
        for d in self.frap_img.bleach_distances:
            distance = ETree.SubElement(self.bleach_distances, 'distance')
            distance.text = str(d)
        self.bleach_profile = ETree.SubElement(self.other_metrics, 'bleach_profile')
        for v in self.frap_img.bleach_profile:
            value = ETree.SubElement(self.bleach_profile, 'value')
            value.text = str(v)
        self.set_other_metrics()

        # Segmentation params
        self.segmentation_params = ETree.SubElement(self.FI, 'segmentation_params')
        self.kernel_size_param = ETree.SubElement(self.segmentation_params, 'kernel_size')
        self.kernel_size_param.text = str(self.frap_img.get_kernel_size())
        self.threshold_param = ETree.SubElement(self.segmentation_params, 'threshold')
        self.threshold_param.text = str(self.frap_img.get_threshold())


    def set_intensity_data(self):
        """
        Saves intensity data
        :return:
        """
        self.data.remove(self.intensity_data)
        # Save intensity data
        self.intensity_data = ETree.SubElement(self.data, 'intensity_data')
        for dp in self.frap_img.mean_intensity_data:
            intensity = ETree.SubElement(self.intensity_data, 'intensity')
            intensity.text = str(dp)

    def set_keyframe_data(self):
        """
        Saves keyframe information
        :return:
        """
        # Save keyframe information
        self.keyframe_info.remove(self.keyframe_data)
        self.keyframe_info.remove(self.roi_viewer_coords)

        self.keyframe_data = ETree.SubElement(self.keyframe_info, 'kf_data')
        for kf_key in self.frap_img.keyframes.keys():
            keyframe_set = ETree.SubElement(self.keyframe_data, kf_key)
            for kf in self.frap_img.keyframes[kf_key]:
                keyframe = ETree.SubElement(keyframe_set, 'keyframe')

                # frame
                frame = ETree.SubElement(keyframe, 'frame')
                frame.text = str(kf)

                # coords
                coords = self.frap_img.keyframes[kf_key].get(kf)
                x_coord = ETree.SubElement(keyframe, 'x_coord')
                x_coord.text = str(coords[0])
                y_coord = ETree.SubElement(keyframe, 'y_coord')
                y_coord.text = str(coords[1])

        self.roi_viewer_coords = ETree.SubElement(self.keyframe_info, 'roi_viewer_coords')
        for coord in self.frap_img.roi_viewer_coords:
            coordinate = ETree.SubElement(self.roi_viewer_coords, 'coord')
            x_coord = ETree.SubElement(coordinate, 'x_coord')
            x_coord.text = str(coord[0])
            y_coord = ETree.SubElement(coordinate, 'y_coord')
            y_coord.text = str(coord[1])

    def set_processing_info(self):
        """
        Sets the processing information
        :return:
        """
        self.processing_info.remove(self.photobleaching_params)
        self.processing_info.remove(self.normalization_method)
        self.processing_info.remove(self.prebleach_ss)
        # Save processing information
        self.photobleaching_params = ETree.SubElement(self.processing_info, 'photobleaching_params')
        if self.frap_img.photobleaching_params is not None:
            for param in self.frap_img.photobleaching_params:
                parameter = ETree.SubElement(self.photobleaching_params, 'param')
                parameter.text = str(param)
        else:
            parameter = ETree.SubElement(self.photobleaching_params, 'param')
            parameter.text = 'None'

        self.normalization_method = ETree.SubElement(self.processing_info, 'normalization_method')
        self.normalization_method.text = str(self.frap_img.normal_method)
        self.prebleach_ss = ETree.SubElement(self.processing_info, 'prebleach_ss')
        self.prebleach_ss.text = str(self.frap_img.prebleach_ss)
        self.set_nonbleach_data()

    def set_nonbleach_data(self):
        """
        Sets the nonbleach intensities
        :return:
        """
        self.nonbleach_data.remove(self.nonbleach_intensities)
        self.nonbleach_intensities = ETree.SubElement(self.nonbleach_data, 'nonbleach_intensities')
        if self.frap_img.nonbleach_intensities is not None:
            for inten in self.frap_img.nonbleach_intensities:
                intensity = ETree.SubElement(self.nonbleach_intensities,  'intensity')
                intensity.text = str(inten)
        else:
            intensity = ETree.SubElement(self.nonbleach_intensities, 'intensity')
            intensity.text = 'None'

        self.corrected_nonbleach_intensities = ETree.SubElement(self.nonbleach_data, 'corrected_nonbleach_intensities')
        if self.frap_img.corrected_nonbleach_intensities is not None:
            for inten in self.frap_img.corrected_nonbleach_intensities:
                intensity = ETree.SubElement(self.corrected_nonbleach_intensities,  'intensity')
                intensity.text = str(inten)
        else:
            intensity = ETree.SubElement(self.corrected_nonbleach_intensities, 'intensity')
            intensity.text = 'None'

    def set_other_metrics(self):
        """
        Sets other metrics (gap ratio and bleaching depth)
        :return:
        """
        self.other_metrics.remove(self.gap_ratio)
        self.other_metrics.remove(self.bleaching_depth)
        self.gap_ratio = ETree.SubElement(self.other_metrics, 'gap_ratio')
        self.gap_ratio.text = str(self.frap_img.get_gap_ratio())
        self.bleaching_depth = ETree.SubElement(self.other_metrics, 'bleaching_depth')
        self.bleaching_depth.text = str(self.frap_img.get_bleaching_depth())

    def set_segmentation_params(self):
        """
        Sets segentation parameters t and k
        """
        # Segmentation params
        self.segmentation_params.remove(self.kernel_size_param)
        self.segmentation_params.remove(self.threshold_param)
        self.kernel_size_param = ETree.SubElement(self.segmentation_params, 'kernel_size')
        self.kernel_size_param.text = str(self.frap_img.get_kernel_size())
        self.threshold_param = ETree.SubElement(self.segmentation_params, 'threshold')
        self.threshold_param.text = str(self.frap_img.get_threshold())


    def update(self):
        """
        Updater for the class. Calls all other functions

        :return:
        """

        self.set_other_metrics()
        self.set_nonbleach_data()
        self.set_processing_info()
        self.set_intensity_data()
        self.set_keyframe_data()
        self.set_segmentation_params()

    def save_file(self, path):
        """
        Saves a .frap file at the given path
        :param path: a path
        :return:
        """
        # create file with results
        tree = ETree.ElementTree(self.FI)
        tree.write(path, encoding ='utf-8', xml_declaration = True)
