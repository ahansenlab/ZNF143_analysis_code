"""
ImageReader.py - Script that reads in images and relevant metadata
"""
import numpy as np
import pandas as pd
from .ProcessingUtilities import estimateRadius, computeBleachingProfile, \
    fitBleachingProfile, normalizeFRAPCurve, processImage, computeBestStartFrame, double_segment_full_movie, \
    computeQCMetrics
from .ReaderUtilities import make_metadata_tree, make_frame_metadata, \
    make_roi, make_dimension_metadata, read_image
from .frapFile import FRAPFile

class FRAPImage:
    """
    This is a class for storing FRAP image data and metadata.
    """

    def __init__(self, path):
        """
        Constructor for FRAPImage class

        :param path: (str) path to image
        """
        # Path and name
        self.path = path
        self.name = path.split('/')[-1]
        print('Reading metadata...')

        # Metadata
        self.metadata_tree = make_metadata_tree(self.path) # contains the original metadata for the image

        self.frame_data = make_frame_metadata(self.path)  # frame data array [index, DeltaT]
        self.shape, self.physical_size = make_dimension_metadata(self.path)
        self.xdim, self.ydim, self.tdim = self.shape
        self.roi_ind = 0
        self.roi_coords, self.roi_radii = make_roi(self.path, self.roi_ind)
        self.roi_viewer_coords = [(self.roi_coords[0] - self.roi_radii[0],
                                   self.roi_coords[1] - self.roi_radii[1]) for i in range(self.tdim)]

        self.roi_nuc_coords, self.roi_bg_coords = (30, 30), (10, 10)
        self.roi_nuc_radii, self.roi_bg_radii = (5.0, 5.0), (5.0, 5.0)
        self.roi_nuc_viewer_coords = [(self.roi_nuc_coords[0] - self.roi_nuc_radii[0],
                                       self.roi_nuc_coords[1] - self.roi_nuc_radii[1]) for i in range(self.tdim)]
        self.roi_bg_viewer_coords = [(self.roi_bg_coords[0] - self.roi_bg_radii[0],
                                       self.roi_bg_coords[1] - self.roi_bg_radii[1]) for i in range(self.tdim)]
        self.viewer_coord_dict = {"Bleach": self.roi_viewer_coords, "Nucleus": self.roi_nuc_viewer_coords,
                                  "Background": self.roi_bg_viewer_coords}
        self.radii_dict = {"Bleach": self.roi_radii, "Nucleus": self.roi_nuc_radii,
                                  "Background": self.roi_bg_radii}
        self.keyframes = {"Bleach": {0:self.roi_viewer_coords[0], self.tdim - 1:self.roi_viewer_coords[-1]},
                          "Nucleus": {0:self.roi_nuc_viewer_coords[0], self.tdim - 1:self.roi_nuc_viewer_coords[-1]},
                          "Background": {0:self.roi_bg_viewer_coords[0], self.tdim - 1:self.roi_bg_viewer_coords[-1]}}

        print('Reading image data...')
        self.image_data = read_image(self.path)
        self.raw_image_data = self.image_data.copy()
        
        # ROI Normalization Flag
        self.roi_normalization_flag = False

        self.set_mean_intensity_data(update=False)

        # Get nuclear segmentation 
        self.kernel_size = 10
        self.threshold = "otsu"
        self.segmentation_data = double_segment_full_movie(self)

        # Set photobleaching params
        self.photobleaching_params = None

        # Initialize normalization information
        self.normal_method = None
        self.prebleach_ss = None

        # Initialize cell intensity information
        self.nonbleach_intensities = None
        self.corrected_nonbleach_intensities = None

        # Estimate nuclear radius
        self.nuclear_radius = estimateRadius(self)

        # Set initial values for gap ratio and bleaching depth
        self.gap_ratio = -1
        self.bleaching_depth = -1

        # Fix start time
        self.start_frame = computeBestStartFrame(self)
        self.set_start_frame(self.start_frame)
        x_data = self.get_frame_metadata()[:, 1]
        x_data -= x_data[self.start_frame - 1]

        # Compute bleaching profile
        self.bleach_distances, self.bleach_profile = computeBleachingProfile(self)
        self.bleach_profile_popt, self.bleach_profile_pcov = fitBleachingProfile(self)

        # Attach model
        self.Model = None

        # Photobleaching correction factors
        self.correction_factors = np.ones(self.tdim)

        # FRAP File
        self.file = FRAPFile(self)
        self.file.update()
        self.calc_nonbleach_intensity_data()

    def reset_image_data(self):
        """
        Resets image data
        :return:
        """
        print('\nReading image data...')
        self.image_data = self.raw_image_data

        self.set_mean_intensity_data()

        # Get nuclear segmentation 
        self.refresh_segmentation()

        # Set photobleaching params
        self.photobleaching_params = None

        # Initialize normalization information
        self.normal_method = None
        self.prebleach_ss = None

        # Initialize cell intensity information
        self.nonbleach_intensities = None
        self.corrected_nonbleach_intensities = None
        self.file.update()
        self.calc_nonbleach_intensity_data()

    def get_tdim(self):
        """
        Getter for tdim size

        :return: (int) size along t dimension
        """
        return self.tdim

    def get_frame_metadata(self):
        """
        Getter for frame metadata

        :return: (ndarray) has frame index and DeltaT
        """
        return self.frame_data

    def get_frame(self, idx):
        """
        Getter for an individual frame's data

        :param idx: (int) the index of the desired frame
        :return: (ndarray) image data for that frame
        """
        return self.image_data[:, :, idx]

    def get_segment_frame(self, idx):
        """
        Getter for an individual frame's data

        :param idx: (int) the index of the desired frame
        :return: (ndarray) image data for that frame
        """
        return self.segmentation_data[:, :, idx]

    def get_physical_size(self):
        """
        Getter for image physical size

        :return: (tuple) real X dimension, real Y dimension
        """
        return self.physical_size

    def get_viewer_coords(self, roi_key="Bleach"):
        """
        Getter for the ROI coordinates for the viewer

        :return: Array of tuples. X coordinate for ROI, Y coordinate for ROI
        """
        return self.viewer_coord_dict.get(roi_key)

    def get_roi_coords(self):
        """
        Getter for the ROI coordinates

        :return: (tuple) X coordinate for ROI, Y coordinate for ROI
        """
        return self.roi_coords

    def get_roi_radii(self, roi_key="Bleach"):
        """
        Getter for ROI radii

        :return: (tuple) X radius, Y radius
        """
        curr_radii = self.radii_dict.get(roi_key)
        return curr_radii

    def set_roi_radii(self, radii, roi_key="Bleach"):
        """
        Setter for ROI radii
        """
        curr_radii = self.radii_dict.get(roi_key)
        if isinstance(radii, tuple):
            curr_radii = radii

        elif isinstance(radii, (float, int)):
            curr_radii = (radii, radii)

        else:
            raise ValueError("Radii must be tuple or numeric.")

        self.radii_dict[roi_key] = curr_radii
        if roi_key=="Bleach":
            self.roi_radii = curr_radii

        self.set_mean_intensity_data()

    def set_image_data(self, img):
        """
        Setter for image data

        :param img: new image data
        """
        self.image_data = img
        self.set_mean_intensity_data()
        self.file.update()

    def get_image_data(self):
        """
        Getter for image data

        :return: (ndarray) array containing image data
        """
        return self.image_data

    def set_keyframe(self, idx, viewer_coords, roi_key="Bleach"):
        """
        Sets a new keyframe

        :param idx: index of new keyframe
        :param viewer_coords: coordinates of new keyframe
        :return:
        """
        curr_keyframes = self.keyframes[roi_key]
        curr_keyframes[idx] = viewer_coords
        self.update_viewer_coords(roi_key)
        self.file.update()
        self.refresh_segmentation()

    def del_keyframe(self, idx, roi_key="Bleach"):
        """
        Deletes a keyframe

        :param idx: index of keyframe to be deleted
        :return:
        """
        curr_keyframes = self.keyframes[roi_key]
        if idx in curr_keyframes.keys():
            curr_keyframes.pop(idx)
            self.keyframes[roi_key] = curr_keyframes
            self.update_viewer_coords(roi_key)
            self.refresh_segmentation()

    def get_keyframes(self):
        """
        Returns a dictionary of keyframes

        :return: (dict) of keyframes
        """
        return self.keyframes

    def get_nucleus_keyframes(self):
        """
        Returns a dictionary of keyframes

        :return: (dict) of keyframes
        """
        curr_keyframes = self.keyframes["Nucleus"]
        return curr_keyframes

    def get_bg_keyframes(self):
        """
        Returns a dictionary of keyframes

        :return: (dict) of keyframes
        """
        ### TODO FIXXXXXXX
        curr_keyframes = self.keyframes["Background"]
        return curr_keyframes
    
    def get_nucleus_intensity_data(self):
        """
        Gets intensity data for nucleus ROI
        """
        radius = self.radii_dict.get("Nucleus")[0]
        r_coords = [(pos[0] + radius, pos[1] + radius) for pos in self.viewer_coord_dict.get("Nucleus")]
        raw_data, corr_data = self.get_roi_values(r_coords, radius)
        return raw_data, corr_data
    
    def get_bg_intensity_data(self):
        """
        Gets intensity data for bg ROI
        """
        radius = self.radii_dict.get("Background")[0]
        r_coords = [(pos[0] + radius, pos[1] + radius) for pos in self.viewer_coord_dict.get("Background")]
        raw_data, corr_data = self.get_roi_values(r_coords, radius)
        return raw_data, corr_data

    def update_viewer_coords(self, roi_key="Bleach"):
        """
        Updates viewer coords and mean intensity data based on current keyframe dictionary
        """
        curr_keyframes = self.keyframes[roi_key]
        keyvals = sorted(curr_keyframes.items())
        new_coords = []
        next_pos = None
        for i in range(len(keyvals) - 1):
            cur_frame, cur_pos = keyvals[i]
            next_frame, next_pos = keyvals[i+1]
            frame_range = next_frame-cur_frame
            new_coords += list(zip(np.linspace(cur_pos[0], next_pos[0], num=frame_range, endpoint=False),
                                   np.linspace(cur_pos[1], next_pos[1], num=frame_range, endpoint=False)))
        new_coords += [(next_pos[0], next_pos[1])]
        self.viewer_coord_dict[roi_key] = new_coords

        if roi_key == "Bleach":
            # update mean intensity data
            self.roi_viewer_coords = new_coords
            r_coords = [(pos[0] + self.roi_radii[0], pos[1] + self.roi_radii[1]) for pos in new_coords]
            self.set_mean_intensity_data(r_coords)
        self.file.update()

    def get_mean_intensity_data(self):
        """
        Getter for intensity data

        :return: (list) of mean ROI intensities
        """
        return self.mean_intensity_data

    def set_mean_intensity_data(self, r_coords = None, update=True):
        """
        Setter for mean intensities

        :param r_coords: (list) of coordinates for ROI
        """
        if r_coords is None:
            # r_coords = [(pos[0] + self.roi_radii[0], pos[1] + self.roi_radii[1]) for pos in self.roi_viewer_coords]
            r_coords = [(pos[0] + self.roi_radii[0], pos[1] + self.roi_radii[1]) for pos in self.roi_viewer_coords]
        self.raw_mean_intensity_data, self.mean_intensity_data = self.get_roi_values(r_coords, self.roi_radii[0])
        if update:
            self.file.update()

    def get_roi_values(self, roi_coords, roi_radius):
        """
        Gets values from ROI for arbitrary set of ROI coordinates:
        Assumes roi_coords is in the center of the roi
        """
        mean_intensities = np.empty(self.tdim)
        raw_mean_intensities = np.empty(self.tdim)
        for frame in range(self.tdim):
            X, Y = np.ogrid[0:self.xdim, 0:self.ydim]
            dist_from_center = np.sqrt((X - roi_coords[frame][0])**2 + \
                                       (Y - roi_coords[frame][1])**2)
            image_data = self.get_frame(frame)
            raw_image_data = self.raw_image_data[:, :, frame]
            mean_intensities[frame] = np.mean(image_data[dist_from_center <= roi_radius])
            raw_mean_intensities[frame] = np.mean(raw_image_data[dist_from_center <= roi_radius])

        return raw_mean_intensities, mean_intensities

    def normalize_frap_curve(self, method, prebleach_ss):
        """
        Normalizes FRAP curve based on the method given
        :param method: method to perform normalization
        :return:
        """
        self.normal_method = method
        self.prebleach_ss = prebleach_ss
        # Fix time 0
        if method == 'Fullscale':
            x_data = self.get_frame_metadata()[:, 1]
            x_data -= x_data[self.start_frame]
        else:
            x_data = self.get_frame_metadata()[:, 1]
            x_data -= x_data[self.start_frame - 1]

        # Do normalization
        self.set_mean_intensity_data()

        self.mean_intensity_data = normalizeFRAPCurve(self.mean_intensity_data, self.start_frame,
                                                      method, prebleach_ss)
        self.set_corrected_nonbleach_intensities(normalizeFRAPCurve(self.corrected_nonbleach_intensities, self.start_frame,
                                                      method, prebleach_ss))
        print(self.mean_intensity_data)
        self.file.update()

    def correct_photobleaching(self, subtract_bg):
        """
        Corrects photobleaching
        :param subtract_bg: whether to perform background subtraction as well
        :return:
        """
        print('\nProcessing image...')
        self.photobleaching_params = processImage(self, self.roi_normalization_flag, subtract_bg)
        self.file.update()

    def get_time_intensity_pt(self, idx):
        """
        Get a time intensity value pair

        :param idx: (int) desired frame
        :return: (tuple of floats) time intensity value pair
        """
        return self.frame_data[idx,1], self.mean_intensity_data[idx]

    def attach_model(self, class_name, start_frame = None):
        """
        Attaches a FRAP recovery model to the FRAPImage class instance

        :param class_name: (class) a class name
        :param start_frame: (int or none) frame index of photobleaching
        """
        if start_frame is None:
            start_frame = self.start_frame
        self.Model = class_name(self, start_frame)
        self.Model.fit()
        self.ModelParams = (self.Model.get_parameters(), self.Model.get_cov())
        self.ModelFun = self.Model.func
        self.ModelData = self.Model.get_fit_pts()

    def get_start_frame(self):
        """
        Getter for start_frame

        :return: start_frame
        """
        return self.start_frame

    def set_start_frame(self, new_start_frame):
        """
        Setter for start_frame

        :return:
        """
        self.start_frame = new_start_frame
        # Recompute bleaching profile
        self.bleach_distances, self.bleach_profile = computeBleachingProfile(self)
        self.bleach_profile_popt, self.bleach_profile_pcov = fitBleachingProfile(self)

    def reset_start_frame(self):
        """
        Resets start frame to argmin

        :return: new start_frame
        """
        self.start_frame = np.argmin(self.get_mean_intensity_data())
        return self.start_frame


    def get_model_params(self):
        """
        Getter for model parameters

        :return: model parameters
        """
        return self.ModelParams

    def get_model_fun(self):
        """
        Getter for model function

        :return: model function
        """
        return self.ModelFun

    def get_model_data(self):
        """
        Getter for model data

        :return: Model evaluated at timepoints
        """
        return self.ModelData

    def get_photobleaching_params(self):
        """
        Getter for photobleaching fit

        :return: optimal parameters for photobleaching fit
        """
        return self.photobleaching_params

    def get_gap_ratio(self):
        """
        Getter for the gap ratio

        :return: gap ratio of the experiment
        """
        return self.gap_ratio

    def set_gap_ratio(self, gr):
        """
        Setter for the gap ratio

        :param gr: new gap ratio
        :return:
        """
        self.gap_ratio = gr
        self.file.set_other_metrics()

    def get_bleaching_depth(self):
        """
        Getter for the bleaching depth

        :return: bleaching depth of the experiment
        """
        return self.bleaching_depth

    def set_bleaching_depth(self, bd):
        """
        Setter for the bleaching depth

        :param bd: new bleaching depth value
        :return:
        """
        self.bleaching_depth = bd
        self.file.set_other_metrics()

    def detach_model(self):
        """
        Detaches the FRAP recovery model
        :return:
        """
        self.Model = None
        self.ModelParams = None
        self.ModelFun = None
        self.ModelData = None

    def get_nonbleach_intensities(self):
        """
        Getter for nonbleach intensities
        :return:
        """
        return self.nonbleach_intensities

    def set_nonbleach_intensities(self, intensities):
        """
        Sets the intensity values for the nonbleached areas
        :param intensities: a list of intensity values
        :return:
        """
        self.nonbleach_intensities = intensities
        self.file.update()

    def get_corrected_nonbleach_intensities(self):
        """
        Getter for corrected nonbleach intensities
        :return:
        """
        return self.corrected_nonbleach_intensities

    def set_corrected_nonbleach_intensities(self, intensities):
        """
        Sets the intensity values for the corrected nonbleached areas
        :param intensities: a list of intensity values
        :return:
        """
        self.corrected_nonbleach_intensities = intensities
        self.file.update()

    def get_nuclear_radius(self):
        """
        Getter for the nuclear radius
        :return: nuclear radius in um
        """
        return self.nuclear_radius

    def get_real_roi_radius(self):
        """
        Returns ROI radius in real units
        :return:
        """
        return self.roi_radii[0] * float(self.physical_size[0]) 

    def get_bleaching_profile(self):
        """
        Returns bleaching distances and values
        :return:
        """
        return self.bleach_distances, self.bleach_profile

    def save_data(self, path):
        """
        Saves data in custom xml format for later use/analysis
        :param path: (str) location of file save
        :return:
        """
        self.file.save_file(path)

    def export_to_csv(self, path):
        """
        Exports intensity data as a csv without metdata
        :param path: (str) location of file save
        :return:
        """
        full_data = pd.DataFrame({'time': self.frame_data[:, 1],
                                  'intensity': self.mean_intensity_data,
                                  'intensity_raw':self.raw_mean_intensity_data,
                                  'nonbleach':self.nonbleach_intensities,
                                  'corrected_nonbleach':self.corrected_nonbleach_intensities})
        individual_params = pd.DataFrame({'nuclear_radius':self.nuclear_radius,
                                          'roi_radius':self.get_real_roi_radius(),
                                          'gap_ratio':self.gap_ratio,
                                          'bleaching_depth':self.bleaching_depth,
                                          'radius_uniform':self.bleach_profile_popt[0]}, index=[0])
        bleach_profile = pd.DataFrame({'bleach_distances':self.bleach_distances,
                                       'bleach_profile':self.bleach_profile})
        full_data = pd.concat([full_data, individual_params, bleach_profile], axis=1)
        full_data.to_csv(path)

    def set_roi(self, i):
        """
        Sets the roi to the roi in the file with the given index if possible, otherwise roi 1
        TODO: Fix this! :(
        """
        self.set_roi_ind(i)
        self.roi_coords, self.roi_radii = make_roi(self.path, self.roi_ind)
        self.roi_viewer_coords = [(self.roi_coords[0] - self.roi_radii[0],
                                   self.roi_coords[1] - self.roi_radii[1]) for i in range(self.tdim)]
        self.viewer_coord_dict["Bleach"] = self.roi_viewer_coords
        self.keyframes["Bleach"] = {0:self.roi_viewer_coords[0], self.tdim - 1:self.roi_viewer_coords[-1]}
        self.update_viewer_coords()
        self.file.update()

    def set_roi_ind(self, i):
        """
        Sets roi_ind. 
        """
        self.roi_ind = i

    def get_kernel_size(self):
        """
        Gets kernel size
        """
        return self.kernel_size

    def set_kernel_size(self, k):
        """
        Sets kernel size
        """
        self.kernel_size = k
        self.refresh_segmentation()

    def get_threshold(self):
        """
        Gets kernel size
        """
        return self.threshold

    def set_threshold(self, t):
        """
        Sets kernel size
        """
        self.threshold = t
        self.refresh_segmentation()

    def refresh_segmentation(self):
        """
        Recalculates segmentation
        """
        self.segmentation_data = double_segment_full_movie(self)
        self.calc_nonbleach_intensity_data()

    def calc_nonbleach_intensity_data(self, r_coords = None, update=True):
        """
        Setter for mean intensities

        :param r_coords: (list) of coordinates for ROI
        """
        if self.roi_normalization_flag:
            nonbleach_intensities, _ = self.get_nucleus_intensity_data()
        else:
            if r_coords is None:
                r_coords = [(pos[0] + self.roi_radii[0], pos[1] + self.roi_radii[1]) for pos in self.roi_viewer_coords]
            nonbleach_intensities = np.empty(self.tdim)
            for frame in range(self.tdim):
                X, Y = np.ogrid[0:self.xdim, 0:self.ydim]
                dist_from_center = np.sqrt((X - r_coords[frame][0])**2 + \
                                           (Y - r_coords[frame][1])**2)
                raw_image_data = self.raw_image_data[:, :, frame]
                excluded_region = np.zeros(raw_image_data.shape)
                excluded_region[dist_from_center <= self.roi_radii[0]] = 2
                nonbleach_intensities[frame] = np.mean(raw_image_data[(self.segmentation_data[:, :, frame] - excluded_region) == 2])

        self.nonbleach_intensities = nonbleach_intensities
        self.set_corrected_nonbleach_intensities(nonbleach_intensities)
        if update:
            self.file.update()

    def recalculate_qc_metrics(self, prebleach_ss, subtract_bg):
        """
        Recalculates the nuclear radius, roi radius, gap ratio, bleaching depth, and uniform roi radius
        """
        self.correct_photobleaching(subtract_bg)
        self.nuclear_radius = estimateRadius(self)

        return self.nuclear_radius, self.get_real_roi_radius(), self.gap_ratio, self.bleaching_depth, self.bleach_profile_popt[0]

    def get_correction_factors(self):
        """
        Getter for correction factors variable
        """
        return self.correction_factors

    def set_correction_factors(self, corr_factors):
        """
        Setter for correction factors variable
        """
        self.correction_factors = corr_factors

    def set_roi_flag(self, use_roi):
        """
        Sets the roi flag to use_roi
        """
        self.roi_normalization_flag = use_roi

class UnknownFiletypeError(Exception):
    """
    User defined error for unrecognized filetype
    """

    def __init__(self, filetype):
        self.message = f"Extension {filetype} is not recognized."
        super().__init__(self.message)
