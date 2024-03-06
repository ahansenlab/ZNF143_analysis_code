"""
ReaderUtilities.py - Script that provides utilities to extract information from image formats
"""

import re
import os
import numpy as np
import aicsimageio as imageio
from aicsimageio import AICSImage
from aicsimageio.readers.czi_reader import CziReader
from xml.etree import ElementTree as ETree
import czifile as cz

class UnknownFiletypeError(Exception):
    """
    User defined error for unrecognized filetype
    """

    def __init__(self, filetype):
        self.message = f"Extension {filetype} is not recognized."
        super().__init__(self.message)

def read_image(path, as_dask=False):
    """
    Makes image series from path

    :param path: (str) path to image file
    :param as_dask: (bool) indicates whether to use dask array
    :return: (xml etree) xml etree object containing metadata
    """
    extension = os.path.splitext(path)[-1]
    if extension == '.czi':
        if not as_dask:
            image = imageio.AICSImage(path, reader=CziReader)
            image_data = image.get_image_data("XYT", H=0, S=0, C=0, Z=0)
            return image_data

    elif extension == ".czmbi":
        # get paths to each czi file
        tree = ETree.parse(path)
        documents = tree.find("Documents")
        image_paths = [os.path.join(os.path.dirname(path), *doc_path.text.split("\\")) for doc_path in documents]

        images = [imageio.AICSImage(image_path, reader=CziReader).get_image_data("XYT", 
            H=0, S=0, C=0, Z=0) for image_path in image_paths]

        return np.concatenate(images, axis=2)

    else:
        raise UnknownFiletypeError(extension)

def make_frame_metadata(path):
    """
    Gets frame numbers and time deltas for image

    :param path: (str) path to image file
    :return: numpy array of frame data [index, time]
    """
    extension = os.path.splitext(path)[-1]
    if extension == '.czi':
        with cz.CziFile(path) as czi:
            # find timestamps
            for attachment in czi.attachments():
                if attachment.attachment_entry.name == 'TimeStamps':
                    timestamps = attachment.data()
                    break
        return np.array([[float(i), timestamps[i]] for i in range(len(timestamps))])

    elif extension == ".czmbi":
        tree = ETree.parse(path)
        documents = tree.find("Documents")
        image_paths = [os.path.join(os.path.dirname(path), *doc_path.text.split("\\")) for doc_path in documents]

        timestamp_list = []
        frame_offset = 0.0
        for i, img_path in enumerate(image_paths):
            with cz.CziFile(img_path) as czi:
                # find timestamps
                for attachment in czi.attachments():
                    if attachment.attachment_entry.name == 'TimeStamps':
                        timestamps = attachment.data()
                        break

            curr_timestamps = np.array([[frame_offset + float(i), 
                timestamps[i]] for i in range(len(timestamps))])

            timestamp_list.append(curr_timestamps)
            frame_offset += curr_timestamps[-1, 0]
        return np.concatenate(timestamp_list, axis=0)

    else:
        raise UnknownFiletypeError(extension)

def make_dimension_metadata(path):
    """
    Extracts pixel and channel data to dictionary

    :param path: (str) path to image file
    :return: dimensions and physical sizes of image
    """
    extension = os.path.splitext(path)[-1]
    if extension == '.czi':
        img = AICSImage(path, reader=CziReader)  # selects the first scene found
        shape = img.dims['X', 'Y', 'T']
        root = make_metadata_tree(path)
        scales = root.find("Metadata/Scaling/Items")
        physical_size =[float(distance.find("Value").text) * 1000000 # convert to um
                        for distance in scales.iter("Distance")]
        return shape, tuple(physical_size)

    elif extension ==".czmbi":
        tree = ETree.parse(path)
        documents = tree.find("Documents")
        image_paths = [os.path.join(os.path.dirname(path), *doc_path.text.split("\\")) for doc_path in documents]

        img = AICSImage(image_paths[0], reader=CziReader)  # selects the first scene found
        shape = img.dims['X', 'Y', 'T']
        root = make_metadata_tree(image_paths[0])
        scales = root.find("Metadata/Scaling/Items")
        physical_size =[float(distance.find("Value").text) * 1000000 # convert to um
                        for distance in scales.iter("Distance")]

        new_shape = list(shape)
        for img_path in image_paths[1:]:
            curr_img = AICSImage(img_path, reader=CziReader)  # selects the first scene found
            curr_shape = curr_img.dims['X', 'Y', 'T']
            new_shape[2] += curr_shape[2]

        return new_shape, tuple(physical_size)

    elif extension == '.tiff':
        #TODO: IMPLEMENT FOR OTHER FILE TYPES. INTENDED TO WORK WITH OMEXML
        metadata_tree = None # Doesn't currently work
        ome_str = re.search('{(.+?)}', metadata_tree[0].tag).group()
        pixel_data = [el.attrib for el in metadata_tree.iter(ome_str + 'Pixels')]
        channel_data = [el.attrib for el in metadata_tree.iter(ome_str + 'Channel')]
        acquisition_data = {**pixel_data[0], **channel_data[0]}
        physical_size = (acquisition_data.get('PhysicalSizeX', None),
                         acquisition_data.get('PhysicalSizeY', None))  # in um
        xdim = int(acquisition_data.get('SizeX', None))
        ydim = int(acquisition_data.get('SizeY', None))
        tdim = int(acquisition_data.get('SizeT', None))
        shape = (xdim, ydim, tdim)
        return shape, physical_size

    else:
        raise UnknownFiletypeError(extension)

def make_metadata_tree(path):
    """
    Makes xml etree object from path

    :param path: (str) path to image file
    :return: (xml etree) xml etree object containing metadata
    """
    extension = os.path.splitext(path)[-1]
    if extension == '.czi':
        with cz.CziFile(path) as czi:
            metadata_tree = ETree.fromstring(czi.metadata())

    elif extension ==".czmbi":
        tree = ETree.parse(path)
        documents = tree.find("Documents")
        image_paths = [os.path.join(os.path.dirname(path), *doc_path.text.split("\\")) for doc_path in documents]
        # TODO: Fix this. Figure out a way to merge xml files from multiple czis
        with cz.CziFile(image_paths[0]) as czi:
            metadata_tree = ETree.fromstring(czi.metadata())
    else:
        raise UnknownFiletypeError(extension)

    return metadata_tree

def make_roi(path, i):
    """
    Gets ROI coordinates and radii from metadata tree

    :param path: (str) path to image file
    :param i: (int) index of the roi to load
    :return: tuple of coordinates and tuple of radii
    """
    extension = os.path.splitext(path)[-1]

    if extension == '.czi':
        root = make_metadata_tree(path)

        ROI_coords = []
        ROI_radii = []
        for layer in root.iter("Layers"):
            for circle in layer.iter("Circle"):
                xCoord = circle.find("Geometry/CenterX").text
                yCoord = circle.find("Geometry/CenterY").text
                radius = circle.find("Geometry/Radius").text
                ROI_coords.append((float(xCoord), float(yCoord)))
                ROI_radii.append((float(radius), float(radius))) # OME XML specifies in terms
                                                   # of ellipse so both X and Y radii are given
        try:
            return ROI_coords[i], ROI_radii[i]
        except:
            return ROI_coords[0], ROI_radii[0]

    elif extension == '.czmbi':
        tree = ETree.parse(path)
        documents = tree.find("Documents")
        image_paths = [os.path.join(os.path.dirname(path), *doc_path.text.split("\\")) for doc_path in documents]

        ROI_coords = []
        ROI_radii = []

        for img_path in image_paths:
            root = make_metadata_tree(img_path)
            for layer in root.iter("Layers"):
                for circle in layer.iter("Circle"):
                    xCoord = circle.find("Geometry/CenterX").text
                    yCoord = circle.find("Geometry/CenterY").text
                    radius = circle.find("Geometry/Radius").text
                    ROI_coords.append((float(xCoord), float(yCoord)))
                    ROI_radii.append((float(radius), float(radius))) # OME XML specifies in terms
                                                       # of ellipse so both X and Y radii are given
        try:
            return ROI_coords[i], ROI_radii[i]
        except:
            return ROI_coords[0], ROI_radii[0]

    elif extension == '.tiff':
        #TODO: IMPLEMENT FOR OTHER FILE TYPES
        # TODO: FIX FOR LOADING ROI WITH INDEX I
        metadata_tree = None # Doesn't currently work
        ome_str = re.search('{(.+?)}', metadata_tree[0].tag).group()
        roi_data = [el.attrib for el in metadata_tree.iter(ome_str + 'Ellipse')]
        roi_coords = (abs(float(roi_data[0].get('X', None))),
                      abs(float(roi_data[0].get('Y', None))))
        roi_radii = (float(roi_data[0].get('RadiusX', None)),
                     float(roi_data[0].get('RadiusY', None)))
        return roi_coords, roi_radii

    else:
        raise UnknownFiletypeError(extension)
