# -*- coding: utf-8 -*-
""" Image Utilities module.
This module is in charge of providing utilities for importing images.

"""
import itertools
import os

import numpy as np
from scipy import ndimage as im


def get_subjects_dictionary(path_to_subjects, extension):
    """Returns a dictionary of subject's list of images (represented with numpy's ndarrays) with the given extension.

    Args:
        path_to_subjects (str): Path to the directory holding the subdirectory with images of each subject.
        extension (str): Extension to be used to select the images (for example, "bmp").
    Returns:
        A dictionary holding subjects (subject subdirectory name)
        together with a list of images (represented as numpy's ndarrays) of the given subject.
    Raises:
        ValueError: If there is a mismatch between the amount of images in each of the subdirectories
        of the given path_to_subjects (e.g one has 9 and another one has 10),
        or if there is a mismatch between images sizes (all must be the same size).
        IOError: If there is any IO error while reading image files.
    """
    subjects_images_paths = _get_paths_array(os.path.abspath(path_to_subjects), extension)

    all_subjects = dict(itertools.izip(subjects_images_paths.keys(),
                                       map(lambda subject: map(lambda image: _to_array(image), subject),
                                           subjects_images_paths.values())))
    _validate(all_subjects)  # Will throw ValueError if the list is not valid TODO: check exception type to use
    return all_subjects


def _get_paths_array(path_to_subjects, extension):
    """Returns a dictionary of subject's list of images paths with the given extension.
    Note: The directory with the given path_to_subjects must contain subdirectories (one per subject),
    in which the images of each of them are held.

    Args:
        path_to_subjects (str): Path to the directory holding the subdirectory with images of each subject.
        extension (str): Extension to be used to select the images (for example, "bmp").
    Returns:
        dict: A dictionary holding subjects (subject subdirectory name)
        together with the paths of the given subject's images.
    """
    subjects_directory_names = filter(lambda subject_images_dir: os.path.isdir(os.path.join(path_to_subjects,
                                                                                            subject_images_dir)),
                                      os.listdir(path_to_subjects))
    # Filter directory (get only subdirectories of the given path_to_subjects
    subjects_directories = map(lambda subject_images_dir: os.path.join(path_to_subjects, subject_images_dir),
                               subjects_directory_names)
    # Filter data-set (get only those files whose extension is the given one)
    subjects_images = map(lambda images_list:
                          filter(lambda image_file:
                                 os.path.splitext(image_file)[1] == "." + extension, images_list),
                          map(lambda subject_images_dir:
                              map(lambda image_file: os.path.join(path_to_subjects, subject_images_dir, image_file),
                                  os.listdir(subject_images_dir)), subjects_directories))

    return dict(itertools.izip(subjects_directory_names, subjects_images))


def _to_array(path_to_image):
    """ Transforms a path to an image file into an ndarray with the image's shape.

    Args:
        path_to_image (str): Path to the image to be transformed.
    Returns:
        ndarray: The result array of applying the transformation.
    """
    return im.imread(path_to_image) / 255.0  # Read image


def _validate(all_subjects_dict):
    """ Validates the given all_subjects list, which must hold a list of images represented by numpy's ndarray.
    All lists of images must be of the same size, and all images of the same shape.
    Args:
        all_subjects_dict (dict): A dictionary holding each subject's list of images.

    Raises:
        ValueError: If the list of all_subjects is not valid.
    """
    # First, check type
    if all_subjects_dict is None or not isinstance(all_subjects_dict, dict):
        raise ValueError("The subject's images argument must be a list")
    if not all_subjects_dict:
        return  # If the list is empty, return  TODO: return or raise exception? Maybe check this later in the execution
    all_subjects = all_subjects_dict.values()

    # Check each element in the list is a list
    if filter(lambda list_of_subject_images: not isinstance(list_of_subject_images, list), all_subjects):
        # TODO: define custom exception?
        raise ValueError("There is one or more elements in the list of lists of images per subject that is not a list")

    # Check each list of images of each subject has the same size
    amount_of_images_per_subject = len(all_subjects[0])
    if filter(lambda subject_images: len(subject_images) != amount_of_images_per_subject, all_subjects):
        # TODO: define custom exception?
        raise ValueError("Mismatch between sizes of list of images")
    if amount_of_images_per_subject == 0:
        return  # If all lists are empty, return
        # TODO: return or raise exception? Maybe check this later in the execution

    # Check each image is represented by an ndarray
    if filter(lambda list_of_subject_images:
              filter(lambda image_array:
                     not isinstance(image_array, np.ndarray), list_of_subject_images), all_subjects):
        # TODO: define custom exception?
        raise ValueError("There is one or more elements that is not an ndarray (representation of image)")

    # Check each image has the same shape
    image_shape = all_subjects[0][0].shape
    if filter(lambda subject_images:
              filter(lambda image_array: image_array.shape != image_shape, subject_images), all_subjects):
        # TODO: define custom exception?
        raise ValueError("Mismatch between image shapes")
