# -*- coding: utf-8 -*-
import argparse
import logging
import sys

import image_utils
from pca_recognition import PCARecognizer
from face_shazam import __version__

_logger = logging.getLogger(__name__)


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    # Basic arguments
    parser = argparse.ArgumentParser(
        description="Face Shazam")
    parser.add_argument(
        '-V',
        '--version',
        action='version',
        version='face_shazam {ver}.'.format(ver=__version__))
    parser.add_argument(
        '-v',
        '--verbose',
        dest="log_level",
        help="Set log level to INFO.",
        action='store_const',
        const=logging.INFO)
    parser.add_argument(
        '-vv',
        '--very-verbose',
        dest="log_level",
        help="Set log level to DEBUG.",
        action='store_const',
        const=logging.DEBUG)

    # Image arguments
    parser.add_argument(
        "-s",
        "--subjects-path",
        dest="path_to_subjects",
        help="Set the path where the subjects images are. "
             "This path must contain subdirectories where each one has the images of a given subject.",
        default="./subjects",
        action="store"
    )
    parser.add_argument(
        "-ext",
        "--image-extension",
        dest="image_extension",
        help="Set the file extension to be used for images.",
        default="pgm",
        action="store"
    )

    return parser.parse_args(args)


def setup_logging(log_level):
    """Setup basic logging

    Args:
      log_level (int): minimum loglevel for emitting messages
    """
    logging.basicConfig(level=log_level,
                        stream=sys.stdout,
                        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.log_level)
    _logger.info("Starting application...")

    try:
        all_subjects = image_utils.get_subjects_dictionary(args.path_to_subjects, args.image_extension)
    except ValueError as e:
        _logger.error("There were validation errors with the images. Exception message: {}".format(e))
        exit(1)
    except IOError as e:
        _logger.error("There were IO errors while loading the images. Exception message: {}".format(e))
        exit(1)

    # noinspection PyUnboundLocalVariable
    recognizer = PCARecognizer(all_subjects, 0.6)  # If all_subjects is not initialized, exit(1) was executed
    recognizer.train(240)
    print("The score achieved is {}%".format(recognizer.test() * 100))
    return


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
