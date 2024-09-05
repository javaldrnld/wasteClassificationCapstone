"""usage: partition_dataset.py [-h] [-i IMAGEDIR] [-o OUTPUTDIR] [-train TRAINRATIO] [-test TESTRATIO] [-valid VALIDRATIO] [-x]

Partition dataset of images into training, testing and validation sets

optional arguments:
    -h, --help            show this help message and exit
    -i IMAGEDIR, --imagedir IMAGEDIR
                    Path to the folder where the image dataset is stored. If not specified, the CWD will be used.
    -o OUTPUTDIR, --outputdir OUTPUTDIR
                    Path to the output folder where the train, test and validation dirs should be created. Defaults to the same directory as the IMAGEDIR.
    -train TRAINRATIO, --trainratio TRAINRATIO
                    The ratio of the dataset to be used for training. Defaults to 0.8
    -test TESTRATIO, --testratio TESTRATIO
                    The ratio of the dataset to be used for testing. Defaults to 0.1
    -valid VALIDRATIO, --validratio VALIDRATIO
                    The ratio of the dataset to be used for validation. Defaults to 0.1
    -x, --xml        Sett this flag if the dataset contains xml files with annotations. If set, the xml files will be copied to the output directory as well.

"""

import argparse
import math
import os
import random
import re
from shutil import copyfile


def iterate_dir(
    source: str | os.PathLike[str],
    dest: str | os.PathLike[str],
    rat_train: float,
    rat_test: float,
    rat_valid: float,
    copy_xml: bool,
):
    source = source.replace("\\", "/")
    dest = dest.replace("\\", "/")
    train_dir = os.path.join(dest, "train")
    test_dir = os.path.join(dest, "test")
    valid_dir = os.path.join(dest, "valid")

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)

    file_pattern = re.compile(r"(?i)([a-zA-Z0-9\s_\\.\-():])+(\.jpg|\.jpeg|\.png)$")

    images = [file for file in os.listdir(source) if file_pattern.search(file)]

    num_images = len(images)
    num_test_images = math.ceil(num_images * rat_test)
    num_train_images = math.ceil(num_images * rat_train)
    num_valid_images = math.ceil(num_images * rat_valid)

    for i in range(num_train_images):
        idx = random.randint(0, len(images) - 1)
        filename = images[idx]
        copyfile(os.path.join(source, filename), os.path.join(train_dir, filename))

        if copy_xml:
            xml_filename = os.path.splitext(filename)[0] + ".xml"
            copyfile(
                os.path.join(source, xml_filename),
                os.path.join(train_dir, xml_filename),
            )
        images.remove(images[idx])

    for i in range(num_test_images):
        idx = random.randint(0, len(images) - 1)
        filename = images[idx]
        copyfile(os.path.join(source, filename), os.path.join(test_dir, filename))

        if copy_xml:
            xml_filename = os.path.splitext(filename)[0] + ".xml"
            copyfile(
                os.path.join(source, xml_filename),
                os.path.join(test_dir, xml_filename),
            )
        images.remove(images[idx])

    for i in range(num_valid_images):
        idx = random.randint(0, len(images) - 1)
        filename = images[idx]
        copyfile(os.path.join(source, filename), os.path.join(valid_dir, filename))

        if copy_xml:
            xml_filename = os.path.splitext(filename)[0] + ".xml"
            copyfile(
                os.path.join(source, xml_filename),
                os.path.join(valid_dir, xml_filename),
            )
        images.remove(images[idx])


def main() -> None:
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Partition dataset of images into training, testing and validation sets",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--imageDir",
        help="Path to the folder where the image dataset is stored. If not specified, the CWD will be used.",
        type=str,
        default=os.getcwd(),
    )
    parser.add_argument(
        "-o",
        "--outputDir",
        help="Path to the output folder where the train, test and validation dirs should be created. Defaults to the same directory as the IMAGEDIR.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-train",
        "--trainRatio",
        help="The ratio of the dataset to be used for training. Defaults to 0.8",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "-test",
        "--testRatio",
        help="The ratio of the dataset to be used for testing. Defaults to 0.1",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "-valid",
        "--validRatio",
        help="The ratio of the dataset to be used for validation. Defaults to 0.1",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "-x",
        "--xml",
        help="Sett this flag if the dataset contains xml files with annotations. If set, the xml files will be copied to the output directory as well.",
        action="store_true",
    )

    args = parser.parse_args()

    if args.outputDir is None:
        args.outputDir = args.imageDir

    iterate_dir(
        args.imageDir,
        args.outputDir,
        args.trainRatio,
        args.testRatio,
        args.validRatio,
        args.xml,
    )


if __name__ == "__main__":
    main()
