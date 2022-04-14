"""
Mask R-CNN
Train on the Corrosion dataset

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 corrosion.py train --dataset=/path/to/corrosion/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 corrosion.py train --dataset=/path/to/corrosion/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 corrosion.py train --dataset=/path/to/corrosion/dataset --weights=imagenet

"""

import os
import sys
from pathlib import Path

import numpy as np
from xml.etree import ElementTree as ET
import skimage.draw

# Root directory of the project


ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn.model import MetricsCallback

from imgaug import augmenters as iaa
# if truncated image load not crash
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = None


############################################################
#  Configurations
############################################################

class CorrosionConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Corrosion"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + corrosion

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1217 // IMAGES_PER_GPU
    VALIDATION_STEPS = 100 // IMAGES_PER_GPU

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between corrosion and BG
    DETECTION_MIN_CONFIDENCE = 0.7

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 400

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Image mean (RGB)
    # corrected for corrosion train images (1217 images)
    MEAN_PIXEL = np.array([95.98, 91.02, 87.38])

    # smaller for avoid overfitting
    BACKBONE = "resnet50"

    # we have both small and large objects
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels


class InferenceConfig(CorrosionConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


############################################################
#  Dataset
############################################################

class CorrosionDataset(utils.Dataset):
    def load_data(self, dataset_dir, subset):
        """Load corrosion dataset
        dataset_dir: The root directory of the dataset.
        subset: What to load (train, val)
        """

        # Add classes. We have only one class to add.
        self.add_class("corrosion", 1, "corrosion")
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        images_dir = os.path.join(dataset_dir, "images")

        tree = ET.parse(Path(dataset_dir) / "annotations.xml")
        tree_root = tree.getroot()

        # Add images
        for i, im in enumerate(tree_root.findall(".//image")):
            im_name = im.attrib["name"]
            width = int(im.attrib['width'])
            height = int(im.attrib['height'])
            image_path = os.path.join(images_dir, im_name)

            # get all annotations
            annotations = im.findall(".//polyline")
            annotations.extend(im.findall(".//polygon"))

            annotations = [p for p in annotations if p.attrib['label'] == "corrosion"]

            self.add_image(
                "corrosion",
                image_id=i,
                path=image_path,
                width=width, height=height,
                annotations=annotations)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a corrosion dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "corrosion":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["annotations"])],
                        dtype=np.uint8)
        for i, ann in enumerate(info["annotations"]):
            # Get indexes of pixels inside the polygon and set them to 1
            str_points = ann.attrib['points'].split(";")
            pts = np.array([np.array(point.split(",")).astype(np.float64) for point in str_points])
            all_points_y = pts[:, 1]
            all_points_x = pts[:, 0]
            rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "corrosion":
            return info["path"]
        else:
            super(CorrosionDataset, self).image_reference(image_id)


def train(model, augment=False):
    """Train the model."""
    # Training dataset.
    dataset_train = CorrosionDataset()
    dataset_train.load_data(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CorrosionDataset()
    dataset_val.load_data(args.dataset, "val")
    dataset_val.prepare()

    augmentation = None
    if augment:
        augmentation = iaa.SomeOf((0, 2), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.Affine(rotate=90),
                       iaa.Affine(rotate=180),
                       iaa.Affine(rotate=270)]),
            iaa.Multiply((0.8, 1.5)),
            iaa.GaussianBlur(sigma=(0.0, 5.0))
        ])

    infConf = InferenceConfig()
    model_inference = modellib.MaskRCNN(mode="inference", config=infConf, model_dir=MODEL_DIR)
    mean_average_precision_callback = MetricsCallback(model, model_inference, dataset_val,
                                                      calculate_map_at_every_X_epoch=1)

    # print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                augmentation=augmentation,
                layers='heads',
                custom_callbacks=[mean_average_precision_callback])

    # print("Train all layers")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=20,
    #             augmentation=augmentation,
    #             layers='all',
    #             custom_callbacks=[mean_average_precision_callback])


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect corrosion.')
    # parser.add_argument("command",
    #                     metavar="<command>",
    #                     help="'train' or 'eval'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/corrosion/dataset/",
                        help='Directory of the Corrosion dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    args = parser.parse_args()

    args.command = "train"
    args.dataset = "corrosion"
    args.weights = "coco"
    args.augment = False

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "evaluate":
        assert args.image, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("augment: ", args.augment)

    MODEL_DIR = args.logs
    # Configurations
    if args.command == "train":
        config = CorrosionConfig()
    else:
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=MODEL_DIR)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=MODEL_DIR)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))
