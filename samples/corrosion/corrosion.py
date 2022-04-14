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
import datetime

import numpy as np
from xml.etree import ElementTree as ET
import skimage.draw

# Root directory of the project
from keras.callbacks import Callback
from tqdm import tqdm

ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
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


class MeanAveragePrecisionCallback(Callback):
    def __init__(self, train_model, inference_model, dataset,
                 calculate_map_at_every_X_epoch=5, dataset_limit=None,
                 verbose=0):
        super().__init__()
        self.train_model = train_model
        self.inference_model = inference_model
        self.dataset = dataset
        self.calculate_map_at_every_X_epoch = calculate_map_at_every_X_epoch
        self.dataset_limit = len(self.dataset.image_ids)
        if dataset_limit is not None:
            self.dataset_limit = dataset_limit
        self.dataset_image_ids = self.dataset.image_ids.copy()

        if inference_model.config.BATCH_SIZE != 1:
            raise ValueError("This callback only works with the bacth size of 1")

        self._verbose_print = print if verbose > 0 else lambda *a, **k: None

    def on_epoch_end(self, epoch, logs=None):

        if epoch > 2 and (epoch + 1) % self.calculate_map_at_every_X_epoch == 0:
            self._verbose_print("Calculating mAP...")
            self._load_weights_for_model()

            mAPs, precisions, recalls = self._calculate_mean_average_precision()
            mAP = np.nanmean(mAPs)
            precision = np.nanmean(precisions)
            recall = np.nanmean(recalls)

            if logs is not None:
                logs["val_mean_average_precision"] = mAP
                logs["val_precision"] = precision
                logs["val_recall"] = recall

            self._verbose_print("mAP at epoch {0} is: {1}".format(epoch + 1, mAP))

        super().on_epoch_end(epoch, logs)

    def _load_weights_for_model(self):
        last_weights_path = self.train_model.find_last()
        self._verbose_print("Loaded weights for the inference model (last checkpoint of the train model): {0}".format(
            last_weights_path))
        self.inference_model.load_weights(last_weights_path,
                                          by_name=True)

    def _calculate_mean_average_precision(self):
        mAPs = []
        precisions = []
        recalls = []
        # Use a random subset of the data when a limit is defined
        np.random.shuffle(self.dataset_image_ids)

        for image_id in tqdm(self.dataset_image_ids[:self.dataset_limit]):
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(self.dataset,
                                                                                      self.inference_model.config,
                                                                                      image_id, use_mini_mask=False)

            if len(gt_bbox) == 0 or len(gt_mask) == 0:
                print('skipping iimage from eval since no GT annotations. Id: ', image_id)
                continue

            molded_images = np.expand_dims(modellib.mold_image(image, self.inference_model.config), 0)
            results = self.inference_model.detect(molded_images, verbose=0)
            r = results[0]
            # Compute mAP - VOC uses IoU 0.5
            AP, prec, rec, _ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"],
                                                r["class_ids"], r["scores"], r['masks'])
            mAPs.append(AP)
            precisions.append(prec)
            recalls.append(rec)

        return np.array(mAPs), np.array(precisions), np.array(recalls)


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
    mean_average_precision_callback = MeanAveragePrecisionCallback(model, model_inference, dataset_val,
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
