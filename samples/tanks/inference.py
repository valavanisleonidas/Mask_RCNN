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
import matplotlib
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Agg backend runs without a display
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
from pathlib import Path
import datetime

import numpy as np
from xml.etree import ElementTree as ET
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

RESULTS_DIR = os.path.join(ROOT_DIR, "results/tanks/")

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class CorrosionConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "tanks"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + corrosion

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 42 // IMAGES_PER_GPU
    VALIDATION_STEPS = 7 // IMAGES_PER_GPU

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between corrosion and BG
    DETECTION_MIN_CONFIDENCE = 0.7

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 400

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Image mean (RGB)    
    # mean for tanks train set
    MEAN_PIXEL = np.array([[73.06563891, 76.45901485, 71.69669517]])

    
    # smaller for avoid overfitting
    BACKBONE = "resnet50"

    # we have both small and large objects
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels


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
            pts = np.array([np.array(point.split(",")).astype(np.float) for point in str_points])
            all_points_y = pts[:, 1]
            all_points_x = pts[:, 0]
            rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "corrosion":
            return info["path"]
        else:
            super(CorrosionDataset, self).image_reference(image_id)


############################################################
#  Inference
############################################################

def calc_mean_dataset(dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)
    
    config = CorrosionConfig()
    
    # Read dataset
    dataset = CorrosionDataset()
    dataset.load_data(dataset_dir, subset)
    dataset.prepare()

    channel_sum = 0
    pixel_num = 0
    channel_sum_squared = 0

    for image_id in tqdm(dataset.image_ids):
        # Load image and run detection
        image = dataset.load_image(image_id)

        image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)

        pixel_num += image.shape[0] * image.shape[1]
        channel_sum += np.sum(image, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(image), axis=(0, 1))

    mean = np.array(channel_sum / pixel_num)
    print('mean of dataset is ', mean)


def detect(model, dataset_dir, subset, weights_path):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    post_fix = "_".join(weights_path.split("/")[-2:]).replace("tanks","").replace("mask_rcnn__","").replace(".h5","")
    submit_dir = "submit_{:%Y%m%dT%H%M%S}_".format(datetime.datetime.now())
    submit_dir += post_fix
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = CorrosionDataset()
    dataset.load_data(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    for image_id in tqdm(dataset.image_ids):
        # Load image and run detection
        # image = dataset.load_image(image_id)
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)

        # Detect objects
        r = model.detect([image], verbose=0)[0]

        # # Save image with masks
        # visualize.display_instances(
        #     image, r['rois'], r['masks'], r['class_ids'],
        #     dataset.class_names, r['scores'],
        #     show_bbox=True, show_mask=True,
        #     title="Predictions")
        # plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["path"].split("/")[-1]))

        try:
            resp = visualize.display_differences_filtered(image,
                gt_bbox, gt_class_id, gt_mask,
                r['rois'], r['class_ids'], r['scores'], r['masks'],
                dataset.class_names, ax=get_ax(),
                show_box=False, show_mask=False,
                iou_threshold=0.7, score_threshold=0.7)

            if resp == False:
                continue
            plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["path"].split("/")[-1]))
        except:
            pass
        
def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    fig.tight_layout()
    return ax
    
    
class InferenceConfig(CorrosionConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7
    RPN_NMS_THRESHOLD = 0.5
    
    
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect tanks.')
    # parser.add_argument("command",
    #                     metavar="<command>",
    #                     help="'train' or 'eval'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/tanks/dataset/",
                        help='Directory of the tanks dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    args.command = "evaluate"
    args.dataset = "tanks"
    # args.weights = "/home/ubuntu/leonidas/resnet101/mask_rcnn_corrosion_0030.h5"
    args.subset = 'val'
        
    # print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    
    
    # calc_mean_dataset(args.dataset, args.subset)
    for i in [2]:
        weights = f"/home/ubuntu/leonidas/Mask_RCNN/logs/tanks20230303T1649/mask_rcnn_tanks_000{i}.h5"
        print(weights)
        
        config = InferenceConfig()
        config.display()

        # Create model
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
        model.load_weights(weights, by_name=True)

        detect(model, args.dataset, args.subset, weights)
