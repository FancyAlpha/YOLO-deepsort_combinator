# vim: expandtab:ts=4:sw=4
# from __future__ import division
from __future__ import division, print_function, absolute_import

from PyTorchYOLOv3.models import *
# from PyTorchYOLOv3.models import *
from PyTorchYOLOv3.utils.utils import *
from PyTorchYOLOv3.utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image
import tensorflow as tf

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

# import argparse
# import os

import cv2
import numpy as np

from deepsort.application_util import preprocessing
from deepsort.application_util import visualization
from deepsort.deep_sort import nn_matching
from deepsort.deep_sort.detection import Detection
from deepsort.deep_sort.tracker import Tracker


def bool_string(input_string):
    if input_string not in {"True", "False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")


# deepsort code start here
def gather_sequence_info(sequence_dir):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        # * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
    }
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    # detections = None
    # if detection_file is not None:
    #     detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    min_frame_idx, max_frame_idx = 0, 0

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        print("y'all we ain't findin' no ings hea.\n")
        exit()
    #     min_frame_idx = int(detections[:, 0].min())
    #     max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    # feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        # "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        # "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """
    [deepsort ]
    Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """

    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


if __name__ == "__main__":

    """
    [Deepsort docs]
    Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.
    """

    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_folder", type=str, default="PyTorchYOLOv3/data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="PyTorchYOLOv3/config/yolov3.cfg",
                        help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="PyTorchYOLOv3/weights/yolov3.weights",
                        help="path to weights file")
    parser.add_argument("--class_path", type=str, default="PyTorchYOLOv3/data/coco.names",
                        help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    # parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")

    # deepsort arguments
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    # parser.add_argument(
    #     "--detection_file", help="Path to custom detections.", default=None,
    #     required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
                              " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
                                 "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
                                       "box height. Detections with height smaller than this value are "
                                       "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap", help="Non-maxima suppression threshold: Maximum "
                                  "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
                                      "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
                            "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)

    opt = parser.parse_args()
    print(opt)


    # deeposort init
    seq_info = gather_sequence_info(opt.sequence_dir)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", opt.max_cosine_distance, opt.nn_budget)
    tracker = Tracker(metric)
    results = []
    # end deepsort init

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

# img_size=seq_info["image_size"]
    # Set up model
    model = Darknet(opt.model_def, 416).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    image_dir = os.path.join(opt.sequence_dir, "img1")

    # we use the data loader instead of deepsort's implementation fingers crossed
    dataloader = DataLoader(
        ImageFolder(image_dir, 416),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    iterable_dataloader = iter(dataloader)
    # data = []

    # data = [(batch_i, (img_paths, input_imgs)) for batch_i, (img_paths, input_imgs) in enumerated_dataloader]

    def frame_callback(vis, frame_idx):

        print("Processing frame %05d" % frame_idx)

        # darknet's image input?
        gui_image = cv2.imread(
            seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)

        # image = Image.open(seq_info["image_filenames"][frame_idx])
        # image_transform = transforms.Compose([transforms.ToTensor()])
        #
        # img_in = image_transform(image).float()
        # img_in = np.array(ima ge)
        # img_in = torch.from_numpy(img_in)
        # img_in = torch.Tensor(img_in)


    # Configure input
    #     img_in = img_in.unsqueeze_(0)

        (img_paths, input_imgs) = next(iterable_dataloader)

        input_imgs = Variable(input_imgs.type(Tensor))
        # input_imgs = input_imgs.permute(0, 3, 1, 2)
        # Get detections
        with torch.no_grad():
            torch_detections = model(input_imgs)
            torch_detections = non_max_suppression(torch_detections, opt.conf_thres, opt.nms_thres)

        # bridge the gap between the detections above and below

        torch_detections = torch_detections.data.numpy()
        print(torch_detections)

        detections = []
        for (x1, y1, x2, y2, object_conf, class_score, class_pred) in torch_detections:

            detections.append(Detection(x1, y1, x2-x1, y2-y1, object_conf, class_score, class_pred))


        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if opt.display:

            vis.set_image(gui_image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if opt.display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(opt.output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]), file=f)
