"""Utility functions for visualizations."""

import os
import random

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from utils.primitives import BBoxPrim


def show_image(image_path, red_bboxes=None, cyan_bboxes=None):
    """Draw red and cyan boxes in an image.

    Args:
        image_path: Location of the image.
        red_bboxes (optional): List of bboxes to be drawn in red.
        cyan_bboxes (optional): List of bboxes to be drawn in cyan.
    """

    def show_bbox(bbox, color):
        if type(bbox).__name__ == "BBoxPrim":
            bbox_obj = bbox
        else:
            bbox_obj = BBoxPrim(bbox)
        ax.add_patch(
            Rectangle(
                (bbox_obj.x0, bbox_obj.y0),
                bbox_obj.width,
                bbox_obj.height,
                fill=False,
                edgecolor=color,
                linewidth=1,
            )
        )

    I = Image.open(image_path)
    plt.imshow(I)
    ax = plt.gca()

    if red_bboxes:
        for r_bbox in red_bboxes:
            show_bbox(r_bbox, "red")

    if cyan_bboxes:
        for c_bbox in cyan_bboxes:
            show_bbox(c_bbox, "cyan")

    plt.show()


def view_n_image_rels(relationships, n, image_dir="data/VisualGenome/VG_100K/"):
    """Displays a relationship from `n` distinct images.
    
    Args:
        relationships: list of relationships from VG (e.g. relationships.json)
        n: images to show
        image_dir: directory where images live
    """

    from utils.visual_genome import get_vg_obj_name

    # Shuffle order that we see the images
    relationships_idx = list(range(len(relationships)))
    random.shuffle(relationships_idx)

    n_seen = 0

    for idx in relationships_idx:
        a = relationships[idx]
        rels = a["relationships"]
        if len(rels) > 0:
            n_seen += 1

            for rel in rels:
                sub = BBoxPrim.from_vg_obj(rel["subject"])
                obj = BBoxPrim.from_vg_obj(rel["object"])
                sub_name = get_vg_obj_name(rel["subject"])
                obj_name = get_vg_obj_name(rel["object"])
                pred = rel["predicate"]
                print(f"{sub_name} <{pred}> {obj_name}")
                image_fn = os.path.join(image_dir, f"{a['image_id']}.jpg")
                show_image(image_fn, [sub], [obj])
                break

        if n_seen >= n:
            break
