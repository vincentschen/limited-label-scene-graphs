"""Classes for constructing image-agnostic features."""

import numpy as np
from types import SimpleNamespace

# Import tqdm_notebook if in Jupyter notebook
try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except (AttributeError, ImportError):
    from tqdm import tqdm
else:
    from tqdm import tqdm


class CategoricalPrim(object):
    """Simple one-hit category wrapper.
    """

    def __init__(self, subject_index, object_index, num_categories=100):
        """Contructor for CategoryPrim.

        Args:
            subject_index: Category label of the subject.
            object_index: Category label of the object.
        """
        self.subject_index = subject_index
        self.object_index = object_index
        self.num_categories = num_categories

    def extract_features(self):
        """Returns two one hot vectors for the two objects.

        Returns:
            A list of features.
        """
        subs = [0] * self.num_categories
        subs[self.subject_index] = 1
        objs = [0] * self.num_categories
        objs[self.object_index] = 1
        return subs + objs

    def __eq__(self, other):
        """Checks if two categories are the same.

        Args:
            other: An instance of CategoryPrim.

        Returns:
            A boolean indicating whether they are equal.
        """
        return (
            self.subject_index == other.subject_index
            and self.object_index == other.object_index
        )


class BBoxPrim(object):
    """Simple bounding box wrapper that stores the basic box info.
    """

    def __init__(self, bbox):
        """Constructor for SimpleBBoxPrim.

        Args:
            bbox: A list containing the top, bottom, left, and right
                coordinates of the box.
        """
        top, bottom, left, right = tuple(bbox)
        self.x0 = left
        self.y0 = top
        self.x1 = right
        self.y1 = bottom
        self.width = self.x1 - self.x0
        self.height = self.y1 - self.y0
        self.area = self.width * self.height
        self.perimeter = 2 * self.width + 2 * self.height

    @classmethod
    def from_vg_obj(cls, obj):
        """Initializes the BBoxPrim using VisualGenome-format object

        Args:
            obj: VG object with fields 'x', 'y', 'w', 'h'

        Returns: initialization of BboxPrim class
        """
        bbox = [obj["y"], obj["y"] + obj["h"], obj["x"], obj["x"] + obj["w"]]
        return cls(bbox)

    def extract_features(self):
        """Creates a list of features for Reef.

        Returns:
            A list of features.
        """
        return [
            self.x0,
            self.x1,
            self.y0,
            self.y1,
            self.width,
            self.height,
            self.area,
            self.perimeter,
        ]

    def get_bbox(self):
        """Returns the original bounding box.

        Returns:
            [y0, y1, x0, x1].
        """
        return [self.y0, self.y1, self.x0, self.x1]

    def __eq__(self, other):
        """Checks if two boxes are the same.

        Args:
            other: The other SimpleBBoxPrim instance.

        Returns:
            A boolean indicating whether they are equal.
        """
        return (
            self.x0 == other.x0
            and self.x1 == other.x1
            and self.y0 == other.y0
            and self.y1 == other.y1
        )

    def __hash__(self):
        """Hash of object.
        """
        return hash(str(self.get_bbox()))


class SpatialPrim(object):
    """Wrapper that stores spatial primitive for a relationship.
    """

    def __init__(self, subject_bbox, object_bbox):
        """Construtor for RelationshipPrim.

        Args:
            subject_bbox: BBoxPrim instance.
            object_bbox: BBoxPrim instance.
        """
        self.subject_bbox = subject_bbox
        self.object_bbox = object_bbox

    def extract_features(self):
        """Creates a list of features for Reef.

        Returns:
            A list of features.
        """
        delta_x0 = (self.subject_bbox.x0 - self.object_bbox.x0) / (
            1.0 * self.subject_bbox.width
        )
        delta_x1 = (self.subject_bbox.x1 - self.object_bbox.x1) / (
            1.0 * self.subject_bbox.width
        )
        delta_y0 = (self.subject_bbox.y0 - self.object_bbox.y0) / (
            1.0 * self.subject_bbox.height
        )
        delta_y1 = (self.subject_bbox.y1 - self.object_bbox.y1) / (
            1.0 * self.subject_bbox.height
        )
        width_ratio = self.object_bbox.width / float(self.subject_bbox.width)
        height_ratio = self.object_bbox.height / float(self.subject_bbox.height)
        area_ratio = self.object_bbox.area / float(self.subject_bbox.area)
        return [
            delta_x0,
            delta_x1,
            delta_y0,
            delta_y1,
            width_ratio,
            height_ratio,
            area_ratio,
        ]

    def __eq__(self, other):
        """Checks if two Relationships are the same.

        Args:
            other: The other RelationshipPrim instance.

        Returns:
            A boolean indicating whether they are equal.
        """
        return (
            self.subject_bbox == other.subject_bbox
            and self.object_bbox == other.object_bbox
        )

    def __hash__(self):
        """Hash of object.
        """
        return hash(str(self.subject_bbox.hash()) + "-" + str(self.object_bbox.hash()))


def find_name_in_syns(name, syns):
    for k, v in syns.items():
        if name in v:
            return k

    # not found
    raise ValueError("%s not found in syns" % name)


def get_primitive_features(relationships, entity_list, object_synonyms):
    from utils.visual_genome import get_vg_obj_name

    """ Generates primitive matrices given list of relationships
    Args:
        annotations: list of relationships from VG (e.g. relationships.json)
            with 'UNLABELED' predicates
        object_list: list of objets. indexes will be used as features.
        object_synonyms: dict mapping object entities to list of possible synonyms

    Returns:
        List of examples with .spatial and .categorical feature attributes
    """

    examples = []
    for a in tqdm(relationships):
        for r in a["relationships"]:
            # create features
            sub = BBoxPrim.from_vg_obj(r["subject"])
            obj = BBoxPrim.from_vg_obj(r["object"])
            rel = SpatialPrim(sub, obj)
            spatial_features = rel.extract_features()

            # get object name, or find the common name from list of synonyms
            sub_name = get_vg_obj_name(r["subject"])
            if sub_name not in entity_list:
                sub_name = find_name_in_syns(sub_name, object_synonyms)
            obj_name = get_vg_obj_name(r["object"])
            if obj_name not in entity_list:
                obj_name = find_name_in_syns(obj_name, object_synonyms)

            sub_id = entity_list.index(sub_name)
            obj_id = entity_list.index(obj_name)
            cat = CategoricalPrim(sub_id, obj_id, num_categories=len(entity_list))
            categorical_features = cat.extract_features()

            x = SimpleNamespace(
                spatial=np.array(spatial_features), 
                categorical=np.array(categorical_features)
            )
            examples.append(x)

    return examples


def get_deep_features(relationships):
    """ Generates matrix of deep features given relationships.
    Args:
        annotations: list of relationships from VG (e.g. relationships.json)
    Returns: deep features (N x 2048)
    """

    from torch.utils.data import DataLoader
    from utils.deep_features import BBoxDataset, extract_resnet_features

    data = []  # collect list of tuples (filename, bbox1, bbox2)
    for a in relationships:
        for r in a["relationships"]:
            sub = BBoxPrim.from_vg_obj(r["subject"]).get_bbox()
            obj = BBoxPrim.from_vg_obj(r["object"]).get_bbox()
            fn = str(a["image_id"]) + ".jpg"
            data.append((fn, sub, obj))

    dataset = BBoxDataset(data, image_dir="data/VisualGenome/VG_100K", image_size=224)
    data_loader = DataLoader(
        dataset=dataset, batch_size=8, shuffle=False, num_workers=4
    )
    features = extract_resnet_features(data_loader, batch_size=8)
    return features
