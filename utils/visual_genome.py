"""Contains code to generate the individual datasets for all the predicates.
"""

import copy
import random
import warnings
from collections import defaultdict

import numpy as np

# Import tqdm_notebook if in Jupyter notebook
try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except (AttributeError, ImportError):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def get_vg_obj_name(obj):
    """ Gets the name of an object from the VG dataset.
    This is necessary because objects are inconsistent--
    some have 'name' and some have 'names'. :(

    Args:
        obj: an object from the VG dataset
    Returns string containing the name of the object
    """
    try:
        name = obj["name"]
    except KeyError:
        name = obj["names"][0]
    return name


def filter_relationships(annotations, condition, inplace=False):
    """Filters list of relationships for the predicates specified.

    Args:
        annotations: list of relationships from VG (e.g. relationships.json)
        condition: function operating over each rel `r` specifying when to keep a relationship
        inplace: whether to modify a set of annotations inplace.
    Returns:
        filtered_annotations: list of relationships containing only those
            associated with predicate_synonyms
    """

    def invalid_bbox(obj):
        """Determine whether a bounding box construction is invalid."""
        return obj["h"] == 0 or obj["w"] == 0

    if inplace:
        filtered_annotations = annotations
    else:
        filtered_annotations = copy.deepcopy(annotations)

    for a in tqdm(filtered_annotations):
        rels = a["relationships"]
        selected_rels = []
        for idx, r in enumerate(rels):
            if invalid_bbox(r["subject"]):
                msg = f"Invalid bbox: {r['subject']}. Skipping rel."
                warnings.warn(msg)
                continue
            elif invalid_bbox(r["object"]):
                msg = f"Invalid bbox: {r['object']}. Skipping rel."
                warnings.warn(msg)
                continue

            r["predicate"] = r["predicate"].lower()
            if condition(r):
                selected_rels.append(r)

        a["relationships"] = selected_rels

    if not inplace:
        return filtered_annotations


def count_relationships(annotations, syns_to_preds=None):
    """ Counts the number of relationships per-predicate.

    Args:
        annotations: list of relationships from VG (e.g. relationships.json)
        syns_to_preds: dict mapping synonyms to their corresponding predicates
    Returns: 
        rel_counts: dict of pred_name to counts
    """
    rel_counts = defaultdict(int)

    for a in annotations:
        rels = a["relationships"]
        for r in rels:
            pred = r["predicate"]
            if syns_to_preds and pred in syns_to_preds:
                pred = syns_to_preds[pred]

            rel_counts[pred] += 1

    rel_counts["_TOTAL"] = sum(rel_counts.values())
    return dict(rel_counts)


def sample_relationships(relationships, pred_counts, n_per_pred):
    """Randomly samples up to `n_per_pred` relationships per predicate.
    
    Args:
        relationships: list of relationships from VG (e.g. relationships.json)
        pred_counts: dict of predicate names to corresponding counts in relationships
        n_per_pred: number of labels to keep per predicate
    Returns:
        sample_train: copy of relationships with `UNLABELED` in place of relationships
            with removed labels
    """

    # Store per-predicate indexes for relationships that we should keep
    pred_idx_samples = {}

    # First, sample for the lowest-label relationships
    sorted_counts = sorted(pred_counts.items(), key=lambda kv: kv[1])
    for pred, pred_count in sorted_counts:
        pred = pred.lower()
        # Randomly sample up to `n_per_red` relationships
        if pred_count < n_per_pred:
            # Sample all `pred_count` relationships
            pred_idx_samples[pred] = list(range(pred_count))
        else:
            pred_idx_samples[pred] = random.sample(list(range(pred_count)), n_per_pred)

    # Keep track of the idx for each relationship
    rel_idx = {pred: -1 for pred in pred_counts.keys()}
    sampled_train = copy.deepcopy(relationships)

    for a in sampled_train:
        updated_rels = []
        for r in a["relationships"]:
            pred = r["predicate"].lower()

            # Increment idx for a relationship when we encounter it
            rel_idx[pred] += 1

            # Unsampled relationships are "UNLABELED"
            if rel_idx[pred] not in pred_idx_samples[pred]:
                r["predicate"] = "UNLABELED"

            updated_rels.append(r)

        a["relationships"] = updated_rels
    return sampled_train


def get_labels(relationships, predicates, syns_to_preds=None):
    """Extract [n_nested_relationships, n_predicates] matrix of labels from sampled
    relationships.
    
    Args:
        relationships: list of relationships from VG (e.g. relationships.json)
        predicates: list of predicates for which we should extract labels
        syns_to_preds: dict mapping synonyms to their corresponding predicates
    Returns:
        labels: [num_examples, num_predicates] matrix of labels 
            -1 is UNLABELED
            0 is Negative
            +1 is Positive
    """

    predicates = sorted(predicates)  # important: sorted so labels correspond

    labels = []  # -1 > unlabeled, 0 -> negative, +1 > positive

    for a in tqdm(relationships):
        for r in a["relationships"]:
            pred = r["predicate"]

            # create labels
            if pred == "UNLABELED":
                label = np.ones(len(predicates)) * -1
            else:
                pred = pred.lower()
                if pred not in predicates:
                    if not syns_to_preds:
                        raise ValueError(f"{pred} not found.")
                    pred = syns_to_preds[pred]

                pidx = predicates.index(pred)
                label = np.zeros(len(predicates))
                label[pidx] = 1
            labels.append(label)

    return np.array(labels)


def extract_obj_categories(annotations, predicates, object_synonyms):
    """Extract possible object categories for a list of predicates.
        Args:
            annotations: list of relationships from VG (e.g. relationships.json)
            predicates: list of predicates which we should consider
            object_synonyms: dict of object name to list of possible synonyms
        Returns:
            obj_categories: list of object categories in `annotations`
    """

    # Get entities for predicates
    obj_entities = defaultdict(int)
    sub_entities = defaultdict(int)
    for a in annotations:
        for r in a["relationships"]:
            if r["predicate"] in predicates:

                obj_name = get_vg_obj_name(r["object"])
                sub_name = get_vg_obj_name(r["subject"])

                obj_entities[obj_name] += 1
                sub_entities[sub_name] += 1

    all_entities = list(obj_entities.keys()) + list(sub_entities.keys())

    # Filter objects based on synonyms of original `object_list`
    obj_categories = sorted(
        list(
            set(
                [
                    obj
                    for obj, obj_syns in object_synonyms.items()
                    for item in all_entities
                    if item in obj_syns
                ]
            )
        )
    )

    return obj_categories
