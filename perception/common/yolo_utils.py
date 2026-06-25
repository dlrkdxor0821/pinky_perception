"""Shared helpers for turning Ultralytics results into plain dicts.

Both the server (PyTorch) and edge (NCNN) detectors produce Ultralytics
`Results`; we normalize them to the same JSON-friendly schema so the two
paths are directly comparable.
"""


def names_lookup(names, cls):
    if isinstance(names, dict):
        return names.get(cls, str(cls))
    try:
        return names[cls]
    except (IndexError, KeyError, TypeError):
        return str(cls)


def ultralytics_to_dets(result, names):
    """Convert one Ultralytics Result to a list of detection dicts."""
    dets = []
    for b in result.boxes:
        cls = int(b.cls[0])
        dets.append({
            "cls": cls,
            "name": names_lookup(names, cls),
            "conf": float(b.conf[0]),
            "xyxy": [float(x) for x in b.xyxy[0].tolist()],
        })
    return dets
