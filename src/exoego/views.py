"""Assembly101 camera views.

Eight fixed RGB cameras (exocentric) and four head-mounted monochrome cameras
(egocentric). The ego cameras were re-serialised partway through collection, so
each of the four ego slots has two possible names; a given recording carries one
name from each pair. Preference order below follows the upstream slot order
(e1..e4), so we consistently pick the same physical camera slot where available.

Source: https://github.com/assembly-101/assembly101-download-scripts (view_dict)
"""

EXO_VIEWS = [
    "C10095_rgb",
    "C10115_rgb",
    "C10118_rgb",
    "C10119_rgb",
    "C10379_rgb",
    "C10390_rgb",
    "C10395_rgb",
    "C10404_rgb",
]

# Flattened e1..e4, each slot's two naming variants adjacent.
EGO_VIEWS = [
    "HMC_84346135_mono10bit",
    "HMC_21176875_mono10bit",
    "HMC_84347414_mono10bit",
    "HMC_21176623_mono10bit",
    "HMC_84355350_mono10bit",
    "HMC_21110305_mono10bit",
    "HMC_84358933_mono10bit",
    "HMC_21179183_mono10bit",
]


def is_ego(view: str) -> bool:
    return view.startswith("HMC_")


def is_exo(view: str) -> bool:
    return view.startswith("C1")


def pick_preferred(available, preference):
    """First entry of `preference` present in `available`, else None."""
    for candidate in preference:
        if candidate in available:
            return candidate
    return None


def pick_ego(available):
    return pick_preferred(available, EGO_VIEWS)


def pick_exo(available):
    return pick_preferred(available, EXO_VIEWS)
