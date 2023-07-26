rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]

# 0 - clip_vision, 1 - clip_vision_text, 2 - sam
best_ids_dict = {'V1v': 2,
 'V1d': 2,
 'V2v': 2,
 'V2d': 2,
 'V3v': 2,
 'V3d': 2,
 'hV4': 2,
 'EBA': 1,
 'FBA-1': 1,
 'FBA-2': 1,
 'OFA': 1,
 'FFA-1': 1,
 'FFA-2': 1,
 'OPA': 1,
 'PPA': 1,
 'RSC': 1,
 'OWFA': 1,
 'VWFA-1': 1,
 'VWFA-2': 1,
 'mfs-words': 1,
 'mTL-words': 1,
 'early': 2,
 'midventral': 2,
 'midlateral': 1,
 'midparietal': 2,
 'ventral': 1,
 'lateral': 1,
 'parietal': 1}

best_roi_class = {
    'prf-visualrois': 2,
    'floc-bodies': 1,
    'floc-faces': 1,
    'floc-places': 1,
    'floc-words': 1,
    'streams': 1
}


def get_roi_class(roi):
    if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
        roi_class = 'prf-visualrois'
    elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
        roi_class = 'floc-bodies'
    elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
        roi_class = 'floc-faces'
    elif roi in ["OPA", "PPA", "RSC"]:
        roi_class = 'floc-places'
    elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
        roi_class = 'floc-words'
    elif roi in ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]:
        roi_class = 'streams'

    return roi_class