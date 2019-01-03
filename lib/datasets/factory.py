# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
#from datasets.coco import coco
from datasets.linkface_dataset import LinkfaceDataset
import numpy as np

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
'''
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))
'''

# Set up coco_2015_<split>
'''
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))
'''

# for ds in ['VehicleText', 'DriverText']:
#   classes = ('__background__',  'text')
#   for split in ['trainval', 'test', 'val']:
#     name = '{}_{}'.format(ds, split)
#     __sets[name] = (lambda dataset=ds, split=split, classes=classes: LinkfaceDataset(ds, split, classes))

# for ds in ['VehicleLicense', 'DriverLicense']:
#   classes = ('__background__',  'license')
#   for split in ['trainval', 'test', 'val']:
#     name = '{}_{}'.format(ds, split)
#     __sets[name] = (lambda dataset=ds, split=split, classes=classes: LinkfaceDataset(ds, split, classes))


__set_classes = {
    "plate": ('__background__', 'text'),
    "ICDAR2015": ('__background__', 'text'),
    "VdibLicense": ('__background__', 'vehicle', 'drive', 'id', 'card'),
    "VehicleText": ('__background__', 'text'),
    "DriverText": ('__background__', 'text', 'seal', 'pic'),
    "VehicleLicense": ('__background__', 'text'),
    "DriverLicense": ('__background__', 'text'),
    "IdText": ('__background__', 'number', 'name', 'year', 'month', 'day', 'sex', 'nation', 'address', 'authority', 'timelimit'),
    "IdCard": ('__background__', 'text'),
    "Cow": ('__background__', 'cow'),
    "Head": ('__background__', 'head'),
    "VPText": ('__background__', 'text'),
    "VPAllText": ('__background__', 'text'),
    "VPNewText": ('__background__', 'text'),
    "Audit": ('__background__', 'text'),
    "DML": ('__background__', 'clear','vague'),
    "MedicalDetectName": ('__background__', 'text'),
    #"Audit": ('__background__', 'text_up', 'text_right', 'text_down', 'text_left'),
}


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name in __sets:
        return __sets[name]()

    dataset_names = name.split('_')
    if dataset_names[0] in __set_classes:
        return LinkfaceDataset(dataset_names[0], dataset_names[1], __set_classes[dataset_names[0]])

    raise KeyError('Unknown dataset: {}'.format(name))


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
