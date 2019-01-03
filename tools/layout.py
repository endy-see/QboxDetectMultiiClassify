# --------------------------------------------------------
#  Use the position info to count the layout of vechile license
# --------------------------------------------------------
import numpy as np

# get the keyword by the grid position
keywordGridV1 = [
    ['plateNo', 'vechileType'],
    ['owner', 'owner'],
    ['address', 'address'],
    ['useCharacter', 'model'],
    ['vin', 'vin'],
    ['engineNo', 'engionNo'],
    ['registerDate', 'issueDate']
]

keywordGridV2 = [
    ['plateNo', 'vechileType'],
    ['owner', 'owner'],
    ['address', 'address'],
    ['model', 'useCharacter'],
    ['engineNo', 'engionNo'],
    ['vin', 'vin'],
    ['registerDate', 'issueDate']
]

keyPoints = {
    'plateNo': [110, 90],
    'vechileType': [360, 90],
    'owner': [110, 130],
    'address': [110, 175],
    'useCharacter': [110, 225],
    'model' [310, 225],
    'vin' [290, 275],
    'model' [270, 320],
    'registerDate' [250, 370],
    'issueDate' [475, 370]
}

def layout(boxes):
    ys = []
    lengthes = []
    for box in boxes:
        ys.append(int((box[1] + box[3]) / 60))
        lengthes.append(int(box[0] - box[2]))

    sorted_ind = np.argsort(ys)
    sorted_scores = np.sort(ys)
    
    
    
    
