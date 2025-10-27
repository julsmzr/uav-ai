# Dataset Processing

This file specifies the source format and target format that is used to format the data ready for training.

## Anti-UAV300

### Folder Structure
```
dataset/
├── train/
│   ├── sequence1/
│   │   ├── visible.mp4
│   │   ├── visible.json
│   │   ├── infrared.mp4
│   │   └── infrared.json
│   └── sequence2/
│       ├── visible.mp4
│   │   ├── visible.json
│   │   ├── infrared.mp4
│   │   └── infrared.json
├── test/
│   ├── sequence3/
│   │   └── ...
│   └── sequence4/
│       └── ...
└── val/
    └── ...
```

### Label Format
```
{
    "exist": [
        0,
        0,
        0,
        1,
        1,
        0,
        ...
    ],
    "gt_rect": [
        255,    # x_min
        281,    # y_min
        40,     # width
        56      # height
    ],
    [
        188,
        523,
        109,
        151
    ],
    ...
}
```

where coordinates are <b>not</b> normalized, the origin is the top-left image corner and the axis are growing X-right and Y-down

### Cleaning and Filtering
we remove the gt_rect entries where the exists flag is set to 0

Example:
```
1 [973, 601, 145, 85]
0 [1036, 598, 83, 56]
0 [0, 0, 0, 0]
0 []
0 [895, 619, 113, 88]
0 []
1 [812, 606, 125, 80]
1 [760, 604, 121, 70]
```

will be transformed to 

```
1 [973, 601, 145, 85]
0 []
0 []
0 []
0 []
0 []
0 []
0 []
1 [812, 606, 125, 80]
1 [760, 604, 121, 70]
```

---
## COCO
The COCO format as proposed by Lin et. al (2015)[1] describes the following structure:

### Folder Structure
```
dataset/
├── images/
│   ├── train/
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   └── val/
│       ├── 0003.jpg
│       ├── 0004.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── 0001.txt
    │   ├── 0002.txt
    │   └── ...
    └── val/
        ├── 0003.txt
        ├── 0004.txt
        └── ...
```

### Label Format
```
cls x_min y_min width height
```

Example:
```
0 0.479492 0.688771 0.955609 0.5955
0 0.736516 0.247188 0.498875 0.476417
1 0.637063 0.732938 0.494125 0.510583
```

where coordinates are normalized, the origin is the top-left image corner and the axis are growing X-right and Y-down


<sub><b>[1] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, & Piotr Dollár. (2015).</b> Microsoft COCO: Common Objects in Context.</sub>