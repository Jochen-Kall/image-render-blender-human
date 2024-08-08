### About [OPLINNO-242](https://github.com/mnt1lr/image-render-blender-human/tree/feature/OPLINNO-242---Convert-Eyebrow-particles-to-mesh):


### What changed?

Added
 - `anyhuman2/labelling/mapping/eyebrows/Eyebrows_001.json`
 - `anyhuman2/labelling/mapping/eyebrows/Eyebrows_002.json`
 - `anyhuman2/labelling/mapping/eyebrows/Eyebrows_003.json`
 - `anyhuman2/labelling/mapping/eyebrows/Eyebrows_004.json`
 - `anyhuman2/labelling/mapping/eyebrows/Eyebrows_005.json`
 - `anyhuman2/labelling/mapping/eyebrows/Eyebrows_006.json`
 - `anyhuman2/labelling/mapping/eyebrows/Eyebrows_007.json`
 - `anyhuman2/labelling/mapping/eyebrows/Eyebrows_008.json`
 - `anyhuman2/labelling/mapping/eyebrows/Eyebrows_009.json`

Each json file contains WFLW labels [33-50] for each eyebrow style, `HG_Eyebrows_001 `through `HG_Eyebrows_009`

Added new entry `sEyebrowLabelsPath` which refers to folder path that contains eyebrow label files (e.g `Eyebrows_009.json`), to `anyhuman2\personas\FILE_male.json`, as follows

```json
{
    "dictCustom": {
        "sGender": "male",
        "sArmatureName": "Armature.001",
        "bOpenPoseHandLabels": true,
        "sOpenposeHandLabelFile": "labelling//mapping//openpose_hand_humgen.json",
        "bFacialRig": true,
        "sWFLWLableFile": "labelling//mapping//WFLW_bones.json",
        "sEyebrowLabelsPath":"labelling//mapping//eyebrows//",
        "sPoseFilename": {
            "set": null
        },
        "dBeardLength": {...}
    },
    "dictHumGen_V4": {...}
}
```

> Case **FILE** mode:

When a new human is created from `FILE_male.json` file, depending on eyebrow style set in `dictHumGen_V4['hair']['eyebrows']['set']`, appropriate JSON file will be read from the path taken from `dictCustom['sEyebrowLabelsPath']`

Suppose `dictHumGen_V4['hair']['eyebrows']['set']` = `Eyebrows_007` and `dictCustom['sEyebrowLabelsPath']` = `labelling//mapping//eyebrows//`then JSON file will be taken as `labelling//mapping//eyebrows//Eyebrows_007.json`


> Case **RANDOM** mode:

When a new human is generated, appropriate JSON file will be read from the path mentioned in `dictCustom['sEyebrowLabelsPath']` by getting eyebrow style from HumGen3D API `human.hair.eyebrows.as_dict()['set']`


### How to test [OPLINNO-242](https://github.com/mnt1lr/image-render-blender-human/tree/feature/OPLINNO-242---Convert-Eyebrow-particles-to-mesh):

 - 1. In `anyhuman2\personas\FILE_male.json`, change `sEyebrowLabelsPath` if required
 - 2. In `anyhuman2\personas\FILE_male.json`, set `dictHumGen_V4['hair']['eyebrows']['set']` to required eyebrow style, for now you can test
    - `Eyebrows_005`
    - `Eyebrows_009`
 - 3. TODO: add all labels for all eyebrows style for both left and right side