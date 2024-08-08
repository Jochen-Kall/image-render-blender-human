### About [MARS-123](https://github.com/mnt1lr/image-render-blender-human/tree/feature/MARS-123---integrate-IMS-labels):


### What changed?

[MARS-123](https://github.com/mnt1lr/image-render-blender-human/tree/feature/MARS-123---integrate-IMS-labels) created on top of [OPLINNO-242](https://github.com/mnt1lr/image-render-blender-human/tree/feature/OPLINNO-242---Convert-Eyebrow-particles-to-mesh) 


Added new entry `sIMSLabels` which refers to file that contains `IMS` labels. This filepath is relative to [anyhuman2](https://github.com/mnt1lr/image-render-blender-human/tree/feature/MARS-123---integrate-IMS-labels/src/anyhuman2) folder as rest of relative paths

Refer [IMS_Bones.json](https://github.com/mnt1lr/image-render-blender-human/blob/feature/MARS-123---integrate-IMS-labels/src/anyhuman2/labelling/mapping/IMS_bones.json)

```json
{
    "dictCustom": {
        "sGender": "male",
        "sArmatureName": "Armature.001",
        "bOpenPoseHandLabels": true,
        "sOpenposeHandLabelFile": "labelling//mapping//openpose_hand_humgen.json",
        "bFacialRig": true,
        "sWFLWLableFile": "labelling//mapping//WFLW_bones.json",
    >   "sIMSLabels":"labelling//mapping//IMS_bones.json",
        "sEyebrowLabelsPath":"labelling//mapping//eyebrows//",
        "sPoseFilename": {
            "set": null
        },
        "dBeardLength": {...}
    },
    "dictHumGen_V4": {...}
}
```

### How to test [MARS-123](https://github.com/mnt1lr/image-render-blender-human/tree/feature/MARS-123---integrate-IMS-labels):

 - 1. In `anyhuman2\personas\FILE_male.json`, change `sIMSLabels` if required
 - 2. Launch blender in debug mode, you should see IMS labels, once you run script [dev.py](https://github.com/mnt1lr/image-render-blender-human/blob/feature/MARS-123---integrate-IMS-labels/src/anyhuman2/dev.py) with necessary local changes