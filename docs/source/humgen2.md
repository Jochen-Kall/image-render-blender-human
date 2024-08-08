<!---
<LICENSE id="CC BY-SA 4.0">

    Image-Render Blender Human add-on module documentation
    Copyright 2022 Robert Bosch GmbH and its subsidiaries

    This work is licensed under the

        Creative Commons Attribution-ShareAlike 4.0 International License.

    To view a copy of this license, visit
        http://creativecommons.org/licenses/by-sa/4.0/
    or send a letter to
        Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

</LICENSE>
--->
<!---
	Copyright (c) 2009, 2018 Robert Bosch GmbH and its subsidiaries.
-->

## Requirements/Preamble

To run anyhuman2 you need HumGen V4 from [HumGenV4 testing fork](https://github.com/mnt1lr/HumGen3D/tree/testing). Using this fork enables you to read JSON files (so called presets) from any path on your local computer. Without this you must place your own presets in the the HumGenV4 assets folder.

## How to generate a human <a name="generate-anyhumans"></a>

The generation of humans with anyhuman2 is based on a configuration dictionary (see [configuration](#anyhuman-configuration)). The dictionary is composed of a regular preset JSON as generated (_as_dict()_) by HumgenV4 and custom part. The overall dict (see e.g. ..\personas\FILE_male.json) consists of dictCustom and dictHumGen_V4, where dictHumGen_V4 is the aforementioned regular HumGenV4 dict.
Since it is cumbersome to specify the whole configuration each time a different human is needed, there exist several parameter generator functions (see ../paramgenerators) to help with the process.

### Configuration Parameters

| key                    | possible values | comments                                                                                                                                                                                                  |
| ---------------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| gender                 | male, female    | Gender of the person to be generated                                                                                                                                                                      |
| bSave                  | True, False     | if True and no sFilePathJSON, file will be saved in personas folder                                                                                                                                       |
| sFilePathJSON          | string, path    | Path to json file, where human configuration dictionary is stored                                                                                                                                         |
| sOpenposeHandLabelFile | string, path    | path to file, which contains information where to attach OpenPose hand bones to the native HumGenV4 skeleton                                                                                              |
| bFacialRig             | True, False     | If True, a native HumGenV4 facial rig is added to the armature                                                                                                                                            |
| sWFLWLableFile         | string, path    | path to file, which contains information where to attach [WFLW facial landmarks](https://wywu.github.io/projects/LAB/WFLW.html) to the native HumGenV4 skeleton                                           |
| sIMSLabels             | string, path    | path to file, which contains information where to attach [ISS facial landmarks](https://inside-docupedia.bosch.com/confluence/x/ukSjY) to the native HumGenV4 skeleton                                    |
| sEyebrowLabelsPath     | string, path    | Different eye brow styles lead to different attachment of the WFLW eye brow landmarks (33...50 in [WFLW facial landmarks](https://wywu.github.io/projects/LAB/WFLW.html)) to the native HumGenV4 armature |

### sMode: Random Realistic

Creates a new human where some parameters are fixed (not randomized) in order to generate a realistic looking human.
See src\anyhuman2\paramgenerators\random_realistic.py

```json
{
  "sDTI": "/catharsys/blender/generate/object/hum-gen-3d:2.0", // watch out to set sDTI to 2.0 to use Anyhuman2
  "sId": "Armature.001",
  "xSeed": "1",
  "sMode": "RANDOM_REALISTIC",
  "mParamConfig": {
    "gender": "male", // optional;
    "bSave": "True", // optional, if True and no sFilePathJSON, file will be saved in personas folder
    "sFilePathJSON": "...\\src\\anyhuman2\\personas\\test.json" // change path, optional
  },
  "lCollectionHierarchy": ["Persons"],
  "lModifiers": [
    // ...
  ]
}
```

### sMode: Random Full

Creates a new human all parameters are randomized.

```json
{
  "sDTI": "/catharsys/blender/generate/object/hum-gen-3d:2.0",
  "sId": "Armature.001",
  "xSeed": "1",
  "sMode": "RANDOM_FULL",
  "mParamConfig": {
    "gender": "female",
    "bSave": "True", // optional, if True and no sFilePathJSON, file will be saved in personas folder
    "sFilePathJSON": "/src/anyhuman2/personas/test.json" // change path, optional
  },
  "lCollectionHierarchy": ["Persons"],
  "lModifiers": [
    // ...
  ]
}
```

### sMode: Persona

### sMode: File

Creates a human from a anyhuman dictionary. Reads the dictionary from file specified by:

```json
{
  "sId": "Seth",
  "sMode": "FILE",
  "mParamConfig": {
    "sFilename": "/src/anyhuman2/personas/FILE_male.json" // change path
  }
}
```

### 2. Full Random Human (see src\anyhuman2\paramgenerators\random_full.py):

To randomize a HumGen you can use dependent parameters such as **age** or independent parameters such as e.g **Forearm Thickness**. In other words setting the age, sets preset values to parameters like **Aged male**

For domain randomization, it is reasonable to create completely random anyhumans. The parameters of the human will be varied over the whole range of valid values, resulting sometimes in funny and questionable configurations.

```json
{
  "sId": "Armature.001",
  "sMode": "RANDOM_FULL",
  "mParamConfig": {
    "sGender": "female",
    "bOpenPoseHandLabels": true,
    "sOpenposeHandLabelFile": "OpenPoseHandLabel",
    "bFacialRig": true,
    "sWFLWLableFile": "WFLWLabel",
    "sIMSLabels": "labelling/mapping/IMS_bones.json"
  }
}
```

### 3. Realistic Random (src\anyhuman2\paramgenerators\random_realistic.py):

Similar to fully random parameter generation. However, the parameters are to be expected in a more realistic range and the generated humans are supposed to be believable.

```json
      sMode: "RANDOM_REALISTIC",
      mParamConfig: {
        // sGender: "$rand.choice{$*{[`male`, `female`]}}",
        "sGender": "$rand.choice{$rndX, $lGender}",
        "bSave": true,
        "sFilePathJSON": "${path-trg}", // /config_$iIndex.json def of $iIndex see gen.json5
        "sOpenposeHandLabelFile": "/src/anyhuman2//labelling/mapping/openpose_hand_humgen.json",
        "bFacialRig": true,
        "sWFLWLableFile": "/src/anyhuman2//labelling/mapping/WFLW_labels_anyhuman.json",
        "sIMSLabels": "/src/anyhuman2//labelling/mapping/IMS_bones.json",
        "sEyebrowLabelsPath": "/src/anyhuman2/labelling/mapping/eyebrows/",
      },
```

### 5. File:

Also, anyhuman configurations can be loaded from a json file:

```json
{
  "sDTI": "/catharsys/blender/generate/object/hum-gen-3d:1.0",
  "sId": "Dave",
  "xSeed": "1",
  "sMode": "FILE",
  "mParamConfig": {
    "sFilename": "./personas/FILE_male.json"
  },
  "lCollectionHierarchy": ["Persons"],
  "lModifiers": [
    // ...
  ]
}
```

### Definitions

#### anyhuman dictionary (_dictAnyHuman_)

Static dictionary describing a human, consisting of a custom dictionary (_dictCustom_) and a standard humgenv4 dictionary (_dictHumGen_V4_). _dictHumGen_V4_ can be directly read into humgenv4. _dictCustom_ needs a custom parser.

```json
dictAnyHuman =     "dictCustom": {
        "sGender": "male",
        "sArmatureName": "Armature.001",
        "bOpenPoseHandLabels": true,
        "sOpenposeHandLabelFile": "labelling//mapping//openpose_hand_humgen.json",
        "bFacialRig": true,
        "sWFLWLableFile": "labelling//mapping//WFLW_labels_anyhuman.json",
        "sIMSLabels":"labelling//mapping//IMS_bones.json",
        "sEyebrowLabelsPath":"labelling//mapping//eyebrows//",
        "sEyebrowStyle":"Eyebrows_007",
        "sPoseFilename": {
            "set": null
        },
```

#### dBeardLength dictionary (_dBeardLength_)

Is a dynamic dictionary containing information about the particle system used to create a beard. The structure of _dBeardLength_ depends on the beard used. See ./hair/face_hair/Beard
E.g. ./hair/face_hair/Beard/Full_Beard_1.json

```json{
    "blend_file": "HG_FH_Collection_1.blend",
    "hair_systems": {
        "fh_Mustache_High": {
            "length": 0.5,
            "children_amount": 20
        },
        "fh_Side_Lines": {
            "length": 0.6392694115638733,
            "children_amount": 20
        },
        "fh_soul_patch": {
            "length": 1.0,
            "children_amount": 10
        },
        "fh_sideburns_low": {
            "length": 0.981506884098053,
            "children_amount": 20
        },
        "fh_beard_medium": {
            "length": 1.0,
            "children_amount": 10
        },
        "fh_neck": {
            "length": 1.0,
            "children_amount": 60
        }
    }
}```
