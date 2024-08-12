"""
File: ops_face_randomization.py
Project: anyhuman
This Code is experimental research code and is not suitable for production use!
File Created: Monday, 12th August 2024 4:10:21 pm
Author: Jochen Kall (Jochen.Kall@de.bosch.com)
-----
Copyright 2022 - 2024 Robert Bosch GmbH
"""

from anybase import convert


def RandomizeFace(Collection, args, sMode, **kwargs):
    """
    Randomize the faces of all humgen v4 models in the collection.
    Only for humgen v4 humans with face controls

    Modifier arguments:
        fRandomizationStrength: float in [0,1] for the strength of the randomization, i.e.
            how extreme the facial expressions are supposed to be
        iSeed: seed for randomization

    Parameters
    ----------
    Collection : Blender Collection containing the humans to be modified
    args : dict
        Dictionary with configuration arguments
    """

    # Extract modifier parameters
    fRandomizationStrength = convert.DictElementToFloat(args, "fRandomizationStrength", fDefault=0.4)
    iSeed = convert.DictElementToInt(args, "iSeed", iDefault=0)

    print("I bins, de Batman!!!")
    print(fRandomizationStrength)
    print(iSeed)
