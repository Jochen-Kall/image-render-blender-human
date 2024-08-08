# <LICENSE id="GPL-3.0">
#
#   Image-Render Blender Human add-on module
#   Copyright (C) 2022 Robert Bosch GmbH and its subsidiaries
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#
# </LICENSE>
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# -----
# Copyright (c) 2022 Robert Bosch GmbH and its subsidiaries.
# All rights reserved.
# -----
###


from . import file
from . import persona
from . import random_full
from . import random_realistic
import os
import json
import datetime
from pathlib import Path
# from . import zwicky


######################################################################
def ComputeParams(mode, params, overwrite, generator_params, rnd) -> dict:
    """
    Computes a set of parameters for human generation.
    Currently available modes are:
    - PERSONA: for a set of names ('alice', 'bob',  ...), return predefined values
    - RANDOM_FULL: randomize every parameter of its possible range
    - RANDOM_REALISTIC: randomize every parameter but within realistically apearing bounds
    - ZWICKY: randomize based on a Zwicky box like description
    - FILE: specify a path to a json file with predefined values
    -

    Parameters
    ----------
    mode : string
        Mode for parameter computation, see above
    params : dict
        dictionary of parameters for the mode, see the implementation of the mode for details
    overwrite : dict
        dict of parameters that shall overwrite the computed values

    Returns
    -------
    dict
        dictionary with parameters for human generation

    """
    new_params:dict = GetParams(mode, params, generator_params, rnd)

    return new_params


# enddef


######################################################################
def GetParams(mode, params, generator_params, rnd):
    """
    Computes a set of parameters for human generation.
    Currently available modes are:
    - PERSONA: creates human from preset humans which are part of the humgenv4 package,
                such as e.g. 'anna' (./models/female/Caucasian/Anna.json)  and 'david' (./models/male/Caucasian/David.json)
    - RANDOM_FULL: randomize every parameter of its possible range
    - RANDOM_REALISTIC: randomize every parameter but within realistically appearing bounds
    - ZWICKY: randomize based on a Zwicky box like description
    - FILE: Reads a dictAnyhuman JSON which was exported using TODO: mention the export function

    Parameters
    ----------
    mode : string
        Mode for parameter computation, see above
    params : dict
        dictionary of parameters for the mode, see the implementation of the mode for details
    generator_params: dict
        dictionary with humgenv4 internal dictionaries, see cls_humgen.

    Returns
    -------
    dict
        dictionary with parameters for human generation

    """
    new_params = {}

    if mode == "RANDOM_FULL":
        new_params = random_full.FullyRandomizeParams(params, generator_params, rnd)
    elif mode == "RANDOM_REALISTIC":
        new_params = random_realistic.RealisticRandomizeParams(params, generator_params, rnd)
    elif mode == "ZWICKY":
        new_params = zwicky.ZwickyParams(params, generator_params)
    elif mode == "PERSONA":
        new_params = persona.PersonaParams(params, generator_params)
    elif mode == "FILE":
        new_params = file.FileParams(params)
    else:
        raise NotImplementedError(f"Please specify a valid mode for anyhuman parameter generation, not {mode}")

    return new_params

def SaveGeneratedParams(params: dict, generated_params: dict):
    """
    Save the generated parameters as a JSON file.

    Args:
        params (dict): A dictionary containing the parameters.
        generated_params (dict): A dictionary containing the generated parameters.

    Returns:
        None
    """
    try:
        saveJSON: dict = params["mParamConfig"].get("bSave", False)
        filepath: dict = params["mParamConfig"].get("sFilePathJSON", {"sFilePathJSON": Path(__file__).resolve().parent})
        current_datetime: str = datetime.datetime.now()
        filename = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        filepath_new = Path(str(filepath) + "/" + f"human_{filename}.json")
        # filepath_new = filepath_new.as_posix()
        if saveJSON is True:
            counter = 0
            while filepath_new.exists():
                file = f"{filepath_new.stem}_{counter:04d}.json"
                with open(file, "w") as json_file:
                    json.dump(generated_params, file)
                counter += 1
            if not filepath_new.exists():
                with open(filepath_new, "w") as json_file:
                    json.dump(generated_params, json_file)
        else:
            pass
    except KeyError as e:
        print(f"{e}: Could not save the generated parameters as a JSON file.")
        pass


