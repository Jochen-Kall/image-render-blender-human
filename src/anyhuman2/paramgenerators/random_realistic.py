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
from .GeneralRandomParameters import GeneralRandomParameters

############################################################################################
def RealisticRandomizeParams(params, generator_config, rnd):
    """
    Create a set of completely random realistic parameters for human generation.
    This randomizer is intended for the generation of visually
    plausible humans.

    The config dict can contain:
    - 'gender': either 'male' or 'female'

    If gender is not given, it will be selected randomly.


    Parameters
    ----------
    params : dict
        set of parameters controlling the randomization. E.g.
                {
                    "sId": "Armature.002",
                    "sMode": "RANDOM_REALISTIC",
                    "mParamConfig": {"sGender": "male"},
                }
    generator_config: dict
        set of all available parameters in HumGenV4

    Returns
    -------
    dict
        Dictionary of parameters for human generator
    """


    universal_params = GeneralRandomParameters(params, generator_config, rnd)
    dPreset = universal_params.GetPreset()
    sArmatureName = universal_params.ArmatureName()
    Male, dFaceHair, dBeardLength, sRegularHair, sEyebrows = universal_params.RandomizeHair()
    sGender = universal_params.GetGender()
    height_150, height_200, height = universal_params.RandomizeHeight()
    outfit = universal_params.RandomizeOutfit()
    sFootwear = universal_params.RandomFootwear()
    sSkinTexture = universal_params.RandomizeSkin()
    # HumGenV4 Config
    NewHumGenV4Config = dPreset
    # Keys
    NewHumGenV4Config["keys"]["Forearm Length"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Forearm Thickness"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Hand Length"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Hand Thickness"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Hand Width"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Upper Arm Length"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Upper Arm Thickness"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Neck Length"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Neck Thickness"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Foot Length"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Shin Length"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Shin Thickness"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Thigh Length"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Thigh Thickness"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Back Muscles"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Biceps"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Calves Muscles"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Chest Muscles"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Forearm Muscles"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Hamstring Muscles"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Lower Butt Muscles"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Quad Muscles"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Shoulder Muscles"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Traps Muscles"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Triceps"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Upper Butt Muscles"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Stylized"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Belly Size"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Breast Size"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Chest Height"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Chest Width"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Hips Height"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Hips Size"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Shoulder Width"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["Waist Thickness"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["keys"]["cheek_fullness"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["cheek_zygomatic_bone"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["cheek_zygomatic_proc"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["chin_dimple"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["chin_height"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["chin_size"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["chin_width"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["ear_antihelix_shape"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["ear_height"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["ear_lobe_size"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["ear_turn"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["ear_width"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["Eye Depth"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["Eye Distance"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["Eye Height"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["eyelid_fat_pad"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["eyelid_rotation"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["eyelid_shift_horizontal"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["eyelid_shift_vertical"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["eye_height"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["eye_orbit_size"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["eye_tilt"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["eye_width"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["jaw_location_horizontal"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["jaw_location_vertical"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["jaw_width"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["muzzle_location_horizontal"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["muzzle_location_vertical"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["lip_cupid_bow"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["lip_height"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["lip_location"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["lip_offset"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["lip_width"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["nose_angle"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["nose_bridge_height"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["nose_bridge_width"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["nose_height"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["nose_location"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["nose_nostril_flare"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["nose_nostril_turn"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["nose_tip_angle"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["nose_tip_length"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["nose_tip_size"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["nose_tip_width"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["Eye Scale"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["browridge_center_size"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["browridge_loc_horizontal"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["browridge_loc_vertical"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["forehead_size"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    NewHumGenV4Config["keys"]["temple_size"] = universal_params.gauss_with_min_max(-1.0, 1.0)
    # Skin
    NewHumGenV4Config["skin"]["tone"] = universal_params.gauss_with_min_max(0, 3.0)
    NewHumGenV4Config["skin"]["redness"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["skin"]["saturation"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["skin"]["normal_strength"] = universal_params.gauss_with_min_max(0.5, 10)
    NewHumGenV4Config["skin"]["roughness_multiplier"] = universal_params.gauss_with_min_max(0.5, 4.0)
    NewHumGenV4Config["skin"]["freckles"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["skin"]["splotches"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["skin"]["texture.set"] = sSkinTexture
    NewHumGenV4Config["skin"]["cavity_strength"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["skin"]["gender_specific"]["mustache_shadow"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["skin"]["gender_specific"]["beard_shadow"] =  universal_params.gauss_with_min_max(0, 1.0)
    # Eyes
    NewHumGenV4Config["eyes"]["pupil_color"] = [universal_params.gauss_with_min_max(0, 1.0), universal_params.gauss_with_min_max(0, 1.0), universal_params.gauss_with_min_max(0, 1.0), 1.00]
    NewHumGenV4Config["eyes"]["sclera"] = [universal_params.gauss_with_min_max(0, 1.0), universal_params.gauss_with_min_max(0, 1.0), universal_params.gauss_with_min_max(0, 1.0), 1.00]
    # Height
    NewHumGenV4Config["height"]["set"] = height
    NewHumGenV4Config["keys"]["height_150"] = height_150
    NewHumGenV4Config["keys"]["height_200"] = height_200
    # Randomize Eye brows
    NewHumGenV4Config["hair"]["eyebrows"]["set"] = sEyebrows
    NewHumGenV4Config["hair"]["eyebrows"]["lightness"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["hair"]["eyebrows"]["redness"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["hair"]["eyebrows"]["roughness"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["hair"]["eyebrows"]["salt_and_pepper"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["hair"]["eyebrows"]["roots"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["hair"]["eyebrows"]["root_lightness"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["hair"]["eyebrows"]["root_redness"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["hair"]["eyebrows"]["roots_hue"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["hair"]["eyebrows"]["fast_or_accurate"] = 1
    NewHumGenV4Config["hair"]["eyebrows"]["hue"] = universal_params.gauss_with_min_max(0, 1.0)
    # Randomize Regular Hair
    NewHumGenV4Config["hair"]["regular_hair"]["set"] = sRegularHair
    NewHumGenV4Config["hair"]["regular_hair"]["lightness"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["hair"]["regular_hair"]["redness"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["hair"]["regular_hair"]["roughness"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["hair"]["regular_hair"]["salt_and_pepper"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["hair"]["regular_hair"]["roots"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["hair"]["regular_hair"]["root_lightness"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["hair"]["regular_hair"]["root_redness"]= universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["hair"]["regular_hair"]["roots_hue"] = universal_params.gauss_with_min_max(0, 1.0)
    NewHumGenV4Config["hair"]["regular_hair"]["fast_or_accurate"] = 1.0
    NewHumGenV4Config["hair"]["regular_hair"]["hue"] = universal_params.gauss_with_min_max(0, 1.0)
    # Face hair
    NewHumGenV4Config["hair"]["face_hair"] = dFaceHair
    # Clothing
    NewHumGenV4Config["clothing"]["outfit"]["set"] = outfit
    # Foot wear
    NewHumGenV4Config["clothing"]["footwear"]["set"] = sFootwear


    dictAnyHuman = {"dictCustom":
             {
                "sGender": sGender,
                "sArmatureName" : sArmatureName,
                "bOpenPoseHandLabels": False,
                "sOpenposeHandLabelFile": params['mParamConfig'].get('sOpenposeHandLabelFile', 'OpenPoseHandLabel'),
                "bFacialRig": True,
                "sWFLWLableFile": params['mParamConfig'].get('sWFLWLableFile', 'WFLWLabel'),
                "sIMSLabels": params['mParamConfig'].get('sIMSLabels', 'IMSLabel'),
                # "sEyebrowLabelsPath": params['mParamConfig'].get('sEyebrowLabelsPath', None),
                "sEyebrowStyle": sEyebrows,
                "sPoseFilename": None,
                "dBeardLength" : dBeardLength,
            },
        "dictHumGen_V4": NewHumGenV4Config
    }
    return dictAnyHuman


# enddef
