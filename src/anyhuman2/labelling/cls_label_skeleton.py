"""WORK IN PROGRESS

Returns
-------
_type_
    _description_

Raises
------
RuntimeError
    _description_
"""


import json
from mathutils import Vector
import bpy
from HumGen3D import Human


class BoneLabel:
    def __init__(self, _human: Human):
        """
        A LabelBone is a label attached to a bone or vertex group and identifies itself as landmark.
        It can be openpose hand label bone and/or WFLW label bone and so on
        """

        self.objRig = _human.objects.rig  # bpy.data.objects["HG_XXXX"]
        self.objArmature = _human.objects.rig.data  # bpy.data.armatures["metarig"]
        self.objHGBody = _human.objects.body  # bpy.data.armatures["HG_Body"]
        self.objEyes = _human.objects.eyes
        print(f"INFO: Working on Human:  {_human.name} ")

        self.lOpenPoseHandLabels = []

    # ************************************* BEGIN HAND LABELS ********************************************************

    def LoadHandMappings(self, _sHandLabelsFile: str):
        try:
            with open(_sHandLabelsFile, "r") as json_file:
                lOpenPoseHandLabels = json.load(json_file)
                return lOpenPoseHandLabels
                # print(f"{len(self.label_config.lOpenPoseHandLabels)} labels found for hand mapping")
        except FileNotFoundError:
            print(f"File not found: {_sHandLabelsFile}")
            return []

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file: {e}")
            return []

    # enddef

    def AddHandLabels(self, _sLabelFile: str, _objArmature: bpy.types.Armature, _objRig: bpy.types.Object):
        self.lOpenPoseHandLabels = self.LoadHandMappings(_sLabelFile)
        if len(self.lOpenPoseHandLabels) is None:
            return
        if _objArmature is None:
            print("Error _objArmature not found")
            return

        bpy.context.view_layer.objects.active = _objRig
        bpy.ops.object.mode_set(mode="EDIT")

        for i in self.lOpenPoseHandLabels:
            sParent = i["sBone"]
            sAttachTo = i["sAttachTo"]
            sOpenposeLabel = i["sLabelBone"]

            parent_bone = _objArmature.edit_bones.get(str(sParent))
            openpose_mark_bone = _objArmature.edit_bones.get(str(sOpenposeLabel))
            if parent_bone is None:
                print(f"Error: {sParent} not found")
                return
            # endif parent_bone is None:
            if openpose_mark_bone is not None:
                print(f"Error: {sOpenposeLabel} already present")
                continue
            # endif openpose_mark_bone is not None:
            if sAttachTo == "head":
                new_bone = _objArmature.edit_bones.new(str(sOpenposeLabel))
                print(f"{sParent}\t{sAttachTo}\t{sOpenposeLabel}")
                # parent_bone = armature.edit_bones.get(str(sParent))
                new_bone.parent = parent_bone
                new_bone.head = parent_bone.head
                new_bone.tail = parent_bone.head.cross(Vector((1, 1, 1)))
                new_bone.length = 0.01  # in meters
                # new_bone.use_connect = True
            # endif sAttachTo == 'head':
            else:
                new_bone = _objArmature.edit_bones.new(str(sOpenposeLabel))
                print(f"{sParent}\t{sAttachTo}\t{sOpenposeLabel}")
                new_bone.parent = parent_bone
                new_bone.head = parent_bone.tail
                new_bone.tail = parent_bone.tail.cross(Vector((1, 1, 1)))
                new_bone.length = 0.01  # in meters
                new_bone.use_connect = True
            # endelse
        # endfor
        # TODO: set to original/previous mode
        bpy.ops.object.mode_set(mode="OBJECT")
        return

    # enddef
    # ******************************************** END HAND LABELS ****************************************************

    # ************************************* BEGIN IMPORT FUNCTIONS ****************************************************

    def UpdateEyebrowLabels(self, _sEyebrowStyle: str, _labelFile: str):
        try:
            with open(_labelFile, "r") as jsonFile:
                dictEyebrow = json.load(jsonFile)
            self.CreateVertexGroups(
                _objMesh=self.objHGBody, _lLabelVertices=dictEyebrow["lLabelVertices"], _replaceExisting=True
            )
        except FileNotFoundError as e:
            print(f"{e}")

    # enddef

    def ImportSkeletonData(self, _sSkeletonDataFile, _replaceVertexGroups):
        # STEP 1: Parse input json and extract skeletal bones with constraints and vertex groups
        dicSkeleton = self.ParseSkeleton(_sInSkeletonFile=_sSkeletonDataFile)
        # STEP 2: Add vertex groups to objHGBody (Mesh)
        if "lLabelVertices" in dicSkeleton:
            lLabelVertices = dicSkeleton["lLabelVertices"]
        else:
            lLabelVertices = []
        self.CreateVertexGroups(
            _objMesh=self.objHGBody,
            _lLabelVertices=lLabelVertices,
            _replaceExisting=_replaceVertexGroups,
        )
        # STEP 3: Add bones
        self.ImportSkeletonBones(_dicSkeleton=dicSkeleton, _objArmature=self.objArmature, _objRig=self.objRig)
        # STEP 4: Add constraints
        self.AddConstraints(_dicSkeleton=dicSkeleton, _objArmature=self.objArmature, _objRig=self.objRig)

        return {"FINISHED"}

    # add bone based on contents of _dicBone
    def AddBone(self, _sSkeletonType, _dicBone, _objArmature, _objRig):
        bpy.context.view_layer.objects.active = _objRig
        bpy.ops.object.mode_set(mode="EDIT")

        sNewBoneName = "AT.Label;" + _sSkeletonType + ";" + _dicBone["sName"]
        objNewBone = _objArmature.edit_bones.get(str(sNewBoneName))
        if objNewBone is not None:
            print("Error: %s already present - only updating envelope_distance & head_radius", str(sNewBoneName))
            # _objArmature.edit_bones.remove(objNewBone)
            objNewBone.envelope_distance = _dicBone["fEnvelope"]
            objNewBone.head_radius = _dicBone["fHeadRadius"]
            return
        objNewBone = _objArmature.edit_bones.new(sNewBoneName)

        xParentBone = _objArmature.edit_bones.get(_dicBone["sParent"])
        objNewBone.parent = xParentBone if xParentBone else None

        lHead = Vector((_dicBone["lHead"][0], _dicBone["lHead"][1], _dicBone["lHead"][2]))
        lTail = Vector((_dicBone["lTail"][0], _dicBone["lTail"][1], _dicBone["lTail"][2]))
        objNewBone.head = lHead
        objNewBone.tail = lTail
        objNewBone.envelope_distance = _dicBone["fEnvelope"]
        objNewBone.head_radius = _dicBone["fHeadRadius"]
        # bpy.ops.object.mode_set(mode="POSE")

    # enddef

    def AddConstraints(self, _dicSkeleton, _objArmature, _objRig):
        # pose mode
        bpy.context.view_layer.objects.active = _objRig
        bpy.ops.object.mode_set(mode="POSE")

        for dicBone in _dicSkeleton["lBones"]:
            # construct bone name
            sPoseBoneName = "AT.Label;" + dicBone["sType"] + ";" + dicBone["sName"]

            # get pose bone
            objPoseBone = _objRig.pose.bones[sPoseBoneName]

            # add constraints
            # TODO: Implement switch (lConstraints) statement for faster execution
            for objConstraint in dicBone["lConstraints"]:
                sContraintType = objConstraint["sType"]
                print(f"Adding constraint {sContraintType} for posebone {sPoseBoneName}")
                if sContraintType == "STRETCH_TO":
                    self.AddConstraintStretchTo(objPoseBone, objConstraint)
                elif sContraintType == "LIMIT_LOCATION":
                    self.AddConstraintLimitLocation(objPoseBone, objConstraint)
                elif sContraintType == "CHILD_OF":
                    self.AddConstraintChildOf(objPoseBone, objConstraint)
                elif sContraintType == "COPY_LOCATION":
                    self.AddConstraintCopyLocation(objPoseBone, objConstraint)
                else:
                    print(f"Error: constraint {sContraintType} not implemented yet")
            # endfor

        # get into object mode
        bpy.ops.object.mode_set(mode="OBJECT")

    # enddef

    # add STRETCH_TO constraint
    def AddConstraintStretchTo(_self, _xPoseBone, _lStretchConstraint):
        xNewConstraint = _xPoseBone.constraints.new("STRETCH_TO")
        xNewConstraint.target = bpy.data.objects[_lStretchConstraint["target"]]
        xNewConstraint.subtarget = _lStretchConstraint["subtarget"]
        xNewConstraint.keep_axis = _lStretchConstraint["sKeepAxis"]
        xNewConstraint.volume = _lStretchConstraint["sVolume"]
        xNewConstraint.influence = _lStretchConstraint["fInfluence"]

    # enddef

    # add LIMIT_LOCATION constraint
    def AddConstraintLimitLocation(self, _xPoseBone, _lLimitLocationConstraint):
        xNewConstraint = _xPoseBone.constraints.new("LIMIT_LOCATION")
        xNewConstraint.target = self.objHGBody
        xNewConstraint.type = _lLimitLocationConstraint["sType"]
        xNewConstraint.max_x = _lLimitLocationConstraint["fMaxX"]
        xNewConstraint.max_y = _lLimitLocationConstraint["fMaxY"]
        xNewConstraint.max_z = _lLimitLocationConstraint["fMaxZ"]
        xNewConstraint.min_x = _lLimitLocationConstraint["fMinX"]
        xNewConstraint.min_y = _lLimitLocationConstraint["fMinY"]
        xNewConstraint.min_z = _lLimitLocationConstraint["fMinZ"]
        xNewConstraint.use_transform_limit = _lLimitLocationConstraint["bUseTransformLimit"]
        xNewConstraint.owner_space = _lLimitLocationConstraint["sOwnerSpace"]
        xNewConstraint.influence = _lLimitLocationConstraint["fInfluence"]

    # enddef

    # add CHILD_OF constraint
    def AddConstraintChildOf(self, _xPoseBone, lChildOfConstraint):
        xNewConstraint = _xPoseBone.constraints.new("CHILD_OF")
        xNewConstraint.target = self.objHGBody
        xNewConstraint.subtarget = lChildOfConstraint["sSubtarget"]
        xNewConstraint.keep_axis = lChildOfConstraint["sKeepAxis"]
        xNewConstraint.volume = lChildOfConstraint["sVolume"]
        xNewConstraint.influence = lChildOfConstraint["fInfluence"]

    # enddef

    # NOTE: Only supported targets = HG_Body & HG_Eyes
    def AddTarget(self, _lConstraintCopyLocation):
        sTarget = _lConstraintCopyLocation["sTarget"]
        if sTarget.startswith("HG_Body"):
            return self.objHGBody
        elif sTarget.startswith("HG_Eyes"):
            return self.objEyes
        else:
            print(f"{sTarget} not found")
            return None

    # enddef

    # add COPY_LOCATION constraint
    def AddConstraintCopyLocation(self, _xPoseBone, _lConstraintCopyLocation):
        xNewConstraint = _xPoseBone.constraints.new("COPY_LOCATION")
        xNewConstraint.target = self.AddTarget(_lConstraintCopyLocation)
        xNewConstraint.subtarget = _lConstraintCopyLocation["sSubtarget"]
        xNewConstraint.use_x = _lConstraintCopyLocation["bUseX"]
        xNewConstraint.use_y = _lConstraintCopyLocation["bUseY"]
        xNewConstraint.use_z = _lConstraintCopyLocation["bUseZ"]
        xNewConstraint.invert_x = _lConstraintCopyLocation["bInvertX"]
        xNewConstraint.invert_y = _lConstraintCopyLocation["bInvertY"]
        xNewConstraint.invert_z = _lConstraintCopyLocation["bInvertZ"]
        xNewConstraint.target_space = _lConstraintCopyLocation["sTargetSpace"]
        xNewConstraint.owner_space = _lConstraintCopyLocation["sOwnerSpace"]
        xNewConstraint.influence = _lConstraintCopyLocation["fInfluence"]

    # enddef

    # Import bones and add to rig
    def ImportSkeletonBones(self, _dicSkeleton, _objArmature, _objRig):
        try:
            sSkeletonType = _dicSkeleton["sSkeletonType"]
            for dicBone in _dicSkeleton["lBones"]:
                self.AddBone(sSkeletonType, dicBone, _objArmature, _objRig)
        except ValueError as e:
            print(f"ERROR: {e}")

    # enddef

    # import skeleton from json file
    def ParseSkeleton(self, _sInSkeletonFile):
        try:
            with open(_sInSkeletonFile, "r") as sJsonFile:
                dicSkeleton = json.load(sJsonFile)
            return dicSkeleton
        except FileNotFoundError:
            print(f"{_sInSkeletonFile} not found")
            return {}

    # enddef

    def AddVertexGroupToMesh(self, _objMesh, _dicVertexGroup: dict, _replaceExisting=True):
        try:
            for dictLabel in _dicVertexGroup["lLabels"]:
                sVertexGroup = dictLabel["sName"]
                lVertices = dictLabel["lVertices"]
                iIndex = _objMesh.vertex_groups.find(sVertexGroup)
                if iIndex == -1:
                    print(f"Creating vertex group {sVertexGroup} in {_objMesh.name}")
                    xVertexGroup = _objMesh.vertex_groups.new(name=sVertexGroup)
                    xVertexGroup.add(lVertices, 1.0, "ADD")
                elif _replaceExisting:
                    print(f"Vertex Group {sVertexGroup} already exists in {_objMesh.name}, replacing as specified")
                    _objMesh.vertex_groups.remove(_objMesh.vertex_groups[iIndex])
                    xVertexGroup = _objMesh.vertex_groups.new(name=sVertexGroup)
                    xVertexGroup.add(lVertices, 1.0, "ADD")
                else:
                    print(f"Vertex Group {sVertexGroup} already exists in {_objMesh.name}")
                    pass
        except ValueError as e:
            print(f"ERROR: {e}")

    # enddef

    # import and create vertex groups
    def CreateVertexGroups(self, _objMesh, _lLabelVertices, _replaceExisting=True):
        for dicVertexGroup in _lLabelVertices:
            sObject = dicVertexGroup["sObject"]
            try:
                if sObject.startswith("HG_Body"):
                    self.AddVertexGroupToMesh(
                        _objMesh=self.objHGBody, _dicVertexGroup=dicVertexGroup, _replaceExisting=True
                    )
                elif sObject.startswith("HG_Eyes"):
                    self.AddVertexGroupToMesh(
                        _objMesh=self.objEyes, _dicVertexGroup=dicVertexGroup, _replaceExisting=True
                    )
                else:
                    print(f"Logic not implemented for Mesh/Object: {sObject}")
            except ValueError as e:
                print(f"ERROR: {e}")
        # endfor

    # enddef


# ************************************* END IMPORT FUNCTIONS ********************************************************
