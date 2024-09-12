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

import bpy
import bmesh

import mathutils
import math
from mathutils import Vector
import numpy as np
from scipy.spatial import KDTree

from mathutils.geometry import intersect_ray_tri
import trimesh


class anyhuman_gen_data:
    def __init__(self, anyhumanversion="gen-07"):
        if anyhumanversion == "gen-07":
            self.ignore_bones = {"eye_settings.R", "eye_settings.L"}
            self.tongue_bone_name = "tongue_out_lt_rt_up_dwn"
            self.eye_gaze_bone_name = "eyeball_lookat_master"

            self.iMarkers_right = [
                5008,
                5055,
                5101,
                5749,
                5796,
                5826,
                7676,
                7797,
                7811,
                3648,
                3625,
                3662,
                3580,
                3733,
                3541,
                3827,
                3868,
                3874,
                3903,
                9112,
                2178,
                2184,
                4730,
                4743,
                4755,
                4824,
                4856,
                8900,
                8958,
                8979,
                9799,
                9845,
                1898,
                1917,
                1957,
                2012,
                2245,
                2293,
                4687,
            ]
            # sparse correlation matrix among face shape bones (in range [-1..1]),
            # if no entry in this map it is assumed to be zero,
            # negative correlation means the corresponding bones behave opposite
            # (e.g. when one eye is open the other is closed)
            self.face_bone_correlation = [
                ("eye_blink_open_L", "eye_blink_open_R", 0.6),
                ("brow_outer_up_L", "brow_outer_up_R", 0.6),
                ("brow_dwn_L", "brow_dwn_R", 0.6),
                ("eye_squint_L", "eye_squint_R", 0.8),
                ("mouth_smile_frown_L", "mouth_smile_frown_R", 0.5),
                ("mouth_lower_down_L", "mouth_lower_down_R", 0.3),
                ("mouth_dimple_L", "mouth_dimple_R", 0.3),
                ("cheek_squint_L", "cheek_squint_R", 0.3),
                ("nose_sneer_L", "nose_sneer_R", 0.5),
            ]
            self.eye_left_center_vertex_idx = [
                4182,
                4247,
            ]  # the vertex indices of the eye mesh giving the gaze direction
            self.eye_right_center_vertex_idx = [
                313,
                1603,
            ]  # the vertex indices of the eye mesh giving the gaze direction

            # these vertices represent the inner mouth and are the same in each HumGen model (version 3)
            self.iIgnore_body_vertices = [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
                86,
                87,
                88,
                89,
                90,
                91,
                92,
                93,
                94,
                95,
                96,
                97,
                98,
                99,
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
                115,
                116,
                117,
                118,
                119,
                120,
                121,
                122,
                123,
                124,
                125,
                126,
                127,
                128,
                129,
                130,
                131,
                132,
                133,
                134,
                135,
                136,
                137,
                138,
                139,
                140,
                141,
                142,
                143,
                144,
                145,
                146,
                147,
                148,
                149,
                150,
                151,
                152,
                153,
                154,
                155,
                156,
                157,
                158,
                159,
                160,
                161,
                162,
                163,
                164,
                165,
                166,
                167,
                168,
                169,
                170,
                171,
                172,
                173,
                174,
                175,
                176,
                177,
                178,
                179,
                180,
                181,
                182,
                183,
                184,
                185,
                186,
                187,
                188,
                189,
                190,
                191,
                192,
                193,
                194,
                195,
                196,
                197,
                198,
                199,
                200,
                201,
                202,
                203,
                204,
                205,
                206,
                207,
                208,
                209,
                210,
                211,
                212,
                213,
                214,
                215,
                216,
                217,
                218,
                219,
                220,
                221,
                222,
                223,
                224,
                225,
                226,
                227,
                228,
                229,
                230,
                231,
                232,
                233,
                234,
                235,
                236,
                237,
                238,
                239,
                240,
                241,
                242,
                243,
                244,
                245,
                246,
                247,
                248,
                249,
                250,
                251,
                252,
                253,
                254,
                255,
                256,
                257,
                258,
                259,
                260,
                261,
                262,
                263,
                264,
                265,
                266,
                267,
                268,
                269,
                270,
                271,
                272,
                273,
                274,
                275,
                276,
                277,
                278,
                279,
                280,
                281,
                282,
                283,
                284,
                285,
                286,
                287,
                288,
                289,
                290,
                291,
                292,
                293,
                294,
                295,
                296,
                297,
                298,
                299,
                300,
                301,
                302,
                303,
                304,
                305,
                306,
                307,
                308,
                309,
                310,
                311,
                312,
                313,
                314,
                315,
                316,
                317,
                318,
                319,
                320,
                321,
                322,
                323,
                324,
                325,
                326,
                327,
                328,
                329,
                330,
                331,
                332,
                333,
                334,
                335,
                336,
                337,
                338,
                339,
                340,
                341,
                342,
                343,
                344,
                345,
                346,
                347,
                348,
                349,
                350,
                351,
                352,
                353,
                354,
                355,
                356,
                357,
                358,
                359,
                360,
                361,
                362,
                363,
                364,
                365,
                366,
                367,
                368,
                369,
                370,
                371,
                372,
                373,
                374,
                375,
                376,
                377,
                378,
                379,
                380,
                381,
                382,
                383,
                384,
                385,
                386,
                387,
                388,
                389,
                390,
                391,
                392,
                393,
                394,
                395,
                396,
                397,
                398,
                399,
                400,
                401,
                402,
                403,
                404,
                4774,
                4775,
                4776,
                4777,
                4778,
                4780,
                4781,
                4859,
                4860,
                4861,
                4862,
                4863,
                8990,
                8991,
                8992,
                8993,
                8994,
                2195,
                2196,
                2197,
                9006,
                9009,
                9010,
                2198,
                9011,
                9012,
                9013,
                9014,
                2199,
                9015,
                9016,
                9760,
                9761,
                9762,
                9764,
                9765,
                9767,
                9768,
                9771,
                9773,
                9822,
                9823,
                9824,
                9825,
                9827,
                9830,
                9831,
                9834,
                9836,
                9846,
                9847,
                9865,
                9866,
                9868,
                9869,
                9914,
                9915,
                9916,
                9917,
                9918,
                9919,
                9921,
                9923,
                9928,
                9929,
                9930,
                9931,
                9932,
                9933,
                9934,
                10024,
                10025,
                10026,
                10027,
                10028,
                10029,
            ]
            return
        raise ("unsupported anyhuman version")


##############################################################################################################
def GetLocConstraints(arm, ignore_bones):
    # first, switch to edit mode
    bpy.context.view_layer.objects.active = arm
    # set object origin to spine to add object constraint to seat empty
    bpy.ops.object.mode_set(mode="POSE")

    mLimits = {}
    for b in arm.pose.bones:
        if b.name in ignore_bones:
            continue
        for c in b.constraints:
            if c.type == "LIMIT_LOCATION":
                mLimits[b.name] = (Vector((c.min_x, c.min_y, c.min_z)), Vector((c.max_x, c.max_y, c.max_z)))
    return mLimits


##############################################################################################################
def RandomFaceExpression(
    object,
    mLimits,
    humgen_settings,
    lCorrelation=[],
    face_expression_sigma=0.4,
    **kwargs,
):
    """Create a random face expression given its HumGen Bone limits and the sparse bone correlation list.

    Args:
        object (_type_): The active HumGen object to be altered (needs a face rig)
        mLimits (dict): the map with bone names as keys and the min and max values for the face bone location.
        lCorrelation (list, optional): The sparse correlation in [-1..1] between 2 bones. Defaults to [].
    """
    # first, switch to edit mode
    bpy.context.view_layer.objects.active = object
    # set object origin to spine to add object constraint to seat empty
    bpy.ops.object.mode_set(mode="POSE")

    shape = (len(mLimits), 3)  # Replace with your desired shape (e.g., (rows, columns))

    # Generate the uniform random array
    # r = np.random.uniform(0, 1, size=shape).as_type(np.float32)
    numParams = shape[0]

    # The new version including the correlation of bones

    # build covariance matrix from correlations and limits (= variances)
    boneCov = np.zeros(shape=(numParams, numParams, 3), dtype=np.float32)
    # the zero pose of the bones is the default rest pose, hence the mean should be all zero
    boneMean = np.zeros(shape=(numParams, 3), dtype=np.float32)
    scaleLow = np.zeros(shape=(numParams, 3), dtype=np.float32)
    scaleUp = np.zeros(shape=(numParams, 3), dtype=np.float32)
    idx2bone = {}
    bone2idx = {}
    for i, (key, limits) in enumerate(mLimits.items()):
        idx2bone[i] = key
        bone2idx[key] = i
        if key in object.pose.bones:
            b = object.pose.bones[key]
            # delta = limits[1] - limits[0]
            l0 = np.float32(limits[0])
            l1 = np.float32(limits[1])
            assert np.all(l1 >= 0) and np.all(l0 <= 0)

            # sigma = np.maximum(l1 - 0, 0 - l0) * face_expression_sigma
            sigma = 1.0  # face_expression_sigma
            scaleLow[i] = 0 - l0  # lower side (negative) scaling
            scaleUp[i] = l1 - 0  # upper side (positive) scaling
            boneCov[i, i] = sigma
        else:
            boneCov[i, i] = np.float32([0.0001, 0.0001, 0.0001])

    for key0, key1, corr in lCorrelation:
        if key0 in bone2idx and key1 in bone2idx:
            i0 = bone2idx[key0]
            i1 = bone2idx[key1]
            assert i0 != i1
            # cov(x,y) = corr(x,y) * var(x) * var(y)
            # cov = corr * boneCov[i0,i0] * boneCov[i1,i1]
            cov = corr  # sigmas are one, hence don't need to scale
            boneCov[i0, i1] = cov
            boneCov[i1, i0] = cov  # must be symmetric
        else:
            print(f"Warning: key {key0} or key {key1} not found in the face bones!")

    # cov = diagonal_standard_deviation_matrix * correlation_matrix * diagonal_standard_deviation_matrix
    boneCov *= (
        face_expression_sigma * face_expression_sigma
    )  # scale the covariance globally by the face expression variance

    sample3D = []
    # sample each of the 3 dimensions independently (no correlation between the axis assumed)
    for dim in range(3):
        boneCovMatrix = boneCov[:, :, dim]  # np.reshape(boneCov, (numParams*3,numParams*3))
        xi = np.random.multivariate_normal(mean=boneMean[:, dim], cov=boneCovMatrix, size=1)
        # print(xi)
        sample3D.append(np.squeeze(xi))
    sample3D = np.stack(sample3D, axis=-1)

    # for (key0,key1,corr) in lCorrelation:
    #     if key0 in bone2idx and key1 in bone2idx:
    #         i0 = bone2idx[key0]
    #         i1 = bone2idx[key1]
    #         print(f"Sample for key {key0} = {sample3D[i0]}, key {key1} = {sample3D[i1]}")

    # since the normal distribution is symmetric but we need an assymetric distribution
    # depending on the lower and upper bounds of the face bone positions
    # we scale the standard deviation of negative and positive side individually
    mask0 = sample3D < 0
    mask1 = sample3D >= 0

    # d[mask0] = scaleLow[mask0] * np.abs(sample3D[mask0])
    # d[mask1] = scaleUp[mask1]  * np.abs(sample3D[mask1])
    sample3D[mask0] = scaleLow[mask0] * sample3D[mask0]
    sample3D[mask1] = scaleUp[mask1] * sample3D[mask1]
    for key, limits in mLimits.items():
        if key in object.pose.bones:
            b = object.pose.bones[key]
            idx = bone2idx[key]
            b.location = Vector(sample3D[idx])


##############################################################################################################
def RandomFacePose(object, head_rot_limits):
    """Set the HumGen object's face rotation randomly inside the limits

    Args:
        object (_type_): HumGen object
        head_rot_limits: Limits for rotation around the three axes in radians
    """
    # first, switch to edit mode
    bpy.context.view_layer.objects.active = object
    # set object origin to spine to add object constraint to seat empty
    bpy.ops.object.mode_set(mode="POSE")

    object.pose.bones["head"].rotation_mode = "XYZ"
    object.pose.bones["head"].rotation_euler[2] = np.clip(
        np.random.normal(0, 0.2 * head_rot_limits[2]), -head_rot_limits[2], head_rot_limits[2]
    )
    object.pose.bones["head"].rotation_euler[1] = np.clip(
        np.random.normal(0, 0.2 * head_rot_limits[1]), -head_rot_limits[1], head_rot_limits[1]
    )
    object.pose.bones["head"].rotation_euler[0] = np.clip(
        np.random.normal(0, 0.2 * head_rot_limits[0]), -head_rot_limits[0], head_rot_limits[0]
    )


##############################################################################################################
def ResetFacePose(object):
    bpy.context.view_layer.objects.active = object
    # set object origin to spine to add object constraint to seat empty
    bpy.ops.object.mode_set(mode="POSE")

    object.pose.bones["head"].rotation_mode = "XYZ"
    object.pose.bones["head"].rotation_euler[2] = 0
    object.pose.bones["head"].rotation_euler[1] = 0
    object.pose.bones["head"].rotation_euler[0] = 0


##############################################################################################################
def ResetFaceExpression(
    object,
    mLimits,
    **kwargs,
):

    # first, switch to edit mode
    bpy.context.view_layer.objects.active = object
    # set object origin to spine to add object constraint to seat empty
    bpy.ops.object.mode_set(mode="POSE")

    shape = (len(mLimits), 3)  # Replace with your desired shape (e.g., (rows, columns))

    # Define the range for the uniform random numbers (you can adjust these values)
    low = 0.0  # Lower bound (inclusive)
    high = 1.0  # Upper bound (exclusive)

    # Generate the uniform random array
    # r = np.random.uniform(0, 1, size=shape).as_type(np.float32)

    for key, limits in mLimits.items():
        if key in object.pose.bones:
            b = object.pose.bones[key]
            delta = limits[1] - limits[0]
            b.location = Vector((0, 0, 0))


##############################################################################################################
def CreateAndEmptyCollection(_col_name: str):
    """Create a collection in blender given its name and if existing clear all objects already contained.

    Args:
        _col_name (str): the name of the collection
    """
    # Create a new collection
    bpy.ops.object.mode_set(mode="OBJECT")
    collection_name = _col_name
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
        # Delete all objects in the collection
        objects_to_delete = [obj for obj in collection.objects]
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects_to_delete:
            obj.select_set(True)
        bpy.ops.object.delete()
    else:
        # If the collection doesn't exist, create a new one
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
    # bpy.context.view_layer.active_layer_collection = collection
    # Set the active collection
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[collection_name]


##############################################################################################################
def SetFacialMarkers(object, faceIndices, **kwargs):
    """Add markers to the object's face by adding small white spheres at the given face vertices.

    Args:
        object (_type_): the humgen model
        faceIndices (_type_): a dictionary of marker names and indices of the face mesh vertices.
    """

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    object.select_set(True)
    bpy.context.view_layer.objects.active = object

    depsgraph = bpy.context.evaluated_depsgraph_get()
    object = object.evaluated_get(depsgraph)

    # if object.type == 'MESH':
    ##bpy.context.view_layer.objects.active = object  # Set the active object
    # bpy.ops.object.mode_set(mode='EDIT')  # Enter edit mode
    # else:
    # print(f"Object '{object_name}' is not an editable mesh.")

    # Deselect all vertices first
    # bpy.ops.mesh.select_all(action='DESELECT')

    # Select specific vertices by their indices
    # bpy.ops.object.mode_set(mode='OBJECT')  # Switch back to object mode

    # bpy.ops.object.mode_set(mode='EDIT')  # Enter edit mode to see vertex selection

    mesh = object.data

    # bm = bmesh.from_edit_mesh(mesh)

    # bpy.ops.object.duplicates_make_real(use_base=True, use_children=True)

    # bpy.ops.mesh.select_mode(type="VERT")
    normalEps = 0.001
    marker_pos = []
    for n, i in faceIndices.items():
        # print("{} -> {}".format(n,i))
        # mesh.vertices[i].select = True
        v = mesh.vertices[i]  # bm.verts[i]
        normalOffset = v.normal * normalEps
        marker_pos.append((n, v.co + normalOffset))

    bpy.ops.object.mode_set(mode="OBJECT")  # Switch back to object mode

    # Create a new collection
    collection_name = "FaceMarkers"
    CreateAndEmptyCollection(collection_name)
    # if collection_name in bpy.data.collections:
    #     collection = bpy.data.collections[collection_name]
    #     # Delete all objects in the collection
    #     objects_to_delete = [obj for obj in collection.objects]
    #     bpy.ops.object.select_all(action='DESELECT')
    #     for obj in objects_to_delete:
    #         obj.select_set(True)
    #     bpy.ops.object.delete()
    # else:
    #     # If the collection doesn't exist, create a new one
    #     collection = bpy.data.collections.new(collection_name)
    #     bpy.context.scene.collection.children.link(collection)

    # #bpy.context.view_layer.active_layer_collection = collection
    # # Set the active collection
    # bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[collection_name]

    for p in marker_pos:
        # Add a sphere at each selected vertex
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.001, location=p[1])
        # Get the newly created sphere object
        sphere = bpy.context.object

        # Link the sphere to the collection
        # collection.objects.link(sphere)
        # Rename the sphere with a unique name
        sphere.name = "Marker_{}".format(p[0])

    # bpy.ops.object.mode_set(mode = 'EDIT')


# enddef

##############################################################################################################


def FindNearestMarkersMirrored(object, iMarkers, mirrorSearchLimit=0.01):
    """Given the HumGen object and the marker mesh vertex indices, find the mirrored vertex
    (assuming the mesh is mirror symmetric)

    Args:
        object (_type_): the object of a humgen assumed to be in the center and mirror axis is X.
        iMarkers (_type_): a container of vertex indices in the object mesh.
        mirrorSearchLimit (float): an epsilon when doing the search.
    Returns:
        _type_: the mirrored vertex indices and -1 for not found
    """
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    object.select_set(True)
    bpy.context.view_layer.objects.active = object

    if object.type == "MESH":
        # bpy.context.view_layer.objects.active = object  # Set the active object
        bpy.ops.object.mode_set(mode="EDIT")  # Enter edit mode
    else:
        print(f"Object '{object_name}' is not an editable mesh.")

    # Deselect all vertices first
    # bpy.ops.mesh.select_all(action='DESELECT')

    # Select specific vertices by their indices
    # bpy.ops.object.mode_set(mode='OBJECT')  # Switch back to object mode

    # bpy.ops.object.mode_set(mode='EDIT')  # Enter edit mode to see vertex selection

    mesh = object.data
    # vertices = [object.matrix_world @ vertex.co for vertex in mesh.vertices]
    vertices = [vertex.co for vertex in mesh.vertices]
    kdtree = KDTree(vertices)

    # bpy.ops.mesh.select_mode(type="VERT")
    marker_pos = []
    for i, idx in enumerate(iMarkers):
        v = mesh.vertices[idx]  # bm.verts[i]
        p = v.co
        p_search = Vector((-p[0], p[1], p[2]))
        dist, nearest_index = kdtree.query(p_search, k=1)
        if idx != nearest_index:
            if dist < mirrorSearchLimit:
                marker_pos.append(nearest_index)
            else:
                print("Could not find the mirrored vertex index!")
                marker_pos.append(-1)
    return marker_pos


##############################################################################################################


def get_modified_mesh(ob, cage=False):
    bm = bmesh.new()
    bm.from_object(
        ob,
        bpy.context.evaluated_depsgraph_get(),
        cage=cage,
    )
    ##bm.transform(ob.matrix_world)
    # me = bpy.data.meshes.new("Deformed")
    # bm.to_mesh(me)
    ##bmesh.ops.triangulate(bm, faces=bm.faces)
    # return me
    return bm


def copyModifiedObj(objName):
    obj = bpy.data.objects[objName]
    # make a modified and deformed copy
    ob = obj.copy()
    bm = get_modified_mesh(ob)
    me = bpy.data.meshes.new("Deformed")
    bm.to_mesh(me)
    ##bmesh.ops.triangulate(bm, faces=bm.faces)
    ob.modifiers.clear()
    ob.data = me
    bpy.context.collection.objects.link(ob)


def IsInBoundingVectors(vector_check, vector1, vector2):
    for i in range(0, 3):
        if (
            vector_check[i] < vector1[i]
            and vector_check[i] < vector2[i]
            or vector_check[i] > vector1[i]
            and vector_check[i] > vector2[i]
        ):
            return False
    return True


##############################################################################################################


def CheckForTongueIntersection(iIgnore_body_vertices):
    """Check if the tongue mesh intersects the human head (except for the inner mouth region)

    Returns:
        bool: if the tongue intersects or not
    """
    obj_tongue = bpy.data.objects["HG_TeethLower"]

    # obj_tongue = obj_tongue.copy()

    me_tongue = get_modified_mesh(obj_tongue)
    bmesh.ops.triangulate(me_tongue, faces=me_tongue.faces)

    bbox = obj_tongue.bound_box
    # for face in me_tongue
    bbmin = np.float32((10000, 10000, 10000))
    bbmax = -bbmin
    for vert in me_tongue.verts:
        bbmin = np.minimum(bbmin, np.float32(vert.co))
        bbmax = np.maximum(bbmax, np.float32(vert.co))
    bbsize = 0.5 * (bbmax - bbmin)
    bbcenter = 0.5 * (bbmax + bbmin)
    # bpy.ops.mesh.primitive_cube_add(location=Vector(bbcenter), scale=Vector(bbsize))
    bbsize_ext = 1.5 * bbsize
    bbox_ext = np.float32([bbcenter - bbsize_ext, bbcenter + bbsize_ext])

    # cycle through all vertices
    obj_body = bpy.data.objects["HG_Body"]
    # make a modified and deformed copy
    # obj_body_cp = obj_body.copy()
    obj_body_cp = obj_body
    me_body = get_modified_mesh(obj_body_cp)
    bmesh.ops.triangulate(me_body, faces=me_body.faces)

    count = len(me_body.verts)
    print(count)
    body_verts = np.fromiter((x for v in me_body.verts for x in v.co), dtype=np.float32, count=len(me_body.verts) * 3)
    body_verts.shape = (len(me_body.verts), 3)
    # print(body_verts.shape)
    # verts = np.empty(count*3, dtype=np.float32)
    # me_body.verts.foreach_get('co', verts)
    # verts.shape = shape
    # print(bbox_ext)

    #    ((p>=x[0]) & (p<=x[1])).all(1)
    result = ((body_verts > bbox_ext[np.newaxis, 0]) & (body_verts < bbox_ext[np.newaxis, 1])).all(1)

    # print(result.shape)
    trial_vert_indices_body = np.squeeze(np.argwhere(result))
    print(trial_vert_indices_body.shape)

    ignoreSet = set(iIgnore_body_vertices)
    trial_vert_indices_body = [x for x in trial_vert_indices_body if x not in ignoreSet]

    # print("#intersecting verts = {}".format(len(np.argwhere(result))))
    # Get the faces using the vertex index
    # vertex_faces = [face for face in me_body.faces if vertex_index in face.vertices]
    me_body.verts.ensure_lookup_table()
    me_body.faces.ensure_lookup_table()
    me_body.edges.ensure_lookup_table()
    # trial_face_idx = [f.index for idx in trial_vert_indices_body for f in me_body.verts[idx].link_faces]
    # for f_idx in trial_face_idx:
    #    face = me_body.faces[f_idx]

    trial_edge_idx = [e.index for idx in trial_vert_indices_body for e in me_body.verts[idx].link_edges]
    trial_edge_idx = list(set(trial_edge_idx))
    intersect = False
    ray_origins = np.zeros((len(trial_edge_idx), 3), dtype=np.float32)
    ray_directions = np.zeros((len(trial_edge_idx), 3), dtype=np.float32)
    ray_lens = np.zeros((len(trial_edge_idx),), dtype=np.float32)
    for i, e_idx in enumerate(trial_edge_idx):
        edge = me_body.edges[e_idx]
        v0, v1 = edge.verts

        ray_origins[i] = np.float32(v0.co)
        dir = v1.co - v0.co
        ray_lens[i] = dir.length
        ray_directions[i] = np.float32(dir.normalized())

        # success, co, no, index = tongue_ray_cast(v0.co, (v1.co - v0.co).normalized(), distance = ed.calc_length())
        # success, co, no, index = tongue_ray_cast(v0.co, (v1.co - v0.co), distance = 1.0)
    if False:
        CreateAndEmptyCollection(_col_name="mouth_samples_debug")
        randIdx = np.random.randint(0, len(ray_origins), size=(200,))
        for i, idx in enumerate(randIdx):
            bpy.ops.mesh.primitive_cube_add(location=Vector(ray_origins[idx]), scale=Vector((0.001, 0.001, 0.001)))
            cube = bpy.context.object
            cube.name = "TrialSample_{}".format(i)
    if False:
        CreateAndEmptyCollection(_col_name="mouth_samples_debug")
        randIdx = np.random.randint(0, len(ray_origins), size=(500,))
        for i, idx in enumerate(randIdx):
            v1 = ray_origins[idx]
            dir = ray_directions[idx]
            rLen = ray_lens[idx]
            # Calculate the cylinder's location, size, and rotation
            location = Vector(v1 + dir * (rLen * 0.5))
            rotation_quat = Vector(dir).to_track_quat("Z", "Y")
            # Create a cylinder
            bpy.ops.mesh.primitive_cylinder_add(vertices=4, radius=0.0002, depth=rLen, location=location)
            # Rotate the cylinder
            bpy.context.object.rotation_euler = rotation_quat.to_euler()
            bpy.context.object.name = "TrialSample_{}".format(i)

    tongue_faces = np.fromiter(
        (v.index for f in me_tongue.faces for v in f.verts), dtype=np.int32, count=len(me_tongue.faces) * 3
    )
    tongue_faces.shape = (len(me_tongue.faces), 3)
    tongue_verts = np.fromiter(
        (x for v in me_tongue.verts for x in v.co), dtype=np.float32, count=len(me_tongue.verts) * 3
    )
    tongue_verts.shape = (len(me_tongue.verts), 3)

    # by default, Trimesh will do a light processing, which will
    # remove any NaN values and merge vertices that share position
    # if you want to not do this on load, you can pass `process=False`
    inter_mesh = trimesh.Trimesh(vertices=tongue_verts, faces=tongue_faces, process=False)
    # inter_mesh.export("C:/tmp/debugTongue2.obj")
    # Returns
    # ---------
    # locations: (n) sequence of (m,3) intersection points
    # index_ray: (n,) int, list of ray index
    # index_tri: (n,) int, list of triangle (face) indexes
    locations, index_ray, _ = inter_mesh.ray.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions)

    hits_loc = np.float32(locations)
    # print(f"hit loc shape {hits_loc.shape}")
    ray_idx = np.int64(index_ray)
    # print(f"hit ray idx shape {ray_idx.shape}")
    hit_max_dist = ray_lens[ray_idx]
    # print(f"hit ray max dist shape {hit_max_dist.shape}")
    deltas = hits_loc - ray_origins[ray_idx]

    hit_dist = np.linalg.norm(deltas, axis=1)
    # print(f"hit dist max {np.max(hit_dist)}")
    # print(f"hit dist min {np.min(hit_dist)}")

    # print(len(locations))
    # iIntersections = np.count_nonzero(hit_max_dist < hit_dist)
    iIntersections = np.argwhere(hit_max_dist > hit_dist)
    # print(iIntersections.shape)
    print("Number of intersections = {}".format(len(iIntersections)))
    inter_points = hits_loc[iIntersections[:, 0]]
    if False:
        CreateAndEmptyCollection(_col_name="mouth_intersection_points")
        randIdx = np.arange(0, len(inter_points))  # np.random.randint(0,len(inter_points), size=(200,))
        for i, idx in enumerate(randIdx):
            bpy.ops.mesh.primitive_cube_add(location=Vector(inter_points[idx]), scale=Vector((0.001, 0.001, 0.001)))
            cube = bpy.context.object
            cube.name = "InterSample_{}".format(i)

    # for vert in me_body.verts:
    #    # check if the vertice is in the bounding vectors
    #    if(IsInBoundingVectors(vert.co, bbox_ext[0], bbox_ext[1])):
    ##        vert.select = True
    ##    else:
    ##        vert.select = False

    # clean up after
    me_body.free()
    me_tongue.free()
    # bpy.ops.object.select_all(action='DESELECT')
    # obj_tongue.select_set(True)
    # obj_body_cp.select_set(True)
    # bpy.ops.object.delete()
    return len(iIntersections) > 0


##############################################################################################################
def CorrectRandomTongueExpression(object, mLimits, face_expression_sigma, humgen_settings):
    """Check whether the tongue mesh intersects the head mesh and if so sample a new tongue pose.

    Args:
        object (_type_): the HumGen object
        mLimits (_type_): the bone limits (low and high)

    Returns:
        bool: whether a random pose for the tongue could be found or not
    """
    # first, switch to edit mode
    bpy.context.view_layer.objects.active = object
    # set object origin to spine to add object constraint to seat empty
    bpy.ops.object.mode_set(mode="POSE")

    shape = (len(mLimits), 3)  # Replace with your desired shape (e.g., (rows, columns))

    # Generate the uniform random array
    # r = np.random.uniform(0, 1, size=shape).as_type(np.float32)
    hasIntersection = False

    if humgen_settings.tongue_bone_name in object.pose.bones:
        tongueLimits = mLimits[humgen_settings.tongue_bone_name]
        key = humgen_settings.tongue_bone_name
        b = object.pose.bones[key]
        l0 = np.float32(tongueLimits[0])
        l1 = np.float32(tongueLimits[1])

        countTrials = 0
        while countTrials < 10:
            hasIntersection = CheckForTongueIntersection(humgen_settings.iIgnore_body_vertices)
            if not hasIntersection:
                break
            else:
                print("Tongue is intersecting -> find a new tongue pose...")
            r = face_expression_sigma * np.random.normal(0, 1, size=(3,)).astype(np.float32)
            mask0 = r < 0
            mask1 = r >= 0
            d = np.zeros((3,), dtype=np.float32)
            d[mask0] = l0[mask0] * np.abs(r[mask0])
            d[mask1] = l1[mask1] * np.abs(r[mask1])
            b.location = Vector(d)
            countTrials += 1
    return not hasIntersection


def CorrectEyeGazeDirection(object, humgen_settings):
    # TODO: check whether the eye pupils gaze direction is intersecting the face mesh (the eye lids)
    # and resample the gaze direction (eye_gaze_bone) if both eye are intersecting
    iVert0 = humgen_settings.eye_left_center_vertex_idx
    iVert1 = humgen_settings.eye_right_center_vertex_idx
    sBoneName = humgen_settings.eye_gaze_bone_name
    # [todo] ask robert whats the point of this ^^


def check_collection(collection, object):
    if object.name in collection.objects:
        return True
    else:
        return any([check_collection(child, object) for child in collection.children])


def RandomizeFace(Collection, args, sMode, **kwargs):
    """
    Randomize the faces of all humgen v4 models in the collection.
    Only for humgen v4 humans with face controls

    Modifier arguments:
        fFaceExpressionSigma: float in [0,1] for the strength of the randomization, i.e.
            how extreme the facial expressions are supposed to be
        iSeed (integer): seed for randomization
        lHeadRotLimits [float,float,float], default [0.25 pi, 0.4 pi, 0.17 pi]: list of max rotation angles for the head in radians
        sHumgenVersion (string): identifier of the humgen version targeted, currently only "gen-07" is supported.

    Parameters
    ----------
    Collection : Blender Collection containing the humans to be modified
    args : dict
        Dictionary with configuration arguments
    """

    # import debugpy

    # debugpy.listen(5678)
    # debugpy.wait_for_client()

    # Extract modifier parameters
    face_expression_sigma = convert.DictElementToFloat(args, "fFaceExpressionSigma", fDefault=0.4)
    iSeed = convert.DictElementToInt(args, "iSeed", iDefault=0)
    np.random.seed(iSeed)
    head_rot_limits = args.get("lHeadRotLimits", [math.pi * 0.25, math.pi * 0.4, math.pi * 0.17])

    # Collect all Armatures in the collection
    Armatures = [
        o
        for o in bpy.data.objects
        if o.name.startswith("Armature.") and o.type == "ARMATURE" and check_collection(Collection, o)
    ]

    # this should allow this code to be easily adapted to future humgen versions
    sHumgenVersion = convert.DictElementToString(args, "sHumgenVersion", "gen-07")
    humgen_settings = anyhuman_gen_data(sHumgenVersion)

    for obj in Armatures:
        limits = GetLocConstraints(obj, humgen_settings.ignore_bones)
        RandomFaceExpression(obj, limits, humgen_settings, humgen_settings.face_bone_correlation, face_expression_sigma)
        success = CorrectRandomTongueExpression(obj, limits, face_expression_sigma, humgen_settings)
        if not success:
            print("Tongue could not be fixed!")
        RandomFacePose(obj, head_rot_limits)

        body_obj = bpy.data.objects["HG_Body"]
        iMarkers_left = FindNearestMarkersMirrored(body_obj, humgen_settings.iMarkers_right)
        iMarkers = [*iMarkers_left, *humgen_settings.iMarkers_right]

        mMarkers = {i: idx for i, idx in enumerate(iMarkers) if idx >= 0}
        SetFacialMarkers(body_obj, mMarkers)
