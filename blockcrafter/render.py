# Copyright 2018 Moritz Hilscher
#
# This file is part of Blockcrafter.
#
# Blockcrafter is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Blockcrafter is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Blockcrafter.  If not, see <http://www.gnu.org/licenses/>.

import os
import numpy as np
import math
import time
from PIL import Image
from vispy import app, gloo, geometry
from vispy.util import transforms

from blockcrafter import mcmodel

VERTEX = """
attribute vec3 a_position;
attribute vec2 a_texcoord;
attribute vec3 a_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_texcoord;
uniform mat4 u_normal;

varying vec3 v_position;
varying vec2 v_texcoord;
varying vec3 v_normal;
varying float v_face;

void main() {
    v_position = a_position;
    v_normal = a_normal;

    v_texcoord = (u_texcoord * (vec4(a_texcoord - 0.5, 0.0, 1.0))).xy + 0.5;

    /*
    Detect the quadrant the face is in, by looking at the normal after block rotation
    0 - west
    1 - east
    2 - up
    3 - down
    4 - south
    5 - north
    */
    vec4 t_nor = u_normal * vec4(a_normal, 0.0);
    vec4 t_norabs = abs(t_nor);
    if (t_norabs.x > t_norabs.y) {
        if (t_norabs.x > t_norabs.z) {
            if (t_norabs.x > 0) {
                v_face = 1.0 / 6.0; // East
            } else {
                v_face = 0.0 / 6.0; // West
            }
        } else {
            if (t_norabs.z > 0) {
                v_face = 4.0 / 6.0; // South
            } else {
                v_face = 5.0 / 6.0; // North
            }
        }
    } else {
        if (t_norabs.y > t_norabs.z) {
            if (t_norabs.y > 0) {
                v_face = 2.0 / 6.0; // Up
            } else {
                v_face = 3.0 / 6.0; // Down
            }
        } else {
            if (t_norabs.z > 0) {
                v_face = 4.0 / 6.0; // South
            } else {
                v_face = 5.0 / 6.0; // North
            }
        }
    }

    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
}
"""

FRAGMENT_BLOCK_COLOR = """
uniform sampler2D u_texture;

varying vec2 v_texcoord;
varying vec3 v_normal;

void main() {
    vec4 t_color = texture2D(u_texture, v_texcoord);

    if (t_color.a <= 0.00001) {
        discard;
    }

    gl_FragColor = t_color;
}
"""

FRAGMENT_BLOCK_UV = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_normal;

uniform sampler2D u_texture;
varying vec3 v_position;
varying vec3 v_normal;
varying vec2 v_texcoord;
varying float v_face;

void main() {
    vec4 t_color = texture2D(u_texture, v_texcoord);
    if (t_color.a <= 0.00001) {
        discard;
    }

    // Process the value of the Z position and scale it to be
    // stored in the alpha channel of pixel, hence why
    // the blending is disabled for UV mode.
    // It's purpose is mainly to merge blocks, such as waterlog
    vec4 t_modelrot = u_model * vec4(v_position, 1.0);
    float t_z = min(max(((t_modelrot.z + 1.0) * 0.5) * 255.0/256.0 + 1.0/256.0, 1.0/256.0), 1.0);

    vec4 t_normalrot = u_normal * vec4(v_position, 1.0);
    vec4 t_cap = min(max(((t_normalrot + 1.0) * 0.5) * 255.0/256.0 + 1.0/256.0, 1.0/256.0), 1.0);

    vec2 t_uv;
    if (v_face >= 5.0/6.0) {
        t_uv = t_cap.xy;
        t_uv = vec2(0.0, 1.0) + t_uv * vec2(1.0, -1.0);
    } else
    if (v_face >= 4.0/6.0) {
        t_uv = t_cap.xy;
        t_uv = vec2(0.0, 1.0) + t_uv * vec2(1.0, -1.0);
    } else
    if (v_face >= 3.0/6.0) {
        t_uv = t_cap.xz;
    } else
    if (v_face >= 2.0/6.0) {
        t_uv = t_cap.xz;
    } else
    if (v_face >= 1.0/6.0) {
        t_uv = t_cap.zy;
        t_uv = vec2(0.0, 1.0) + t_uv * vec2(1.0, -1.0);
    } else
    {
        t_uv = t_cap.zy;
        t_uv = vec2(0.0, 1.0) + t_uv * vec2(1.0, -1.0);
    }

    gl_FragColor = vec4(t_uv.xy, v_face, t_z);
}
"""

FRAGMENT_LINE_COLOR = """
uniform vec4 u_color;

void main() {
    gl_FragColor = u_color;
}
"""

class Lines(gloo.Program):
    def __init__(self, count, vertex=VERTEX, fragment=FRAGMENT_LINE_COLOR):
        super().__init__(vertex, fragment, count=count)

        self["a_position"] = gloo.VertexBuffer(np.zeros((count, 3), dtype=np.float32))

    def render(self, points, model, view, projection, color=(1.0, 1.0, 1.0, 1.0)):
        if not isinstance(points[0], (tuple, list)):
            points = np.stack(points, axis=0)
        self["a_position"].set_data(points)

        self["u_model"] = model
        self["u_view"] = view
        self["u_projection"] = projection

        self["u_color"] = color

        self.draw(gloo.gl.GL_LINE_STRIP)

line_program = None
def draw_line(p0, p1, model, view, projection, color=(1.0, 1.0, 1.0, 1.0)):
    global line_program
    if line_program is None:
        line_program = Lines(count=2)

    line_program.render([p0, p1], model, view, projection, color=color)

# from stackoverflow: https://stackoverflow.com/a/13849249
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class Element:

    CUBE_POINTS = [
        [-1, -1, -1], # V1
        [ 1, -1, -1], # V2
        [ 1,  1, -1], # V3
        [-1,  1, -1], # V4

        [-1, -1,  1], # V5
        [ 1, -1,  1], # V6
        [ 1,  1,  1], # V7
        [-1,  1,  1], # V8
    ]

    CUBE_FACES = [
        [5, 4, 0, 1], # V6, V5, V1, V2  down
        [7, 6, 2, 3], # V3, V4, V8, V7  up x
        [1, 0, 3, 2], # V2, V1, V4, V3  north
        [4, 5, 6, 7], # V5, V6, V7, V8  south
        [0, 4, 7, 3], # V1, V5, V8, V4  west
        [5, 1, 2, 6], # V6, V2, V3, V7  east
    ]

    CUBE_NORMALS = [
        [  0, -1,  0], # down
        [  0,  1,  0], # up
        [  0,  0, -1], # north
        [  0,  0,  1], # south
        [ -1,  0,  0], # west
        [  1,  0,  0], # east
    ]

    CUBE_TEXTURE_DIRS = [
        [0, 0, -1],
        [0, 0, -1],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]

    _color_program = None 
    _uv_program = None

    @classmethod
    def get_program(cls, mode):
        if mode == "color":
            if cls._color_program is None:
                cls._color_program = gloo.Program(VERTEX, FRAGMENT_BLOCK_COLOR)
                cls._color_program["a_position"] = gloo.VertexBuffer(np.zeros((4, 3), dtype=np.float32))
                cls._color_program["a_normal"] = gloo.VertexBuffer(np.zeros((4, 3), dtype=np.float32))
            return cls._color_program
        if mode == "uv":
            if cls._uv_program is None:
                cls._uv_program = gloo.Program(VERTEX, FRAGMENT_BLOCK_UV)
                cls._uv_program["a_position"] = gloo.VertexBuffer(np.zeros((4, 3), dtype=np.float32))
                cls._uv_program["a_normal"] = gloo.VertexBuffer(np.zeros((4, 3), dtype=np.float32))
            return cls._uv_program
        assert False, "Invalid mode!"

    def __init__(self, model, element):
        super().__init__()

        self.faces = Element.load_faces(model, element)
        self.rotation = element.get("rotation", None)

        self.xyz0 = (np.array(element["from"]) - 8.0) / 16.0 * 2.0
        self.xyz1 = (np.array(element["to"]) - 8.0) / 16.0 * 2.0

        self.scale = (self.xyz1 - self.xyz0) * 0.5
        self.translate = (self.xyz1 + self.xyz0) * 0.5
        self.points = np.array(Element.CUBE_POINTS) * self.scale + self.translate
        self.indices = gloo.IndexBuffer(np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32))

    def render_face(self, face_index, texture, uvs, model, view, projection, element_rotation, element_transform, uvlock):
        program = self.current_program

        # ---
        # --- set up attributes ---#
        # ---

        points = self.points[Element.CUBE_FACES[face_index]].astype(np.float32)
        normal = np.array(Element.CUBE_NORMALS[face_index], dtype=np.float32)

        program["a_position"].set_data(points)
        program["a_normal"].set_data(np.stack([normal] * 4))

        program["a_texcoord"] = np.array(uvs, dtype=np.float32)

        # ---
        # --- set up uniforms ###
        # ---

        # if uvlock is wanted, apply correction now to texture
        if uvlock:

            # we need to do some transformation magic to handle uvlock correctly

            # I am going to speak about world coordinates now
            # but actually I mean the model coordinates after element is rotated by element/block

            # get the face normal in world coordinates to determine where texture should point to
            # (that face normal in world coordinates describes as which face actually this face appears to viewer)
            cube_normal = np.round(np.dot(np.append(normal, [0]), element_transform)[:3])

            # actual texture dir in model coordinates
            texture_dir = np.array(Element.CUBE_TEXTURE_DIRS[face_index], dtype=np.float32)

            target_texture_dir = None
            if abs(cube_normal[1]) > 0.001:
                # top face, should point to east
                target_texture_dir = np.array(Element.CUBE_TEXTURE_DIRS[0], dtype=np.float32)
            else:
                # side face, should point to top
                target_texture_dir = np.array(Element.CUBE_TEXTURE_DIRS[2], dtype=np.float32)
            # go from world -> model
            element_transform_inv = np.array(np.matrix(element_transform).I)
            target_texture_dir = np.round(np.dot(np.append(target_texture_dir, [0]), element_transform_inv)[:3]).astype(np.float32)

            # now we can get the angle we have to rotate the texture
            angle = np.round(math.degrees(angle_between(texture_dir, target_texture_dir)))
            # also get a direction related to face normal
            direction = np.dot(np.cross(normal, texture_dir), target_texture_dir)
            if direction < 0:
                angle = 360 - angle

            program["u_texcoord"] = transforms.rotate(angle, (0, 0, 1))

        else:
            program["u_texcoord"] = np.eye(4, dtype=np.float32)

        program["u_texture"] = texture

        # ---
        # --- actual drawing ---
        # ---
        program.draw("triangles", self.indices)

        # old code to debug normals / texture direction stuff
        #center = np.sum(points, axis=0) / len(points) + 0.001 * normal
        #draw_line(center, center + normal * 0.5, model, view, projection, (1.0, 1.0, 0.0, 1.0))
        #draw_line(center, center + texture_dir * 0.5, model, view, projection, (0.0, 1.0, 1.0, 1.0))
        #center += 0.005 * normal
        #draw_line(center, center + cube_target_texture_dir * 0.5, model, view, projection, (0.0, 1.0, 0.0, 1.0))
        #if direction < 0:
        #    draw_line(center, center + normal * 0.25, model, view, projection, (1.0, 0.0, 0.0, 1.0))

    def render(self, model, view, projection, mode="color", block_rotation=0, element_transform=np.eye(4, dtype=np.float32), uvlock=False):
        element_rotation = np.eye(4, dtype=np.float32)
        if self.rotation:
            rotationdef = self.rotation
            axis = {"x" : [1, 0, 0],
                    "y" : [0, 1, 0],
                    "z" : [0, 0, 1]}[rotationdef["axis"]]
            origin = (np.array(rotationdef.get("origin", [8, 8, 8]), dtype=np.float32) - 8.0) / 16.0 * 2.0
            element_rotation = np.dot(transforms.translate(-origin), np.dot(transforms.rotate(rotationdef["angle"], axis), transforms.translate(origin)))

        program = Element.get_program(mode)
        self.current_program = program

        # add rotation of block and element to element transformation
        block_rotation = transforms.rotate(-90 * block_rotation, (0, 1, 0))
        element_transform = np.dot(element_rotation, np.dot(element_transform, block_rotation))
        complete_model = np.dot(element_transform, model)
        program["u_model"] = complete_model
        program["u_view"] = view
        program["u_projection"] = projection
        program["u_normal"] = element_transform

        for i, (texture, uvs) in enumerate(self.faces):
            if texture is None:
                continue
            self.render_face(i, texture, uvs, complete_model, view, projection, element_rotation=element_rotation, element_transform=element_transform, uvlock=uvlock and mode != "uv")

    @staticmethod
    def getShiftedIndex(idx, rot):
        return (idx + rot / 90) % 4
    @staticmethod
    def getReverseIndex(idx, rot):
        return (idx + 4 - rot / 90) % 4
    @staticmethod
    def getU(uvs, idx, rot):
        idx = Element.getShiftedIndex(idx, rot)
        return uvs[(idx != 0 and idx != 1) * 2]
    @staticmethod
    def getV(uvs, idx, rot):
        idx = Element.getShiftedIndex(idx, rot)
        return uvs[(idx != 0 and idx != 3) * 2 + 1]

    @staticmethod
    def load_faces(model, element):
        # order of minecraft directions to order of cube sides
        mc_to_opengl = [
            "down",   # neg y # 0
            "up",     # pos y # 1
            "north",  # neg z # 2
            "south",  # pos z # 3
            "west",   # neg x # 4
            "east",   # pos x # 5
        ]

        faces = {}
        for direction, facedef in element["faces"].items():
            texture_name = model.resolve_texture(facedef["texture"])
            if texture_name is None:
                continue # continue and ignore the face
            f = model.load_texture(texture_name)

            image = Image.open(f).convert("RGBA")
            if texture_name.startswith("block/") and image.size[0] != image.size[1]:
                assert image.size[0] < image.size[1]
                s = image.size[0]
                image = image.crop((0, 0, s, s))

            uvs = np.array(facedef.get("uv", []), dtype=np.float32)
            # If no UVs are provided, auto-define UVs by the position and direction of the face
            if len(uvs) != 4:
                xyz0 = element["from"]
                xyz1 = element["to"]
                if direction == "down":
                    uvs = np.array([xyz0[0], 16 - xyz1[2], xyz1[0], 16 - xyz0[2]], dtype=np.float32)
                elif direction == "up":
                    uvs = np.array([xyz0[0], xyz0[2], xyz1[0], xyz1[2]], dtype=np.float32)
                elif direction == "north":
                    uvs = np.array([16 - xyz1[0], 16 - xyz1[1], 16 - xyz0[0], 16 - xyz0[1]], dtype=np.float32)
                elif direction == "south":
                    uvs = np.array([xyz0[0], 16 - xyz1[1], xyz1[0], 16 - xyz0[1]], dtype=np.float32)
                elif direction == "west":
                    uvs = np.array([xyz0[2], 16 - xyz1[1], xyz1[2], 16 - xyz0[1]], dtype=np.float32)
                elif direction == "east":
                    uvs = np.array([16 - xyz1[2], 16 - xyz1[1], 16 - xyz0[2], 16 - xyz0[1]], dtype=np.float32)
                else:
                    # Like north if we don't know the direction, which should not happen
                    uvs = np.array([16 - xyz1[0], 16 - xyz1[1], 16 - xyz0[0], 16 - xyz0[1]], dtype=np.float32)

            rotation = facedef.get("rotation", 0)
            uvs = uvs / 16.0
            uv0 = [Element.getU(uvs, 1, rotation), Element.getV(uvs, 1, rotation)]
            uv1 = [Element.getU(uvs, 2, rotation), Element.getV(uvs, 2, rotation)]
            uv2 = [Element.getU(uvs, 3, rotation), Element.getV(uvs, 3, rotation)]
            uv3 = [Element.getU(uvs, 0, rotation), Element.getV(uvs, 0, rotation)]

            data = np.array(image)
            semi_transparent = np.all((data[:, :, 3] == 0) | (data[:, :, 3] == 255))
            w, h = image.size
            image = image.resize((w*3, h*3), resample=Image.NEAREST)
            data = np.array(image)
            if semi_transparent:
                data[:, :, 3] = (data[:, :, 3] > 255/2.0) * 255

            if "blockcrafterTint" in facedef:
                r, g, b = facedef["blockcrafterTint"]
                data[:, :, 0] = data[:, :, 0] * r
                data[:, :, 1] = data[:, :, 1] * g
                data[:, :, 2] = data[:, :, 2] * b
            faces[direction] = (gloo.Texture2D(data=data, interpolation="linear"), (uv0, uv1, uv2, uv3))
            f.close()

        # gather faces in order for cube sides
        # remember: each side is (texture, (uv0, uv1, uv2, uv3))
        sides = [ faces.get(direction, None) for direction in mc_to_opengl ]
        # so this is how an non-existant side looks like
        empty = (None, None)
        sides = [ (s if s is not None else empty) for s in sides ]
        assert len(sides) == 6
        return sides

class Model:
    def __init__(self, modeldef):
        self.elements = []
        for elementdef in modeldef.elements:
            self.elements.append(Element(modeldef, elementdef))

    def render(self, model, view, projection, block_rotation=0, mode="color", modelref={}):
        m = np.eye(4, dtype=np.float32)
        if "x" in modelref:
            m = np.dot(m, transforms.rotate(-modelref["x"], (1, 0, 0)))
        if "y" in modelref:
            m = np.dot(m, transforms.rotate(-modelref["y"], (0, 1, 0)))
        if "z" in modelref:
            m = np.dot(m, transforms.rotate(-modelref["z"], (0, 0, 1)))

        uvlock = modelref.get("uvlock", False)
        for element in self.elements:
            element.render(model, view, projection, mode=mode, block_rotation=block_rotation, element_transform=m, uvlock=uvlock)

class Block:
    def __init__(self, blockstate):
        self.blockstate = blockstate
        self.models = {}
        self.conditions = {}

    def _load_condition(self, condition, alt):
        modelrefs = []
        for model, transformation in self.blockstate.evaluate_condition(condition, alt):
            if not model.name in self.models:
                self.models[model.name] = Model(model)
            modelrefs.append((self.models[model.name], transformation))
        return modelrefs

    def render(self, condition, alt, model, view, projection, rotation=0, mode="color"):
        condition_str = mcmodel.encode_condition(condition) + "#" + str(alt)
        if condition_str not in self.conditions:
            self.conditions[condition_str] = self._load_condition(condition, alt)

        modelrefs = self.conditions[condition_str]
        for glmodel, transformation  in modelrefs:
            glmodel.render(model, view, projection, block_rotation=rotation, mode=mode, modelref=transformation)
        return len(modelrefs) != 0

def create_transform_ortho(aspect=1.0, view="isometric", fake_ortho=True):
    model = np.eye(4, dtype=np.float32)

    if view == "isometric":
        if fake_ortho:
            # 0.816479 = 0.5 * sqrt(3) * x = 0.5 * sqrt(2)
            # scale of y-axis to make sides and top of same height: (1.0, 0.81649, 1.0)
            # scale to get block completely into viewport (-1;1): (1.0 / math.sqrt(2), ..., ...)
            model = np.dot(model, transforms.scale((1.0 / math.sqrt(2), 0.816479 / math.sqrt(2), 1.0 / math.sqrt(2))))
        else:
            # this scale factor is just for nicely viewing the block image manually
            model = np.dot(model, transforms.scale((0.5, 0.5, 0.5)))

        # and do that nice tilt
        model = np.dot(model, np.dot(transforms.rotate(45, (0, 1, 0)), transforms.rotate(30, (1, 0, 0))))
    elif view == "topdown":
        model = np.dot(model, transforms.rotate(90, (1, 0, 0)))
    elif view == "side":
        # same thing with scaling factor as with isometric view
        #f = 1.0 / math.sqrt(2)
        f = 0.5 / math.cos(math.radians(45))
        model = np.dot(model, transforms.scale((f / math.cos(math.radians(45)), f, f)))
        model = np.dot(model, transforms.rotate(45, (1, 0, 0)))
    elif view == "default":
        pass
    else:
        assert False, "Invalid view '%s'!" % view

    view = transforms.translate((0, 0, -5))
    projection = transforms.ortho(-aspect, aspect, -1, 1, 2.0, 50.0)
    return model, view, projection

def create_transform_perspective(aspect=1.0):
    model = np.dot(transforms.rotate(45, (0, 1, 0)), transforms.rotate(25, (1, 0, 0)))
    view = transforms.translate((0, 0, -5))
    projection = transforms.perspective(45.0, aspect, 2.0, 50.0)
    return model, view, projection

def apply_model_rotation(model, rotation=0, phi=0.0):
    rotation = transforms.rotate(-rotation * 90 + phi, (0, 1, 0))
    return np.dot(rotation, model)

def apply_face_culling(on=True):
    if on:
        gloo.set_state(cull_face=True)
    else:
        gloo.set_state(cull_face=False)

def set_blending(mode):
    if mode == "premultiplied":
        gloo.set_state(blend=True, depth_test=True)
        gloo.gl.glBlendEquationSeparate(gloo.gl.GL_FUNC_ADD, gloo.gl.GL_FUNC_ADD)
        gloo.gl.glBlendFuncSeparate(gloo.gl.GL_ONE, gloo.gl.GL_ONE_MINUS_SRC_ALPHA, gloo.gl.GL_ONE, gloo.gl.GL_ONE_MINUS_SRC_ALPHA)
    else:
        gloo.set_state(mode, clear_color=(0.0, 0.0, 0.0, 0.0))
