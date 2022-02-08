'''
MIT License

Copyright (c) 2019 Shunsuke Saito, Zeng Huang, and Ryota Natsume

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import numpy as np
import random

from .framework import *
from .cam_render import CamRender


class ColorRender(CamRender):
    def __init__(self, width=1600, height=1200, name='Color Renderer'):
        program_files = ['color.vs', 'color.fs']
        CamRender.__init__(self,
                           width,
                           height,
                           name,
                           program_files=program_files,
                           color_size=3)

        # WARNING: this differs from vertex_buffer and vertex_data in Render
        self.vert_buffer = {}
        self.vert_data = {}

        # normal
        self.norm_buffer = {}
        self.norm_data = {}

        self.color_buffer = {}
        self.color_data = {}

        self.vertex_dim = {}
        self.n_vertices = {}

        self.rot_mat_unif = glGetUniformLocation(self.program, 'RotMat')
        self.rot_matrix = np.eye(3)

        self.norm_mat_unif = glGetUniformLocation(self.program, 'NormMat')
        self.normalize_matrix = np.eye(4)

    def set_norm_mat(self, scale, center):
        N = np.eye(4)
        N[:3, :3] = scale * np.eye(3)
        N[:3, 3] = -scale * center

        self.normalize_matrix = N

    def set_mesh(self, vertices, faces, color, normals, mat_name='all'):

        self.vert_data[mat_name] = vertices[faces.reshape([-1])]
        self.n_vertices[mat_name] = self.vert_data[mat_name].shape[0]
        self.vertex_dim[mat_name] = self.vert_data[mat_name].shape[1]
        self.color_data[mat_name] = color[faces.reshape([-1])]
        self.norm_data[mat_name] = normals[faces.reshape([-1])]

        if mat_name not in self.vert_buffer.keys():
            self.vert_buffer[mat_name] = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vert_buffer[mat_name])
        glBufferData(GL_ARRAY_BUFFER, self.vert_data[mat_name], GL_STATIC_DRAW)

        if mat_name not in self.color_buffer.keys():
            self.color_buffer[mat_name] = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer[mat_name])
        glBufferData(GL_ARRAY_BUFFER, self.color_data[mat_name],
                     GL_STATIC_DRAW)

        if mat_name not in self.norm_buffer.keys():
            self.norm_buffer[mat_name] = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.norm_buffer[mat_name])
        glBufferData(GL_ARRAY_BUFFER, self.norm_data[mat_name], GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def cleanup(self):

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        for key in self.vert_data:
            glDeleteBuffers(1, [self.vert_buffer[key]])
            glDeleteBuffers(1, [self.color_buffer[key]])
            glDeleteBuffers(1, [self.norm_buffer[key]])

        self.norm_buffer = {}
        self.norm_data = {}

        self.vert_buffer = {}
        self.vert_data = {}

        self.color_buffer = {}
        self.color_data = {}

        self.render_texture_mat = {}

        self.vertex_dim = {}
        self.n_vertices = {}

    def draw(self):
        self.draw_init()

        glEnable(GL_MULTISAMPLE)

        glUseProgram(self.program)
        glUniformMatrix4fv(self.norm_mat_unif, 1, GL_FALSE,
                           self.normalize_matrix.transpose())
        glUniformMatrix4fv(self.model_mat_unif, 1, GL_FALSE,
                           self.model_view_matrix.transpose())
        glUniformMatrix4fv(self.persp_mat_unif, 1, GL_FALSE,
                           self.projection_matrix.transpose())
        glUniformMatrix3fv(self.rot_mat_unif, 1, GL_FALSE,
                           self.rot_matrix.transpose())

        for mat in self.vert_buffer:

            # Handle vertex buffer
            glBindBuffer(GL_ARRAY_BUFFER, self.vert_buffer[mat])
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, self.vertex_dim[mat], GL_DOUBLE, GL_FALSE,
                                  0, None)

            # Handle color buffer
            glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer[mat])
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_DOUBLE, GL_FALSE, 0, None)

            # Handle normal buffer
            glBindBuffer(GL_ARRAY_BUFFER, self.norm_buffer[mat])
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 3, GL_DOUBLE, GL_FALSE, 0, None)

            glDrawArrays(GL_TRIANGLES, 0, self.n_vertices[mat])

            glDisableVertexAttribArray(2)
            glDisableVertexAttribArray(1)
            glDisableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glUseProgram(0)

        glDisable(GL_MULTISAMPLE)

        self.draw_end()
