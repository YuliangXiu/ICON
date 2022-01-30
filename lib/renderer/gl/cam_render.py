
# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from .render import Render

GLUT = None


class CamRender(Render):
    def __init__(self,
                 width=1600,
                 height=1200,
                 name='Cam Renderer',
                 program_files=['simple.fs', 'simple.vs'],
                 color_size=1,
                 ms_rate=1,
                 egl=False):
        Render.__init__(self,
                        width,
                        height,
                        name,
                        program_files,
                        color_size,
                        ms_rate=ms_rate,
                        egl=egl)
        self.camera = None

        if not egl:
            global GLUT
            import OpenGL.GLUT as GLUT
            GLUT.glutDisplayFunc(self.display)
            GLUT.glutKeyboardFunc(self.keyboard)

    def set_camera(self, camera):
        self.camera = camera
        self.projection_matrix, self.model_view_matrix = camera.get_gl_matrix()

    def keyboard(self, key, x, y):
        # up
        eps = 1
        # print(key)
        if key == b'w':
            self.camera.center += eps * self.camera.direction
        elif key == b's':
            self.camera.center -= eps * self.camera.direction
        if key == b'a':
            self.camera.center -= eps * self.camera.right
        elif key == b'd':
            self.camera.center += eps * self.camera.right
        if key == b' ':
            self.camera.center += eps * self.camera.up
        elif key == b'x':
            self.camera.center -= eps * self.camera.up
        elif key == b'i':
            self.camera.near += 0.1 * eps
            self.camera.far += 0.1 * eps
        elif key == b'o':
            self.camera.near -= 0.1 * eps
            self.camera.far -= 0.1 * eps

        self.projection_matrix, self.model_view_matrix = self.camera.get_gl_matrix(
        )

    def show(self):
        if GLUT is not None:
            GLUT.glutMainLoop()
