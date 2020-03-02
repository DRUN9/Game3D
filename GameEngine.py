import pygame
from math import sin, cos, tan, sqrt, pi
import numpy
import os
import sys
import random


class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.vec = numpy.array([self.x, self.y, self.z, self.w])

    def __str__(self):
        return str(self.x) + ' ' + str(self.y) + ' ' + str(self.z)


class Triangle:
    def __init__(self, vectors=None, color=255, color_state="red"):
        if vectors is not None:
            self.vectors = numpy.array(vectors)
        self.color = color
        self.color_state = color_state

    def __str__(self):
        return str(self.vectors)


class Figure:
    def __init__(self, triangles=None, filename=None):
        if triangles is not None:
            self.triangles = numpy.array(triangles)
        elif filename is not None:
            with open(filename, mode="r") as file:
                file = file.readlines()
                vectors = []
                self.triangles = []
                for line in file:
                    if line.startswith('v'):
                        line = line.split()
                        vector = Vector3D(float(line[1]),
                                          float(line[2]),
                                          float(line[3]))
                        vectors.append(vector)
                    elif line.startswith('f'):
                        line = line.split()
                        triangle = Triangle([vectors[int(line[1]) - 1],
                                             vectors[int(line[2]) - 1],
                                             vectors[int(line[3]) - 1]])
                        self.triangles.append(triangle)
                self.triangles = numpy.array(self.triangles)


class FigureGroup:
    def __init__(self, figures=None):
        if figures is None:
            self.figures = []
        else:
            self.figures = figures

    def add_figure(self, figure):
        self.figures.append(figure)


class Cube(Figure):
    def __init__(self, color, x_size, y_size):
        self.color = color
        self.color_choice = 0
        self.rastered_triangles = []
        self.make_triangles(x_size, y_size)

    def make_triangles(self, x_size, y_size):
        size_x = random.random() * 10
        size_y = random.random() * 10
        if size_x not in x_size:
            x_size.append(size_x)
        else:
            while size_x in x_size:
                size_x = random.random() * 10
        if size_y not in y_size:
            y_size.append(size_y)
        else:
            while size_y in y_size:
                size_y = random.random() * 10
        # SOUTH
        south_triangle1 = Triangle([Vector3D(size_x, size_y, 0.0), Vector3D(size_x, size_y + 1.0, 0.0),
                                    Vector3D(size_x + 1.0, size_y + 1.0, 0.0)])
        south_triangle2 = Triangle([Vector3D(size_x, size_y, 0.0), Vector3D(size_x + 1.0, size_y + 1.0, 0.0),
                                    Vector3D(size_x + 1.0, size_y, 0.0)])

        # EAST
        east_triangle1 = Triangle([Vector3D(size_x + 1.0, size_y, 0.0), Vector3D(size_x + 1.0, size_y + 1.0, 0.0),
                                   Vector3D(size_x + 1.0, size_y + 1.0, 1.0)])
        east_triangle2 = Triangle([Vector3D(size_x + 1.0, size_y, 0.0), Vector3D(size_x + 1.0, size_y + 1.0, 1.0),
                                   Vector3D(size_x + 1.0, size_y, 1.0)])

        # NORTH
        north_triangle1 = Triangle([Vector3D(size_x + 1.0, size_y, 1.0), Vector3D(size_x + 1.0, size_y + 1.0, 1.0),
                                    Vector3D(size_x + 0.0, size_y + 1.0, 1.0)])
        north_triangle2 = Triangle([Vector3D(size_x + 1.0, size_y, 1.0), Vector3D(size_x + 0.0, size_y + 1.0, 1.0),
                                    Vector3D(size_x + 0.0, size_y, 1.0)])

        # # WEST
        west_triangle1 = Triangle([Vector3D(size_x + 0.0, size_y, 1.0), Vector3D(size_x + 0.0, size_y + 1.0, 1.0),
                                   Vector3D(size_x + 0.0, size_y + 1.0, 0.0)])
        west_triangle2 = Triangle([Vector3D(size_x + 0.0, size_y, 1.0), Vector3D(size_x + 0.0, size_y + 1.0, 0.0),
                                   Vector3D(size_x + 0.0, size_y, 0.0)])

        # TOP
        top_triangle1 = Triangle([Vector3D(size_x + 0.0, size_y + 1.0, 0.0), Vector3D(size_x + 0.0, size_y + 1.0, 1.0),
                                  Vector3D(size_x + 1.0, size_y + 1.0, 1.0)])
        top_triangle2 = Triangle([Vector3D(size_x + 0.0, size_y + 1.0, 0.0), Vector3D(size_x + 1.0, size_y + 1.0, 1.0),
                                  Vector3D(size_x + 1.0, size_y + 1.0, 0.0)])

        # BOTTOM
        bot_triangle1 = Triangle([Vector3D(size_x + 1.0, size_y, 1.0), Vector3D(size_x + 0.0, size_y, 1.0),
                                  Vector3D(size_x + 0.0, size_y, 0.0)])
        bot_triangle2 = Triangle([Vector3D(size_x + 1.0, size_y, 1.0), Vector3D(size_x + 0.0, size_y, 0.0),
                                  Vector3D(size_x + 1.0, size_y, 0.0)])
        self.triangles = numpy.array([south_triangle1, south_triangle2,
                                      east_triangle1, east_triangle2,
                                      north_triangle1, north_triangle2,
                                      west_triangle1, west_triangle2,
                                      top_triangle1, top_triangle2,
                                      bot_triangle1, bot_triangle2])

    def color_change(self):
        if self.color_choice % 2 == 0:
            for triangle in self.triangles:
                triangle.color_state = "green"
            self.color_choice += 1
        else:
            for triangle in self.triangles:
                triangle.color_state = "red"
            self.color_choice += 1

    def get_color(self):
        return self.color


class SpaceInterface:
    def __init__(self):
        self.score = 0

    def draw_ship_interface(self, screen_width, screen_height):
        pygame.draw.polygon(screen, (255, 255, 255), [(0, 0), (10, 0), (0, 10)])
        pygame.draw.polygon(screen, (255, 255, 255), [(10, 0), (screen_width / 2 - 100, screen_height / 2 + 100),
                                                      (screen_width / 2 - 110, screen_height / 2 + 110), (0, 10)])
        pygame.draw.polygon(screen, (200, 200, 200), [(0, screen_height - 20),
                                                      (0, screen_height),
                                                      (screen_width / 2 - 100, screen_height / 2 + 120),
                                                      (screen_width / 2 - 100, screen_height / 2 + 100)])
        pygame.draw.polygon(screen, (228, 228, 228), [(screen_width / 2 - 100, screen_height / 2 + 100),
                                                      (screen_width / 2 - 100, screen_height / 2 + 120),
                                                      (screen_width / 2 + 100, screen_height / 2 + 100),
                                                      (screen_width / 2 + 100, screen_height / 2 + 120)])
        pygame.draw.polygon(screen, (255, 255, 255), [(screen_width / 2 + 100, screen_height / 2 + 100),
                                                      (screen_width / 2 + 110, screen_height / 2 + 110),
                                                      (screen_width, 10),
                                                      (screen_width - 10, 0)])
        pygame.draw.polygon(screen, (200, 200, 200), [(screen_width / 2 + 100, screen_height / 2 + 100),
                                                      (screen_width / 2 + 100, screen_height / 2 + 120),
                                                      (screen_width, screen_height),
                                                      (screen_width, screen_height - 20)])
        pygame.draw.polygon(screen, (255, 255, 255), [(screen_width - 10, 0),
                                                      (screen_width, 0),
                                                      (screen_width, 10)])
        pygame.draw.line(screen, (0, 0, 255),
                         (screen_width / 2 - 10, screen_height / 2),
                         (screen_width / 2 + 10, screen_height / 2))
        pygame.draw.line(screen, (0, 0, 255),
                         (screen_width / 2, screen_height / 2 - 10),
                         (screen_width / 2, screen_height / 2 + 10))

        font = pygame.font.Font(None, 100).render(str(self.score), 1, pygame.Color("green"))
        screen.blit(font, pygame.Rect(5 * screen_width / 6, 0, screen_width * 6, screen_width / 6))

    def shoot(self, x, y, cube_group, x_size, y_size):
        for cube in cube_group.figures:
            for triangle in cube.rastered_triangles:
                x1, x2, x3 = triangle.vectors[0].x, triangle.vectors[1].x, triangle.vectors[2].x
                y1, y2, y3 = triangle.vectors[0].y, triangle.vectors[1].y, triangle.vectors[2].x
                k12, k23, k31 = (y2 - y1) / (x2 - x1), \
                                (y3 - y2) / (x3 - x2), \
                                (y3 - y1) / (x3 - x1)
                b12, b23, b31 = y1 - k12 * x1, y2 - k23 * x2, y3 - k31 * x3
                if (k12 * x + b12 - y) * (k12 * x3 + b12 - y3) > 0:
                    if (k23 * x + b23 - y) * (k23 * x1 + b23 - y1) > 0:
                        if (k31 * x + b31 - y) * (k31 * x2 + b31 - y2) > 0:
                            self.score += 100
                            cube.make_triangles(x_size, y_size)
                            cube.color_change()


class GameEngine:
    def __init__(self, screen_width, screen_height):
        self.mat_proj = self.make_projection_matrix(90.0, screen_height / screen_width, 0.1, 1000.0)
        self.mat_trans = self.make_translation_matrix(0.0, 0.0, 24.0)
        self.mat_trans = self.make_translation_matrix(0.0, 0.0, 16.0)
        self.camera = Vector3D(0.0, 0.0, 0.0)
        self.up = Vector3D(0.0, 1.0, 0.0)
        self.look_direction = Vector3D(0.0, 0.0, 1.0)
        self.target = self.add_vector(self.camera, self.look_direction)

    def mul_vector_matrix(self, vector, matrix):
        vector = numpy.dot(vector.vec, matrix)
        output_vector = Vector3D(vector[0], vector[1], vector[2], vector[3])
        return output_vector

    def mul_matrix_matrix(self, matrix1, matrix2):
        return numpy.dot(matrix1, matrix2)

    def add_vector(self, vector1, vector2):
        return Vector3D(vector1.x + vector2.x, vector1.y + vector2.y, vector1.z + vector2.z)

    def sub_vector(self, vector1, vector2):
        return Vector3D(vector1.x - vector2.x, vector1.y - vector2.y, vector1.z - vector2.z)

    def mul_vector(self, vector, k):
        return Vector3D(vector.x * k, vector.y * k, vector.z * k)

    def div_vector(self, vector, k):
        return Vector3D(vector.x / k, vector.y / k, vector.z / k)

    def dot_product(self, vector1, vector2):
        return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z

    def length_vector(self, vector):
        return sqrt(self.dot_product(vector, vector))

    def normalise_vector(self, vector):
        length = self.length_vector(vector)
        if length < 1e-10:
            length = 1e-10
        return Vector3D(vector.x / length, vector.y / length, vector.z / length)

    def cross_product(self, vector1, vector2):
        x = vector1.y * vector2.z - vector1.z * vector2.y
        y = vector1.z * vector2.x - vector1.x * vector2.z
        z = vector1.x * vector2.y - vector1.y * vector2.x
        return Vector3D(x, y, z)

    def make_rotation_matrix_z(self, angle):
        matrix = numpy.array([cos(angle), sin(angle), 0.0, 0.0,
                              -sin(angle), cos(angle), 0.0, 0.0,
                              0.0, 0.0, 1.0, 0.0,
                              0.0, 0.0, 0.0, 1.0]).reshape(4, 4)
        return matrix

    def make_rotation_matrix_x(self, angle):
        matrix = numpy.array([1.0, 0.0, 0.0, 0.0,
                              0.0, cos(angle / 2), sin(angle / 2), 0.0,
                              0.0, -sin(angle / 2), cos(angle / 2), 0.0,
                              0.0, 0.0, 0.0, 1.0]).reshape(4, 4)
        return matrix

    def make_rotation_matrix_y(self, angle):
        matrix = numpy.array([cos(angle), 0.0, sin(angle), 0.0,
                              0.0, 1.0, 0.0, 0.0,
                              -sin(angle), 0.0, cos(angle), 0.0,
                              0.0, 0.0, 0.0, 1.0]).reshape(4, 4)
        return matrix

    def make_translation_matrix(self, x, y, z):
        matrix = numpy.array([1.0, 0.0, 0.0, 0.0,
                              0.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 1.0, 0.0,
                              x, y, z, 1.0]).reshape(4, 4)
        return matrix

    def make_projection_matrix(self, degrees, aspect_ratio, near, far):
        fov_rad = 1 / tan(degrees * 0.5 / 180 * pi)
        matrix = numpy.array([aspect_ratio * fov_rad, 0.0, 0.0, 0.0,
                              0.0, fov_rad, 0.0, 0.0,
                              0.0, 0.0, (far / (far - near)), 1.0,
                              0.0, 0.0, (-far * near / (far - near)), 0.0]).reshape(4, 4)
        return matrix

    def make_identity_matrix(self):
        matrix = numpy.array([1.0, 0.0, 0.0, 0.0,
                              0.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 1.0, 0.0,
                              0.0, 0.0, 0.0, 1.0]).reshape(4, 4)
        return matrix

    def make_point_at_matrix(self, vector_pos, vector_target, vector_up):
        new_forward = self.sub_vector(vector_target, vector_pos)
        new_forward = self.normalise_vector(new_forward)

        a = self.mul_vector(new_forward, self.dot_product(vector_up, new_forward))
        new_up = self.sub_vector(vector_up, a)
        new_up = self.normalise_vector(new_up)

        new_right = self.cross_product(new_up, new_forward)

        matrix = numpy.array([new_right.x, new_right.y, new_right.z, 0.0,
                              new_up.x, new_up.y, new_up.z, 0.0,
                              new_forward.x, new_forward.y, new_forward.z, 0.0,
                              vector_pos.x, vector_pos.y, vector_pos.z, 1.0]).reshape(4, 4)
        return matrix

    def make_inverse_matrix(self, matrix):
        return numpy.linalg.inv(matrix)

    def draw_triangle(self, x1, y1, x2, y2, x3, y3, color, color_state):
        if color_state == "red":
            pygame.draw.polygon(screen,
                                pygame.Color(color, 0, 0),
                                [(x1, y1), (x2, y2), (x3, y3)])
        elif color_state == "green":
            pygame.draw.polygon(screen,
                                pygame.Color(0, color, 0),
                                [(x1, y1), (x2, y2), (x3, y3)])

        pygame.draw.line(screen, pygame.Color("black"), (x1, y1), (x2, y2))
        pygame.draw.line(screen, pygame.Color("black"), (x2, y2), (x3, y3))
        pygame.draw.line(screen, pygame.Color("black"), (x3, y3), (x1, y1))

    def render(self, figure_group, mat_world, mat_view, camera):
        for figure in figure_group.figures:
            triangles_to_raster = []
            for tri in figure.triangles:
                vec0 = self.mul_vector_matrix(tri.vectors[0], mat_world)
                vec1 = self.mul_vector_matrix(tri.vectors[1], mat_world)
                vec2 = self.mul_vector_matrix(tri.vectors[2], mat_world)
                tri_transformed = Triangle([vec0, vec1, vec2])

                line1 = self.sub_vector(tri_transformed.vectors[1], tri_transformed.vectors[0])
                line2 = self.sub_vector(tri_transformed.vectors[2], tri_transformed.vectors[0])

                normal = self.cross_product(line1, line2)
                normal = self.normalise_vector(normal)

                camera_ray = self.sub_vector(tri_transformed.vectors[0], camera)
                dot_p = self.dot_product(normal, camera_ray)
                if dot_p < 0:
                    light_direction = Vector3D(0.0, 0.0, -1.0)
                    light_direction = self.normalise_vector(light_direction)

                    dp = max(0.1, self.dot_product(light_direction, normal))
                    color = int(255 * abs(dp))

                    tri_viewed = Triangle([None, None, None], color, tri.color_state)
                    tri_viewed.vectors[0] = self.mul_vector_matrix(tri_transformed.vectors[0], mat_view)
                    tri_viewed.vectors[1] = self.mul_vector_matrix(tri_transformed.vectors[1], mat_view)
                    tri_viewed.vectors[2] = self.mul_vector_matrix(tri_transformed.vectors[2], mat_view)

                    tri_projected = Triangle([None, None, None], color, tri.color_state)
                    tri_projected.vectors[0] = self.mul_vector_matrix(tri_viewed.vectors[0], self.mat_proj)
                    tri_projected.vectors[1] = self.mul_vector_matrix(tri_viewed.vectors[1], self.mat_proj)
                    tri_projected.vectors[2] = self.mul_vector_matrix(tri_viewed.vectors[2], self.mat_proj)

                    tri_projected.vectors[0] = self.div_vector(tri_projected.vectors[0], tri_projected.vectors[0].w)
                    tri_projected.vectors[1] = self.div_vector(tri_projected.vectors[1], tri_projected.vectors[1].w)
                    tri_projected.vectors[2] = self.div_vector(tri_projected.vectors[2], tri_projected.vectors[2].w)

                    # Scale into view
                    offset_view = Vector3D(1, 1, 0)
                    tri_projected.vectors[0] = self.add_vector(tri_projected.vectors[0], offset_view)
                    tri_projected.vectors[1] = self.add_vector(tri_projected.vectors[1], offset_view)
                    tri_projected.vectors[2] = self.add_vector(tri_projected.vectors[2], offset_view)

                    tri_projected.vectors[0].x *= (0.5 * width)
                    tri_projected.vectors[0].y *= (0.5 * height)
                    tri_projected.vectors[1].x *= (0.5 * width)
                    tri_projected.vectors[1].y *= (0.5 * height)
                    tri_projected.vectors[2].x *= (0.5 * width)
                    tri_projected.vectors[2].y *= (0.5 * height)

                    triangles_to_raster.append(tri_projected)
            figure.rastered_triangles = triangles_to_raster[:]
            for triangle in triangles_to_raster:
                self.draw_triangle(triangle.vectors[0].x, triangle.vectors[0].y,
                                   triangle.vectors[1].x, triangle.vectors[1].y,
                                   triangle.vectors[2].x, triangle.vectors[2].y, triangle.color, triangle.color_state)


def load_image(name, color_key=None):
    fullname = os.path.join('data', name)
    try:
        image = pygame.image.load(fullname)
    except pygame.error as message:
        print('Не удаётся загрузить:', name)
        raise SystemExit(message)
    image = image.convert_alpha()
    if color_key is not None:
        if color_key is -1:
            color_key = image.get_at((0, 0))
        image.set_colorkey(color_key)
    return image


def terminate():
    pygame.quit()
    sys.exit()


def start_screen():
    intro_text = ["SHOOTER3D", "Нажмите любую кнопку"]
    fon = pygame.transform.scale(load_image("intro.jpg"), size)
    screen.blit(fon, (0, 0))
    font = pygame.font.Font(None, 100)
    text_coord = height / 2 - 50
    for line in intro_text:
        line_rendered = font.render(line, 1, pygame.Color("white"))
        intro_rect = line_rendered.get_rect()
        text_coord += 10
        intro_rect.top = text_coord
        intro_rect.x = width / 6
        text_coord += intro_rect.height
        screen.blit(line_rendered, intro_rect)

    while True:
        for event_start in pygame.event.get():
            if event_start.type == pygame.QUIT:
                terminate()
            if event_start.type == pygame.MOUSEBUTTONDOWN:
                return
            if event_start.type == pygame.KEYDOWN:
                return
        pygame.display.flip()
        clock_fps.tick(FPS)


def menu_screen():
    menu_text = ["    ПАУЗА",
                 "УПРАВЛЕНИЕ:",
                 '\tВверх - стрелочка вверх',
                 '\tВниз - стрелочка вниз',
                 '\tВправо - стрелочка вправо',
                 '\tВлево - стрелочка влево',
                 '\tСтрелять - пробел',
                 'Для возвращения в игру нажмите escape']
    fon = pygame.transform.scale(load_image("menu.jpg"), size)
    screen.blit(fon, (0, 0))
    font = pygame.font.Font(None, 50)
    text_coord = 10
    for line in menu_text:
        line_rendered = font.render(line, 1, pygame.Color("white"))
        menu_rect = line_rendered.get_rect()
        text_coord += 10
        menu_rect.top = text_coord
        menu_rect.x = 10
        text_coord += menu_rect.height
        screen.blit(line_rendered, menu_rect)

    while True:
        for event_menu in pygame.event.get():
            if event_menu.type == pygame.QUIT:
                terminate()
            if event_menu.type == pygame.KEYDOWN:
                if event_menu.key == pygame.K_ESCAPE:
                    return
        pygame.display.flip()
        clock_fps.tick(FPS)


def space_design(stars):
    for i in stars:
        pygame.draw.line(screen, pygame.Color("white"), i, i)


pygame.init()
size = width, height = 1200, 700
screen = pygame.display.set_mode(size)

X_SIZE = []
Y_SIZE = []
STARS = []
for i in range(100):
    STARS.append((random.random() * width, random.random() * height))
cubes = FigureGroup()
for i in range(10):
    cubes.add_figure(Cube(pygame.Color("red"), X_SIZE, Y_SIZE))

engine = GameEngine(width, height)
interface = SpaceInterface()
v = 0.5
theta = 0
clock_v = pygame.time.Clock()
clock_fps = pygame.time.Clock()
FPS = 50
start_screen()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                engine.camera.y += 0.05
            if event.key == pygame.K_DOWN:
                engine.camera.y -= 0.05
            if event.key == pygame.K_LEFT:
                engine.camera.x += 0.05
            if event.key == pygame.K_RIGHT:
                engine.camera.x -= 0.05
            if event.key == pygame.K_ESCAPE:
                menu_screen()
            if event.key == pygame.K_SPACE:
                interface.shoot(width / 2, height / 2, cubes, X_SIZE, Y_SIZE)

    screen.fill(pygame.Color("black"))

    space_design(STARS)

    mat_rot_z = engine.make_rotation_matrix_z(theta)
    mat_rot_x = engine.make_rotation_matrix_x(theta)
    mat_rot_y = engine.make_rotation_matrix_y(theta)

    mat_wrld = engine.mul_matrix_matrix(mat_rot_z, mat_rot_x)
    mat_wrld = engine.mul_matrix_matrix(mat_wrld, mat_rot_z)
    mat_wrld = engine.mul_matrix_matrix(mat_wrld, engine.mat_trans)

    mat_camera = engine.make_point_at_matrix(engine.camera, engine.target, engine.up)
    mat_view = engine.make_inverse_matrix(mat_camera)

    engine.render(cubes, mat_wrld, mat_view, engine.camera)
    interface.draw_ship_interface(width, height)
    theta += v * clock_v.tick() / 1000

    clock_fps.tick(FPS)
    pygame.display.flip()
terminate()
