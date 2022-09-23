import logging

import copy
import numpy as np
import cv2
import nudged
import random

module_logger = logging.getLogger('cnn_detector.image_transform.geometric_generator')


class RandomGeometricGenerator(object):
    def __init__(self, resolution=(160, 80), intersection_area=0.65):
        self.intersection_area = intersection_area
        self.resolution = resolution
        width = 520 / 2 / 520 * self.intersection_area
        height = 110 / 2 / 520 * self.intersection_area
        self.template_points = np.asarray(
            [[-width, -height],
             [+width, -height],
             [+width, +height],
             [-width, +height],
             ], dtype=np.float32)

        self.rot = (-0.07, 0.07)
        self.scale = (-0.1, 0.1)
        self.anisotropic = 0.15
        self.translation = (0.02, 0.02)

    def get_no_transform(self):
        T = np.asarray(
            [[1. / self.resolution[0], 0, -0.5],
             [0, 1. / self.resolution[1], -0.5],
             [0, 0, 1]], dtype=np.float32
        )
        return T

    def generate_crop(self, image, points=[]):
        if issubclass(type(points), list):
            T = self.get_no_transform()
        else:
            T = nudged.estimate(points, self.template_points)
            T = np.asarray(T.get_matrix())

        r = np.random.uniform(*self.rot)
        R = np.asarray(
            [[np.cos(r), np.sin(r), 0],
             [-np.sin(r), np.cos(r), 0],
             [0, 0, 1]])
        s = 2 ** np.random.uniform(*self.scale)
        S = np.asarray(
            [[s * 2 ** (np.random.randn() * self.anisotropic), np.random.randn() * self.anisotropic, 0],
             [0, s * (2 ** (np.random.randn() * self.anisotropic)), 0],
             [0, 0, 1]])

        D = np.asarray(
            [[1, 0, np.random.randn() * self.translation[0]],
             [0, 1, np.random.randn() * self.translation[1]],
             [0, 0, 1]])

        if issubclass(type(points), list):
            V = np.asarray(
                [[self.resolution[0], 0, self.resolution[0] / 2.0],
                 [0, self.resolution[1], self.resolution[1] / 2.0],
                 [0, 0, 1]])
        else:
            pass
            output_scale = self.resolution[0]
            V = np.asarray(
                [[output_scale, 0, self.resolution[0] / 2],
                 [0, output_scale, self.resolution[1] / 2],
                 [0, 0, 1]])

        T = V @ S @ D @ R @ T
        T = np.copy(T[:2, :])

        crop = cv2.warpAffine(image, T, (int(self.resolution[0]), int(self.resolution[1])))

        if type(points) is list:
            transformed_points = []
            for p in points:
                p = p[:, np.newaxis, :].astype(np.float32)
                p_t = cv2.transform(p, T)
                p = p_t[:, 0, :2]
                transformed_points.append(p)
            return crop, transformed_points, T
        else:
            points = points[:, np.newaxis, :]
            points = cv2.transform(points, T)
            points = points[:, 0, :]
            return crop, points, T


class RandomImageCropper(object):
    def __init__(self):
        pass

    def generate_transform(self, image, points=np.zeros((0, 4, 2), dtype=np.float32), output_resolution=None):
        while True:
            output_resolution = np.asarray(output_resolution)
            scales = np.random.uniform(0.00001, 1, size=10)
            pix_counts = scales ** 2
            pix_counts /= pix_counts.sum()
            scale = np.random.choice(scales, 1, p=pix_counts)
            input_resolution = (output_resolution / scale).astype(np.int32)
            if input_resolution[0] < image.shape[1] and input_resolution[1] < image.shape[0]:
                break

        px = np.random.randint(0, image.shape[1] - input_resolution[0])
        py = np.random.randint(0, image.shape[0] - input_resolution[1])

        input_crop = image[py:py + input_resolution[1], px:px + input_resolution[0]]
        output_crop = cv2.resize(input_crop, (output_resolution[0], output_resolution[1]), interpolation=cv2.INTER_AREA)

        points = np.copy(points)
        points[:, :, 0] = (points[:, :, 0] - px) * scale
        points[:, :, 1] = (points[:, :, 1] - py) * scale

        return output_crop, points, None


class RandomImageTransform(object):
    def __init__(self, rot=(-0.0, 0.0), scale=(-0.0, 0.0), anisotropic=0, translation=(0.0, 0.0), aligned_bbox=False,
                 area_downsampling=False):
        self.rot = rot
        self.scale = scale
        self.anisotropic = anisotropic
        self.translation = translation
        self.aligned_bbox = aligned_bbox
        self.area_downsampling = area_downsampling

    def get_scaling_factor(self, T):
        points = np.asarray([[0, 0], [1, 0], [1,1 ], [0, 1], [0, 0]])
        points = points.dot(T[:2, :2])
        distances = np.sum((points[:-1] - points[1:]) ** 2, axis=1) ** 0.5
        return np.mean(distances)

    def generate_transform(self, image, aratio=1, points=np.zeros((0, 4, 2), dtype=np.float32), scale_w=1, scale_h=1,
                           center=None, output_resolution=None):

        input_resolution = (image.shape[1], image.shape[0])
        if output_resolution is None:
            output_resolution = (input_resolution[0] * scale_w, input_resolution[1] * scale_h)

        if center is None:
            center = np.asarray([input_resolution[0] / 2, input_resolution[1] / 2, 1]).reshape(1, 3)

        # TODO - fix this for large images
        # sigmaX = max(0.3 / scale_w, 0.1)
        # sigmaY = max(0.3 / scale_h, 0.1)
        # image = cv2.GaussianBlur(image, (1 + 2 * int(sigmaX + 1.5), 1 + 2 * int(sigmaY + 1.5)), sigmaX, sigmaY)

        # Move image to center and scale to unit size in x and y
        T = np.asarray(
            [[1. / input_resolution[0], 0, -center[0] / input_resolution[0]],
             [0, 1. / input_resolution[1], -center[1] / input_resolution[1]],
             [0, 0, 1]], dtype=np.float32
        )

        # correct aspect ratio
        A = np.asarray(
            [[1, 0, 0],
             [0, 1. / aratio, 0],
             [0, 0, 1]], dtype=np.float32
        )

        # random rotation
        r = np.random.uniform(*self.rot)
        R = np.asarray(
            [[np.cos(r), np.sin(r), 0],
             [-np.sin(r), np.cos(r), 0],
             [0, 0, 1]])

        # random scale
        s = 2 ** np.random.uniform(*self.scale)
        a1 = 2 ** (np.random.randn() * self.anisotropic)
        a2 = 2 ** (np.random.randn() * self.anisotropic)
        S = np.asarray(
            [[s * a1, np.random.randn() * self.anisotropic, 0],
             [0, s * a2, 0],
             [0, 0, 1]])

        # get to target scale
        V = np.asarray(
            [[input_resolution[0] * scale_w, 0, 1],
             [0, input_resolution[1] * scale_h, 1],
             [0, 0, 1]])

        # random translation
        D = np.asarray(
            [[1, 0, np.random.randn() * self.translation[0] * output_resolution[0]],
             [0, 1, np.random.randn() * self.translation[1] * output_resolution[1]],
             [0, 0, 1]])

        # translate to target center position
        C = np.asarray(
            [[1, 0, output_resolution[0] / 2],
             [0, 1, output_resolution[1] / 2],
             [0, 0, 1]])

        T = C @ D @ V @ S @ R @ A @ T

        zoom_factor = int(1 / self.get_scaling_factor(T))
        if self.area_downsampling and zoom_factor > 1:
            S = np.asarray(
                [[zoom_factor, 0, 0],
                 [0, zoom_factor, 0],
                 [0, 0, 1]])
            S_T = S @ T
            S_T = S_T[:2, :]
            crop = cv2.warpAffine(image, S_T, (output_resolution[0] * zoom_factor, output_resolution[1] * zoom_factor),
                                  flags=cv2.INTER_NEAREST)
            crop = cv2.resize(crop, tuple(output_resolution), interpolation=cv2.INTER_AREA)
            T = np.copy(T[:2, :])

        else:
            T = np.copy(T[:2, :])
            crop = cv2.warpAffine(image, T, tuple(output_resolution))

        if points.shape[2] == 2:
            points = np.pad(points, ((0, 0), (0, 0), (0, 1)), 'constant')
            points[:, :, 2] = 1

        points = points.transpose((0, 2, 1))
        points = T @ points
        points = points.transpose((0, 2, 1))

        if self.aligned_bbox:
            min_x = np.min(points[:, :, 0], axis=1)
            max_x = np.max(points[:, :, 0], axis=1)
            min_y = np.min(points[:, :, 1], axis=1)
            max_y = np.max(points[:, :, 1], axis=1)

            transformed_points = []
            for i in range(len(points)):
                transformed_points.append([[min_x[i], min_y[i]], [max_x[i], min_y[i]], [max_x[i], max_y[i]], [min_x[i], max_y[i]]])

            points = np.array(transformed_points)

        return crop, points, T


class FixedImageTransform(object):
    def __init__(self, dx=0, dy=0, interpolation=cv2.INTER_AREA):
        self.dx = dx
        self.dy = dy
        self.interpolation = interpolation

    def generate_transform(self, image, aratio=1, points=np.zeros((0, 4, 2), dtype=np.float32), scale_w=1, scale_h=1,
                           center=None, output_resolution=None):
        resolution = (image.shape[1], image.shape[0])
        if output_resolution is None:
            output_resolution = (resolution[0] * scale_w, resolution[1] * scale_h)

        dx = int(self.dx / scale_w)
        dy = int(self.dy / scale_h)
        dx = random.randint(-dx, dx)
        dy = random.randint(-dy, dy)
        image_tmp = np.zeros_like(image)
        image_tmp[max(0, dy):min(image.shape[0], image.shape[0] + dy), max(0, dx):min(image.shape[1], image.shape[1] + dx)] = \
            image[max(0, -dy):min(image.shape[0], image.shape[0] - dy), max(0, -dx):min(image.shape[1], image.shape[1] - dx)]
        image = cv2.resize(image_tmp, (0, 0), fx=scale_w, fy=scale_h, interpolation=self.interpolation)

        if len(image.shape) == 2:
            crop = np.zeros([output_resolution[1], output_resolution[0]], dtype=image.dtype)
        else:
            crop = np.zeros([output_resolution[1], output_resolution[0], image.shape[2]], dtype=image.dtype)
        crop[:image.shape[0], :image.shape[1]] = image[:crop.shape[0], :crop.shape[1]]

        points_out = np.zeros_like(points)
        points_out[:, :, 0] = (points[:, :, 0] + dx) * scale_w
        points_out[:, :, 1] = (points[:, :, 1] + dy) * scale_h
        T = None
        return crop, points_out, T


class GeometricGenerator(object):
    def __init__(self, name, resolution=(156, 128), area=0.8, points_2=False, offset_x=0.0, offset_y=0.0, template_points=None, absolute_crop=False, aligned_bbox=False, affine=False, horizontal_shrink_distortion=0):
        self.points_2 = points_2
        self.name = name
        self.resolution = resolution
        self.area = area
        if not isinstance(self.area, (list, tuple)):
            self.area = [self.area, self.area]

        if template_points is not None:
            self.template_points = np.asarray(template_points, dtype=np.float32)
        else:
            self.template_points = self.get_template_points(offset_x, offset_y)

        self.absolute_crop = absolute_crop
        self.aligned_bbox = aligned_bbox
        self.affine = affine
        self.horizontal_shrink_distortion = horizontal_shrink_distortion

    def get_template_points(self, offset_x=0.0, offset_y=0.0):
        width = self.resolution[0] * self.area[0] / 2
        height = self.resolution[1] * self.area[1] / 2
        offset_x = self.resolution[0] * offset_x
        offset_y = self.resolution[1] * offset_y
        if self.points_2:
            template_points = np.asarray(
                [[0, -height + offset_y],
                 [0, +height + offset_y]], dtype=np.float32)
        else:
            template_points = np.asarray(
                [[-width + offset_x, -height + offset_y],
                 [+width + offset_x, -height + offset_y],
                 [+width + offset_x, +height + offset_y],
                 [-width + offset_x, +height + offset_y],
                 ], dtype=np.float32)

        return template_points

    def get_base_transform(self, points, ar):
        points = points * np.array([[1, 1. / ar]])
        if self.points_2:
            height = (np.sum((points[0] - points[-1]) ** 2) ** 0.5 + np.sum((points[1] - points[2]) ** 2) ** 0.5) / 2
            left = (points[0] + points[-1]) / 2
            right = (points[1] + points[2]) / 2
            center = (left + right) / 2
            normal = right - left
            normal[0], normal[1] = normal[1], -normal[0]
            dir = normal / (np.sum(normal ** 2) ** 0.5)
            len = height / 2

            bottom = center - dir * len
            top = center + dir * len

            dir_top = points[1] - points[0]
            dir_bottom = points[-1] - points[2]
            _, k2t = np.linalg.solve(np.asarray([[dir_top[0], -dir[0]], [dir_top[1], -dir[1]]]), np.asarray([center[0] - points[0][0], center[1] - points[0][1]]))
            _, k2b = np.linalg.solve(np.asarray([[dir_bottom[0], -dir[0]], [dir_bottom[1], -dir[1]]]), np.asarray([center[0] - points[2][0], center[1] - points[2][1]]))
            intersected_top = center + k2t * dir
            intersected_bottom = center + k2b * dir

            if self.name == "new":
                source_points = np.stack([intersected_top, intersected_bottom], axis=0)
            else:
                source_points = np.stack([top, bottom], axis=0)
        else:
            source_points = points

        if self.affine:
            template_points = np.array(self.template_points)
            if self.horizontal_shrink_distortion:
                unit = (self.resolution[0] * self.horizontal_shrink_distortion) / 10
                population = [i * unit for i in range(0, 10)]
                weights = [0.25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05]
                shifts = random.choices(population, weights, k=4)

                if abs(shifts[0] - shifts[3]) > self.horizontal_shrink_distortion * 0.33:
                    shifts[0] = shifts[3]
                if abs(shifts[1] - shifts[2]) > self.horizontal_shrink_distortion * 0.33:
                    shifts[1] = shifts[2]

                if random.randint(0, 1):
                    shifts[0] = shifts[3]
                else:
                    shifts[1] = shifts[2]

                shifts[1] = -shifts[1]
                shifts[2] = -shifts[2]

                try:
                    for s, shift in enumerate(shifts):
                        template_points[s][0] += shift
                except IndexError:
                    raise Exception("Parameters points_2 and affine are both set to true, but they are mutually exclusive.")

            template_points = np.float32(template_points)
            source_points = np.float32(source_points)

            if np.shape(source_points) != (4,2) or np.shape(template_points) != (4,2):
                T = nudged.estimate(source_points, self.template_points)
                return np.asarray(T.get_matrix())

            T_ = cv2.getAffineTransform(source_points[:3], template_points[:3])
            T = np.zeros((3, 3))
            T[0] = T_[0]
            T[1] = T_[1]
            T[2,2] = 1.0
            return T
        else:
            T = nudged.estimate(source_points, self.template_points)
            return np.asarray(T.get_matrix())

    def generate_crop(self, image, points, ar=1, type=""):
        pass

    def freeze(self, name):
        return {name: {}}


class RandomGeometricGenerator_2(GeometricGenerator):
    def __init__(self, name, resolution=(156, 128), points_2=True, intersection_area=0.8, offset_x=0.0, offset_y=0.0, rot=0.0, scale=0., anisotropic=0.0, vertical_translation=0.00,
                 horizontal_translation=0.00, absolute_crop=False, aligned_bbox=False, affine=False, horizontal_shrink_distortion=0):
        super().__init__(name, resolution, intersection_area, points_2=points_2, offset_x=offset_x, offset_y=offset_y, absolute_crop=absolute_crop, aligned_bbox=aligned_bbox, affine=affine, horizontal_shrink_distortion=horizontal_shrink_distortion)
        self.rot = rot
        self.scale = scale
        self.anisotropic = anisotropic
        self.horizontal_translation = horizontal_translation
        self.vertical_translation = vertical_translation
        self.counter = 0

    def generate_crop(self, image, points, ar=1, type=""):
        if self.aligned_bbox:
            min_x = np.min(points[:, 0])
            max_x = np.max(points[:, 0])
            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])

            points = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])

        points = np.copy(points)
        AR = np.asarray(
            [[1, 0, 0],
             [0, 1.0 / ar, 0],
             [0, 0, 1]])

        T = self.get_base_transform(points, ar)

        r = np.random.randn() * self.rot
        R = np.asarray(
            [[np.cos(r), -np.sin(r), 0],
             [np.sin(r), np.cos(r), 0],
             [0, 0, 1]])
        s = 2 ** (np.random.randn() * self.scale)
        S = np.asarray(
            [[s * 2 ** (np.random.randn() * self.anisotropic), np.random.randn() * self.anisotropic, 0],
             [0, s * (2 ** (np.random.randn() * self.anisotropic)), 0],
             [0, 0, 1]])
        D = np.asarray(
            [[1, 0, np.random.randn() * self.horizontal_translation * self.resolution[0] + self.resolution[0] / 2],
             [0, 1, np.random.randn() * self.vertical_translation * self.resolution[1] + self.resolution[1] / 2],
             [0, 0, 1]])
        T = D @ S @ R @ T @ AR
        T = np.copy(T[:2, :])
        crop = cv2.warpAffine(image, T, (int(self.resolution[0]), int(self.resolution[1])))

        points = points[:, np.newaxis, :]
        points = cv2.transform(points, T)
        points = points[:, 0, :]

        if self.absolute_crop:
            crop_border = 16
            max_x = int(min(np.max(points[:, 0])+crop_border, crop.shape[1]))
            min_x = int(max(np.min(points[:, 0])-crop_border, 0))
            crop[:, :min_x] = 0
            crop[:, max_x:] = 0

        return crop, points, T

    def freeze(self, name):
        return {name: {
            "type": "FixedGeometricGenerator",
            "resolution": self.resolution,
            "intersection_area": self.area,
            "points_2": self.points_2,
            "absolute_crop": self.absolute_crop,
            "aligned_bbox": self.aligned_bbox,
            "affine": self.affine
        }}


class FixedGeometricGenerator(GeometricGenerator):
    def __init__(self, name, points_2=True, resolution=(156, 128), intersection_area=0.65, offset_x=0.0, offset_y=0.0, absolute_crop=False, aligned_bbox=False, affine=False):
        super().__init__(name, resolution, intersection_area, points_2=points_2, offset_x=offset_x, offset_y=offset_y, absolute_crop=absolute_crop, aligned_bbox=aligned_bbox, affine=affine)

    def generate_crop(self, image, points, ar=1, type=""):
        AR = np.asarray(
            [[1, 0, 0],
             [0, 1.0 / ar, 0],
             [0, 0, 1]])

        source_points = points * np.array([[1, 1.0 / ar]])
        T = self.get_base_transform(points, ar)
        D = np.asarray(
            [[1, 0, self.resolution[0] / 2],
             [0, 1, self.resolution[1] / 2],
             [0, 0, 1]])
        T = D @ T @ AR
        T = np.copy(T[:2, :])
        crop = cv2.warpAffine(image, T, (int(self.resolution[0]), int(self.resolution[1])))

        points = points[:, np.newaxis, :]
        points = cv2.transform(points, T)
        points = points[:, 0, :]

        if self.aligned_bbox:
            min_x = np.min(points[:, 0])
            max_x = np.max(points[:, 0])
            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])

            points = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])

        if self.absolute_crop:
            crop_border = 16
            max_x = int(min(np.max(points[:, 0])+crop_border, crop.shape[1]))
            min_x = int(max(np.min(points[:, 0])-crop_border, 0))
            crop[:, :min_x] = 0
            crop[:, max_x:] = 0

        return crop, points, T

    def freeze(self, name):
        return {name: {
            "type": "FixedGeometricGenerator",
            "resolution": self.resolution,
            "intersection_area": self.area,
            "points_2": self.points_2,
            "absolute_crop": self.absolute_crop,
            "aligned_bbox": self.aligned_bbox,
            "affine": self.affine
        }}


class RandomDoubleLPGenerator(object):
    def __init__(self, name, geometric_generator, overlay=0):
        self.name = name
        self.geometric_generator = copy.deepcopy(geometric_generator)
        self.overlay = overlay
        self.resolution = self.geometric_generator.resolution

        self.geometric_generator.resolution = (self.resolution[0] / 2, self.resolution[1])
        self.geometric_generator.template_points = self.geometric_generator.get_template_points()

    def generate_crop(self, image, points, ar=1, type=""):
        points_0 = np.array(
            [points[0], points[1], (points[1] * (1 - self.overlay) + points[2] * (1 + self.overlay)) / 2, (points[0] * (1 - self.overlay) + points[3] * (1 + self.overlay)) / 2])
        points_1 = np.array(
            [(points[0] * (1 + self.overlay) + points[3] * (1 - self.overlay)) / 2, (points[1] * (1 + self.overlay) + points[2] * (1 - self.overlay)) / 2, points[2], points[3]])
        crop0, points0, T0 = self.geometric_generator.generate_crop(image, points_0, ar)
        crop1, points1, T1 = self.geometric_generator.generate_crop(image, points_1, ar)
        crop0[:, -2] = 0
        crop0[:, -1] = 255
        crop1[:, 0] = 0
        crop1[:, 1] = 255

        crop = np.hstack(np.asarray([crop0, crop1]))
        points = np.asarray([points0[0], points0[1], (self.resolution[0] / 2) + points1[2], (self.resolution[0] / 2) + points1[3]])
        return crop, points, []

    def freeze(self, name):
        return {
            name: {
                "type": "FixedGeometricGenerator",
                "resolution": self.resolution,
                "intersection_area": self.geometric_generator.area,
                "transform_double": True,
                "transform_double_ratio": self.overlay,
                "points_2": self.geometric_generator.points_2
            }
        }


class CombineGeometricGenerator(object):
    def __init__(self, name, transformers, key_name, transformer_class_mapping={}, default_transformer_name=""):
        self.name = name
        self.transformers = transformers
        self.key_name = key_name
        self.transformer_class_mapping = transformer_class_mapping
        self.inv_mapping = {}
        self.default_transformer_name = default_transformer_name

        for key in self.transformer_class_mapping:
            for id in self.transformer_class_mapping[key]:
                self.inv_mapping[id] = key

    def generate_crop(self, image, points, ar=1, type=""):
        act_transformer_name = self.default_transformer_name
        if self.key_name in type and type[self.key_name] in self.inv_mapping:
            act_transformer_name = self.inv_mapping[type[self.key_name]]
        else:
            if isinstance(type, list) and len(type) > 0:  # muze byt pole, vezmu prvni
                type = type[0]
            if isinstance(type, str) and type in self.inv_mapping:
                act_transformer_name = self.inv_mapping[type]

        return self.transformers[act_transformer_name].generate_crop(image, points, ar, type)

    def freeze(self, name):
        ret = {}
        trans_names = []
        for id, tran in enumerate(self.transformers):
            trans_names.append(self.transformers[tran].name)
            ret.update(self.transformers[tran].freeze(self.transformers[tran].name))
        ret[name] = {
            "type": "CombineGeometricGenerator",
            "transformers": trans_names,
            "key_name": self.key_name,
            "transformer_class_mapping": self.transformer_class_mapping
        }
        return ret


if __name__ == "__main__":
    from datasets.object_loader import ObjectLoader

    image_gen = RandomImageTransform(rot=(-0.2, 0.2))

    object_loader = ObjectLoader(grayscale=False)
    object_loader.load_from_db(
        database_uri="postgresql+psycopg2://cognitechna:HelsoProAnnotator2020@localhost:5432/annotator_lp",
        ann_filters={'type': ['SINGLE', 'DOUBLE']},
        object_lists=['Double_transformation_test'],
        use_ignored_positions=True,
        image_path="/mnt/HDD2/annotator/"
    )

    img_res = [128, 128]
    generators = [
        (FixedGeometricGenerator("fixed_4", False, img_res, 0.6), True),
        (FixedGeometricGenerator("fixed_2", True, img_res, 0.4), True),
        (FixedGeometricGenerator("new", True, img_res, 0.4), True),
        (RandomDoubleLPGenerator("fixed_double", FixedGeometricGenerator("fixed_2", True, img_res * np.asarray([4, 1]), 0.6, )), False),
        (RandomDoubleLPGenerator("fixed_double", FixedGeometricGenerator("new", True, img_res * np.asarray([4, 1]), 0.6, )), False)
    ]

    for img, position in object_loader:

        crops = []
        crop, image_points, T = generators[0][0].generate_crop(img, position.points, position.image_aratio, position.values)
        pts = image_points.reshape((1, -1, 2)).astype(np.int32)
        cv2.polylines(crop, pts, True, (0, 128, 0), 2)
        crops.append(crop)

        for point in position.points:
            offset = np.random.normal(0, 3.5, size=(2))
            point += offset
        print(position.points)

        for gen, draw in generators:
            crop, image_points, T = gen.generate_crop(img, position.points, position.image_aratio, position.values)
            if draw:
                pts = image_points.reshape((1, -1, 2)).astype(np.int32)
                cv2.polylines(crop, pts, True, (0, 128, 0), 2)
            crops.append(crop)
        crops = np.hstack(crops)
        cv2.imshow("img", crops)
        if cv2.waitKey() == 27:
            break