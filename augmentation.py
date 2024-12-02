import imgaug.augmenters as iaa
import numpy as np


def func_random_images_eraser(images, random_state, parents, hooks):
    min_size = 0.1
    max_size = 0.5
    prob = 0.4
    result = []
    for image in images:
        if np.random.rand() < prob:
            image_aug = np.copy(image)
            size = (np.random.uniform(min_size, max_size, 2) * image.shape[:2] + 0.5).astype(int)
            i = np.random.randint(0, image.shape[0] - size[0])
            j = np.random.randint(0, image.shape[1] - size[1])
            image_aug[i:i+size[0], j:j+size[1]] = 0
            result.append(image_aug)
        else:
            result.append(image.copy())
    return result

AUGMENTATIONS = {
    'NONE': iaa.Identity(),
    'UNIVERSAL':
    iaa.Sequential([
        iaa.OneOf([iaa.geometric.Affine(scale=(0.85, 1.15), translate_percent=(-0.05, 0.05), rotate=(-5, 5),
                                        shear=(-20, 20), order=1),
                   iaa.geometric.PerspectiveTransform(scale=0.08)
                   ]),
        iaa.SomeOf(n=(1, 4), children=[
            iaa.convolutional.DirectedEdgeDetect(alpha=(0.05, 0.2), direction=(0.0, 1.0)),
            iaa.convolutional.EdgeDetect(alpha=(0.05, 0.15)),
            iaa.convolutional.Emboss(alpha=(0.05, 0.2), strength=(0.2, 0.7)),
            iaa.convolutional.Sharpen(alpha=(0.05, 0.2), lightness=(0.8, 1.2)),

            iaa.color.AddToHue(value=(-64, 64)),
            iaa.color.AddToBrightness(add=(-40, 40)),
            iaa.color.AddToSaturation(value=(-64, 64)),
            iaa.color.Grayscale(),
            iaa.color.Grayscale(),
            iaa.color.MultiplyBrightness(mul=(0.8, 1.2)),
            iaa.color.MultiplyHue(mul=(-0.7, 0.7)),
            iaa.color.MultiplySaturation(mul=(0.0, 2.0)),
            iaa.color.Posterize(nb_bits=(2, 8)),

            iaa.contrast.AllChannelsCLAHE(clip_limit=(0.1, 8), tile_grid_size_px=(3, 12), tile_grid_size_px_min=3),

            iaa.contrast.CLAHE(clip_limit=(0.1, 8), tile_grid_size_px=(3, 12), tile_grid_size_px_min=3),
            iaa.contrast.GammaContrast(gamma=(0.6, 1.8)),
            iaa.contrast.LogContrast(gain=(0.6, 1.4)),

            iaa.Alpha((0.2, 0.7), iaa.contrast.AllChannelsHistogramEqualization()),
            iaa.Alpha((0.2, 0.7), iaa.contrast.HistogramEqualization()),

            iaa.blur.BilateralBlur(d=(1, 7), sigma_color=(10, 250), sigma_space=(10, 250)),
            iaa.blur.GaussianBlur(sigma=(0.0, 2.5)),

            iaa.pillike.Solarize(p=1.0, threshold=128),
            iaa.pillike.EnhanceColor(factor=(0.5, 1.5)),
            iaa.pillike.EnhanceContrast(factor=(0.5, 1.5)),
            iaa.pillike.EnhanceBrightness(factor=(0.5, 1.5)),
            iaa.pillike.EnhanceSharpness(factor=(0.5, 1.5)),
            iaa.pillike.FilterEdgeEnhance(),
            iaa.pillike.FilterSharpen(),
            iaa.pillike.FilterDetail()
        ])]),

   'FACE_MASK':
    iaa.Sequential([
        iaa.OneOf([iaa.geometric.Affine(scale=(0.9, 1.1), translate_percent=(-0.03, 0.03), rotate=(-5, 5),
                                        shear=(-10, 10), order=1),
                   iaa.geometric.PerspectiveTransform(scale=0.05)
                   ]),
        iaa.SomeOf(n=(1, 4), children=[
            iaa.convolutional.DirectedEdgeDetect(alpha=(0.05, 0.15), direction=(0.0, 1.0)),
            iaa.convolutional.EdgeDetect(alpha=(0.05, 0.10)),
            iaa.convolutional.Emboss(alpha=(0.05, 0.15), strength=(0.2, 0.6)),
            iaa.convolutional.Sharpen(alpha=(0.05, 0.15), lightness=(0.8, 1.2)),

            iaa.color.AddToHue(value=(-24, 24)),
            iaa.color.AddToBrightness(add=(-32, 32)),
            iaa.color.AddToSaturation(value=(-32, 32)),
            iaa.color.Grayscale(),
            iaa.color.Grayscale(),
            iaa.color.MultiplyBrightness(mul=(0.8, 1.2)),
            iaa.color.MultiplyHue(mul=(-0.3, 0.3)),
            iaa.color.MultiplySaturation(mul=(0.5, 1.5)),
            iaa.color.Posterize(nb_bits=(4, 8)),

            iaa.contrast.GammaContrast(gamma=(0.6, 1.8)),
            iaa.contrast.LogContrast(gain=(0.6, 1.4)),

            iaa.blur.GaussianBlur(sigma=(0.0, 3.5)),
            iaa.blur.MotionBlur(k=(3, 9)),
            iaa.arithmetic.AdditiveGaussianNoise(scale=(5, 30), per_channel=True),
            iaa.arithmetic.AdditiveLaplaceNoise(scale=(5, 30), per_channel=True),
        ]),
        iaa.Lambda(func_images=func_random_images_eraser)
    ]),

    'POLYP':
        iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.geometric.Affine(scale=(0.9, 1.1), translate_percent=(-0.03, 0.03), rotate=(-5, 5),
                                            shear=(-10, 10), order=1),
                       iaa.geometric.PerspectiveTransform(scale=0.05),
                       # random mirrors
                       ]),
            iaa.SomeOf(n=(1, 4), children=[
                iaa.convolutional.DirectedEdgeDetect(alpha=(0.05, 0.15), direction=(0.0, 1.0)),
                iaa.convolutional.EdgeDetect(alpha=(0.05, 0.10)),
                iaa.convolutional.Emboss(alpha=(0.05, 0.15), strength=(0.2, 0.6)),
                iaa.convolutional.Sharpen(alpha=(0.05, 0.15), lightness=(0.8, 1.2)),

                iaa.color.AddToHue(value=(-24, 24)),
                iaa.color.AddToBrightness(add=(-32, 32)),
                iaa.color.AddToSaturation(value=(-32, 32)),
                iaa.color.Grayscale(),
                iaa.color.MultiplyBrightness(mul=(0.8, 1.2)),
                iaa.color.MultiplyHue(mul=(-0.2, 0.2)),
                iaa.color.MultiplySaturation(mul=(0.7, 1.3)),
                iaa.color.Posterize(nb_bits=(4, 8)),

                iaa.contrast.GammaContrast(gamma=(0.6, 1.8)),
                iaa.contrast.LogContrast(gain=(0.6, 1.4)),

                iaa.blur.GaussianBlur(sigma=(0.0, 3.5)),
                iaa.blur.MotionBlur(k=(3, 9)),
                iaa.arithmetic.AdditiveGaussianNoise(scale=(2, 10), per_channel=True),
                iaa.arithmetic.AdditiveLaplaceNoise(scale=(2, 10), per_channel=True),
            ]),
            iaa.Lambda(func_images=func_random_images_eraser)
        ]),
    'POLYP_LITE':
        iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.geometric.Affine(scale=(0.9, 1.1), translate_percent=(-0.03, 0.03), rotate=(-25, 25),
                                            shear=(-15, 15), order=1),
                       iaa.geometric.PerspectiveTransform(scale=0.05),
                       ]),
            iaa.SomeOf(n=(1, 4), children=[
                iaa.convolutional.DirectedEdgeDetect(alpha=(0.05, 0.15), direction=(0.0, 1.0)),
                iaa.convolutional.EdgeDetect(alpha=(0.05, 0.10)),
                iaa.convolutional.Emboss(alpha=(0.05, 0.15), strength=(0.2, 0.6)),
                iaa.convolutional.Sharpen(alpha=(0.05, 0.15), lightness=(0.8, 1.2)),

                iaa.color.AddToHue(value=(-16, 16)),
                iaa.color.AddToBrightness(add=(-24, 24)),
                iaa.color.AddToSaturation(value=(-24, 24)),
                iaa.color.Grayscale(),
                iaa.color.MultiplyBrightness(mul=(0.8, 1.2)),
                iaa.color.MultiplySaturation(mul=(0.8, 1.2)),

                iaa.contrast.GammaContrast(gamma=(0.7, 1.3)),
                iaa.contrast.LogContrast(gain=(0.8, 1.2)),

                iaa.blur.GaussianBlur(sigma=(0.0, 1.5)),
                iaa.blur.MotionBlur(k=(3, 6)),
                iaa.arithmetic.AdditiveGaussianNoise(scale=(2, 10), per_channel=True),
                iaa.arithmetic.AdditiveLaplaceNoise(scale=(2, 10), per_channel=True),
            ]),
            iaa.Lambda(func_images=func_random_images_eraser)
        ]),
    'POLYP_LITE_2':
        iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.geometric.Affine(scale=(0.9, 1.1), translate_percent=(-0.03, 0.03), rotate=(-180, 180),
                                            shear=(-15, 15), order=1),
                       iaa.geometric.PerspectiveTransform(scale=0.05),
                       ]),
            iaa.SomeOf(n=(1, 3), children=[
                iaa.color.AddToHue(value=(-16, 16)),
                iaa.color.AddToBrightness(add=(-24, 24)),
                iaa.color.AddToSaturation(value=(-24, 24)),
                iaa.color.MultiplyBrightness(mul=(0.8, 1.2)),
                iaa.color.MultiplySaturation(mul=(0.8, 1.2)),
                iaa.contrast.GammaContrast(gamma=(0.7, 1.3)),
                iaa.contrast.LogContrast(gain=(0.8, 1.2)),
                iaa.arithmetic.AdditiveGaussianNoise(scale=(2, 10), per_channel=True),
            ]),
            iaa.Lambda(func_images=func_random_images_eraser)
        ]),
    'POLYP_LITE_3':
        iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.geometric.Affine(scale=(0.9, 1.1), translate_percent=(-0.03, 0.03), rotate=(-180, 180),
                                            shear=(-15, 15), order=1),
                       iaa.geometric.PerspectiveTransform(scale=0.05),
                       ]),
            iaa.SomeOf(n=(1, 3), children=[
                iaa.color.AddToHue(value=(-8, 8)),
                iaa.color.AddToBrightness(add=(-12, 12)),
                iaa.color.AddToSaturation(value=(-12, 12)),
                iaa.color.MultiplyBrightness(mul=(0.9, 1.1)),
                iaa.color.MultiplySaturation(mul=(0.9, 1.1)),
                iaa.contrast.GammaContrast(gamma=(0.8, 1.2)),
                iaa.contrast.LogContrast(gain=(0.9, 1.1)),
                iaa.arithmetic.AdditiveGaussianNoise(scale=(2, 10), per_channel=True),
            ]),
            iaa.Lambda(func_images=func_random_images_eraser)
        ]),
    'POLYP_LITE_4':
        iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.geometric.Affine(scale=(0.9, 1.1), translate_percent=(-0.03, 0.03), rotate=(-180, 180),
                                            shear=(-15, 15), order=1),
                       iaa.geometric.PerspectiveTransform(scale=0.05),
                       ]),
            iaa.SomeOf(n=(1, 3), children=[
                iaa.color.AddToHue(value=(-8, 8)),
                iaa.color.AddToBrightness(add=(-34, 34)),
                iaa.color.AddToSaturation(value=(-24, 24)),
                iaa.color.MultiplyBrightness(mul=(0.7, 1.3)),
                iaa.color.MultiplySaturation(mul=(0.8, 1.2)),
                iaa.contrast.GammaContrast(gamma=(0.5, 2)),
                iaa.contrast.LogContrast(gain=(0.7, 1.5)),
            ]),
            iaa.Lambda(func_images=func_random_images_eraser)
        ]),
    'POLYP_GEOMETRIC':
        iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.geometric.Affine(scale=(0.9, 1.1), translate_percent=(-0.03, 0.03), rotate=(-25, 25),
                                            shear=(-15, 15), order=1),
                       iaa.geometric.PerspectiveTransform(scale=0.05),
                       ]),
            iaa.Lambda(func_images=func_random_images_eraser)
        ]),
    'FLIP':
        iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Lambda(func_images=func_random_images_eraser)
        ]),



'LITE':
    iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.OneOf([iaa.geometric.Affine(scale=(0.9, 1.1), translate_percent=(-0.03, 0.03), rotate=(-5, 5),
                                        shear=(-10, 10), order=1),
                   iaa.geometric.PerspectiveTransform(scale=0.05)
                   ]),
        iaa.SomeOf(n=(1, 4), children=[
            iaa.convolutional.DirectedEdgeDetect(alpha=(0.05, 0.15), direction=(0.0, 1.0)),
            iaa.convolutional.EdgeDetect(alpha=(0.05, 0.10)),
            iaa.convolutional.Emboss(alpha=(0.05, 0.15), strength=(0.2, 0.6)),
            iaa.convolutional.Sharpen(alpha=(0.05, 0.15), lightness=(0.8, 1.2)),

            iaa.color.AddToHue(value=(-24, 24)),
            iaa.color.AddToBrightness(add=(-20, 20)),
            iaa.color.AddToSaturation(value=(-24, 24)),
            iaa.color.Grayscale(),
            iaa.color.Grayscale(),
            iaa.color.MultiplyBrightness(mul=(0.85, 1.15)),
            iaa.color.MultiplyHue(mul=(-0.3, 0.3)),
            iaa.color.MultiplySaturation(mul=(0.5, 1.5)),
            iaa.color.Posterize(nb_bits=(4, 8)),

            iaa.contrast.GammaContrast(gamma=(0.7, 1.4)),
            iaa.contrast.LogContrast(gain=(0.7, 1.3)),

            iaa.blur.GaussianBlur(sigma=(0.0, 2.5)),
        ]),
    ]),
   'LITE_MASK':
    iaa.Sequential([
        iaa.OneOf([iaa.geometric.Affine(scale=(0.9, 1.1), translate_percent=(-0.03, 0.03), rotate=(-5, 5),
                                        shear=(-10, 10), order=1),
                   iaa.geometric.PerspectiveTransform(scale=0.05)
                   ]),
        iaa.SomeOf(n=(1, 4), children=[
            iaa.convolutional.DirectedEdgeDetect(alpha=(0.05, 0.15), direction=(0.0, 1.0)),
            iaa.convolutional.EdgeDetect(alpha=(0.05, 0.10)),
            iaa.convolutional.Emboss(alpha=(0.05, 0.15), strength=(0.2, 0.6)),
            iaa.convolutional.Sharpen(alpha=(0.05, 0.15), lightness=(0.8, 1.2)),

            iaa.color.AddToHue(value=(-24, 24)),
            iaa.color.AddToBrightness(add=(-20, 20)),
            iaa.color.AddToSaturation(value=(-24, 24)),
            iaa.color.Grayscale(),
            iaa.color.Grayscale(),
            iaa.color.MultiplyBrightness(mul=(0.85, 1.15)),
            iaa.color.MultiplyHue(mul=(-0.3, 0.3)),
            iaa.color.MultiplySaturation(mul=(0.5, 1.5)),
            iaa.color.Posterize(nb_bits=(4, 8)),

            iaa.contrast.GammaContrast(gamma=(0.7, 1.4)),
            iaa.contrast.LogContrast(gain=(0.7, 1.3)),

            iaa.blur.GaussianBlur(sigma=(0.0, 2.5)),
        ]),
        iaa.Lambda(func_images=func_random_images_eraser)
    ])
}
