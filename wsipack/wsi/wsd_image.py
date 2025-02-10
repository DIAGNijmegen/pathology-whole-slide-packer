from bisect import bisect_left

from typing import List, Tuple

import shutil

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
try: #0.0.16
    from wholeslidedata.accessories.asap.imagewriter import WholeSlideMaskWriter, WholeSlideImageWriter, \
        WholeSlideImageWriterBase
except:
    from wholeslidedata.interoperability.asap.imagewriter import WholeSlideMaskWriter, WholeSlideImageWriter, \
        WholeSlideImageWriterBase
from wholeslidedata.image.wholeslideimage import WholeSlideImage

from wsipack.utils.cool_utils import showims, take_closest_smallest_numer, mkdir, timer

def take_closest_level(spacings, spacing):
    pos = bisect_left(spacings, spacing)
    if pos == 0:
        return pos
    if pos == len(spacings):
        return pos - 1
    if spacings[pos] - spacing < spacing - spacings[pos - 1]:
        return pos
    return pos - 1

class PixelSpacingLevelError(Exception):
    """Raise when there is no level for the spacing within the tolerance."""
    def __init__(self, path, spacing, tolerance, *args, **kwargs):
        super().__init__('no level for spacing %.3f with tolerance %.2f, path: %s, %s' % \
                         (spacing, tolerance, str(path), str(kwargs)))

class ImageReader(WholeSlideImage):
    def __init__(self, path, backend='asap', cache_path=None, verbose=False, spacing_tolerance=0.3):
        if cache_path is not None and str(cache_path)!=str(path) and Path(cache_path)!=Path(path).parent:
            path = Path(path)
            cache_path = Path(cache_path)
            if cache_path.suffix!=path.suffix:
                mkdir(cache_path)
                cache_path = cache_path/path.name
            if not cache_path.exists():
                shutil.copyfile(src=str(path), dst=str(cache_path))
            cache_path = str(cache_path)
            path = cache_path
        super().__init__(str(path), backend=backend)
        # self._calc_spacing_ranges()
        self.cache_path = cache_path
        self.verbose = verbose
        self.spacing_tolerance = spacing_tolerance
        if self.verbose:
            print('initialized reader for %s, backend=%s' % (self.path, str(self._backend)))
            print('%d channels, shapes %s, downsamplings %s' % (self.channels, str(self.shapes), str(self.downsamplings)))

    @property
    def channels(self):
        return self._backend.getSamplesPerPixel()

    @property
    def path(self):
        return self._path

    @property
    def shapes(self) -> List[Tuple[int, int]]:
        """ (h,w) """
        shapes_ = super().shapes
        #convert from (w,h) to (h,w)
        shapes = [(w,h) for (h,w) in shapes_]
        return shapes

    def content(self, spacing):
        content = self.get_slide(spacing)
        # TODO: check openslide compatibility
        # content = self._mask_convert(content)
        return content

    def shape(self, spacing):
        level = self.get_level_from_spacing(spacing)
        shape = self.shapes[level]
        return shape

    def get_slide(self, spacing):
        shape = self.shape(spacing)
        if self.verbose:
            print('reading at spacing %.2f (%d,%d)' % (spacing, shape[0], shape[1]))
        content = self.read(spacing, 0, 0, shape[0], shape[1])
        if self.verbose:
            print('read slide with shape %s' % str(content.shape))
        return content
        # spacing = self.refine(spacing)
        # level = self.get_level_from_spacing(spacing)
        # shape = self.shapes[level]
        # return self.get_patch(0, 0, shape[1], shape[0], spacing, center=False)

    def _mask_convert(self, img):
        if 'openslide' in str(self._backend.name).lower() \
                and (img[:,:,0]==img[:,:,1]).all() and (img[:,:,0]==img[:,:,2]).all():
            if self.verbose:
                print('openslide _mask_convert')
            img = img[:,:,0] #openslide returns wxhx3 for masks, asap wxh.1
            img = img[:,:,None]
        return img

    # def read(self, spacing, row, col, width, height):
    #     patch = self.get_patch(col, row, height, width, spacing, center=False, relative=True)
    def read(self, spacing, row, col, height, width):
        patch = self.get_patch(col, row, width, height, spacing, center=False, relative=True)
        # patch = self._mask_convert(patch)
        # return patch.transpose([1,0,2])
        patch = patch.squeeze()
        return patch

    def refine(self, spacing):
        self.level(spacing) #check for missing spacing
        return self.get_real_spacing(spacing)

    def level(self, spacing: float) -> int:
        level = take_closest_level(self._spacings, spacing)
        spacing_margin = spacing * self.spacing_tolerance

        if abs(self.spacings[level] - spacing) > spacing_margin:
            raise PixelSpacingLevelError(self.path, spacing, self.SPACING_MARGIN, spacings=self.spacings)

        return level

    def close(self, clear=True):
        if self.cache_path is not None and Path(self.cache_path).exists():
            if self.verbose: print('deleting cached image %s' % self.cache_path)
            Path(self.cache_path).unlink()
            # Path(self.cache_path).unlink(missing_ok=True) only python 3.8
        super().close()

    def closest_spacing(self, spacing):
        try:
            closest = self.refine(spacing)
            return closest
        except PixelSpacingLevelError as ex:
            pos = take_closest_level(self.spacings, spacing)
            return self.spacings[pos]

class ImageWriter(object):
    def __init__(self, path, spacing, shape, tile_size, quality=80):
        self.path = path
        self.spacing = spacing
        self.shape = shape
        self.tile_size = tile_size
        self.quality = quality

        self.writer = None

    def write(self, patch, row, col):
        self._init_writer(patch)

        if len(patch.shape) == 2:
            patch = patch[:, :, None]

        self.writer.write_tile(patch, (col, row))

    def _init_writer(self, patch):
        if self.writer is None:
            kwargs = {}
            if 'int' in str(patch.dtype):
                if len(patch.shape) == 3 and patch.shape[-1] == 3:
                    self.writer = WholeSlideImageWriter()
                    kwargs['jpeg_quality'] = self.quality
                    kwargs['interpolation'] = 'linear'
                else:
                    self.writer = WholeSlideMaskWriter()
            else:
                self.writer = WholeSlideMaskWriter()

            self.writer.write(self.path, dimensions=self.shape[:2][::-1], spacing=self.spacing,
                              tile_shape=(self.tile_size, self.tile_size), **kwargs)

    def close(self):
        self.writer.finishImage()

class ArrayImageWriter(object):
    def __init__(self, cache_dir=None, tile_size=512, suppress_mir_stdout=True, skip_empty=False, jpeg_quality=80):
        self.cache_dir = cache_dir
        self.tile_size = tile_size
        self.suppress_mir_stdout = suppress_mir_stdout
        self.skip_empty = skip_empty
        self.jpeg_quality = jpeg_quality

    def write_array(self, arr, path, spacing, verbose=False):
        from ..wsi.wsi_utils import write_array_with_writer
        tile_size = self.tile_size
        if min(arr.shape[:2]) < tile_size:
            tile_size = take_closest_smallest_numer([8, 16, 32, 64, 128, 256], min(arr.shape[:2]))
        if len(arr.shape)==2:
            tile_shape = (tile_size, tile_size)
        else:
            tile_shape = (tile_size, tile_size, arr.shape[2])
        if arr.dtype == bool:
            arr = arr.astype(np.uint8)

        jpeg_quality = None
        # f = f.transpose(1, 2, 0)
        kwargs = {}
        if 'int' in str(arr.dtype):
            if len(arr.shape)==3 and arr.shape[-1]==3:
                writer = WholeSlideImageWriter()
                kwargs['jpeg_quality'] = self.jpeg_quality
                kwargs['interpolation'] = 'linear'
            else:
                writer = WholeSlideMaskWriter()
        else:
            writer = WholeSlideMaskWriter()

        if len(arr.shape) == 2:
            arr = arr[:, :, None]

        mkdir(Path(path).parent)
        writer.write(path, dimensions=arr.shape[:2][::-1], spacing=spacing, tile_shape=tile_shape, **kwargs)
        # result = result.squeeze()
        # zoomed = scipy.ndimage.zoom(input=result, zoom=self.shift)
        # zoomed = zoomed[-slide_shape[0]:,-slide_shape[1]:]
        # out_spacing = slide_spacing
        # result = zoomed

        diaglike = WsdWriterWrapper(writer)
        # path = str(Path(path).absolute())
        write_array_with_writer(arr, diaglike, tile_size=tile_size)

class WsdWriterWrapper(object):
    def __init__(self, writer:WholeSlideImageWriterBase):
        self.writer = writer

    def write(self, tile, row, col):
        tile = tile.squeeze()
        self.writer.write_tile(tile, (col, row))

    def close(self):
        self.writer.finishImage()

def write_array(array, path, out_spacing, cache_dir=None):
    writer = ArrayImageWriter(cache_dir)
    writer.write_array(array, path, out_spacing)

