import os
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import functools

from wsipack.utils.df_utils import print_df
from wsipack.utils.path_utils import PathUtils
from wsipack.wsi.wsi_file_utils import delete_slide, copy_slide

from PIL import Image

try:
    from multiresolutionimageinterface import (
        MultiResolutionImageReader,
        MultiResolutionImage,
    )
except ImportError:
    MultiResolutionImageReader = None
    MultiResolutionImage = None
    print("cannot import MultiResolutionImage")


try:
    from openslide import OpenSlide
except ImportError:
    OpenSlide = None
    print("cannot import OpenSLide")


class InvalidSpacingError(ValueError):
    def __init__(self, image_path, spacing, spacings, margin):

        super().__init__(
            f"Image: '{image_path}', with available pixels spacings: {spacings}, does not contain a level corresponding to a pixel spacing of {spacing} +- {margin}"
        )

        self._image_path = image_path
        self._spacing = spacing
        self._spacings = spacings
        self._margin = margin

    def __reduce__(self):
        return (
            InvalidSpacingError,
            (self._image_path, self._spacing, self._spacings, self._margin),
        )


class UnsupportedVendorError(KeyError):
    def __init__(self, image_path, properties):

        super().__init__(
            f"Image: '{image_path}', with properties: {properties}, is not in part of the supported vendors"
        )

        self._image_path = image_path
        self._properties = properties

    def __reduce__(self):
        return (UnsupportedVendorError, (self._image_path, self._properties))


class ImageReader(object):
    def __init__(self, image_path: str, spacing_tolerance: float = 0.3, cache_path=None, verbose=False, zoom=False) -> None:
        self._image_path = image_path
        self._extension = os.path.splitext(image_path)[-1]
        self._spacing_tolerance = spacing_tolerance
        self._cache_path = cache_path
        self._verbose = verbose
        self._cache_image()

        self._zoom = zoom

        self._shapes, self._downsamplings, self._spacings = self._init_shapes_downsamplings_spacings()

    @abstractmethod
    def _init_shapes_downsamplings_spacings(self):
        pass

    def _get_path(self):
        path = str(self._cache_path) if self._cache_path else str(self._image_path)
        return path

    @abstractmethod
    def _close(self):
        pass

    @property
    def filepath(self) -> str:
        return self._image_path

    @property
    def extension(self) -> str:
        return self._extension

    @property
    def spacings(self) -> List[float]:
        return self._spacings

    @property
    def shapes(self) -> List[Tuple[int, int]]:
        """ [(width,height), ...] """
        return self._shapes

    @property
    def downsamplings(self) -> List[float]:
        return self._downsamplings

    @property
    def level_count(self) -> int:
        return len(self.spacings)

    def get_downsampling_from_level(self, level: int) -> float:
        return self.downsamplings[level]

    def level(self, spacing: float) -> int:
        spacing_margin = spacing * self._spacing_tolerance
        for level, spacing_ in enumerate(self.spacings):
            if abs(spacing_ - spacing) <= spacing_margin:
                return level
        raise InvalidSpacingError(
            self._image_path, spacing, self.spacings, spacing_margin
        )

    def shape(self, spacing):
        level = self.level(spacing)
        shape = self.shapes[level]
        return shape

    def get_downsampling_from_spacing(self, spacing: float) -> float:
        return self.get_downsampling_from_level(self.level(spacing))

    @abstractmethod
    def get_patch(self, x, y, width, height, spacing, relative) -> np.ndarray:
        pass

    def read(self, spacing, x, y, width, height, center=False):
        if center:
            x, y = x - width // 2, y - height // 2
            width, height = width + width % 2, height + height % 2
        patch = self.get_patch(x=x, y=y, width=width, height=height, spacing=spacing, relative=True)
        # squeeze masks
        if len(patch.shape)==3 and patch.shape[2]==1:
            patch = patch[:,:,0]
        return patch

    def content(self, spacing=None):
        """
        Load a the content of the complete image from the given pixel spacing.
        Args:
            spacing (float): Pixel spacing to use to find the target level (micrometer).
        Returns:
            (np.ndarray): The loaded image.
        Raises:
            InvalidSpacingError: There is no level found for the given pixel spacing and tolerance.
        """

        if spacing is None:
            spacing = self.spacings[0]

        level = self.level(spacing=spacing)
        shape = self.shapes[level]
        content = self.read(x=int(0), y=int(0), width=int(shape[0]), height=int(shape[1]), spacing=spacing)
        return content

    def refine(self, spacing):
        """
        Get the pixel spacing of an existing level for the given pixel spacing within tolerance.
        Args:
            spacing (float): Pixel spacing (micrometer).
        Returns:
            float: Best matching pixel spacing of the closest level of the given pixel spacing.
        Raises:
            InvalidSpacingError: There is no level found for the given pixel spacing and tolerance.
        """
        return self.spacings[self.level(spacing=spacing)]

    def _cache_image(self):
        """
        Save the source and cached image paths and copy the image to the cache.

        Args:
            image_path (str): Path of the image to load.
            cache_path (str, None): Directory or file cache path.
        """

        if self._cache_path is None:
            return
        elif str(Path(self._image_path).absolute())==str(Path(self._cache_path).absolute()):
            print('no caching since image and cache are the same:', self._image_path)
            self._cache_path = None
            return

        if not os.path.isfile(self._cache_path):
            # Copy the source image to the cache location.
            cache_target = self._cache_path if os.path.splitext(self._image_path)[1].lower() == os.path.splitext(self._cache_path)[
                1].lower() else os.path.join(self._cache_path, os.path.basename(self._image_path))
            self._cache_path = cache_target
            # cached = copy_image(source_path=self._image_path, target_path=self._cache_path, overwrite=False)
            cached = copy_slide(self._image_path, self._cache_path, overwrite=False)
            if self._verbose and cached:
                print('cached %s to %s' % (self._image_path,self._cache_path), flush=True)

        else:
            if self._verbose:
                print('not caching %s since cache_path %s already exists' %\
                      (Path(self._image_path).name, self._cache_path))

    def close(self, clear=True):
        self._close()
        if clear and self._cache_path is not None:
            if self._verbose: print('removing cached %s' % self._cache_path, flush=True)
            delete_slide(self._cache_path)



class OpenSlideReader(ImageReader):
    def get_patch(self, x, y, width, height, spacing, relative=True) -> np.ndarray:
        """ center: samples with x,y as center """
        level = self.level(spacing)
        if relative:
            downsampling = self.get_downsampling_from_spacing(spacing)
            x, y = int(x * downsampling), int(y * downsampling)

        return np.array(
            self._openslide.read_region((int(x), int(y)), int(level), (int(width), int(height)))
        )[:, :, :3]

    def _init_shapes_downsamplings_spacings(self):
        self._openslide = OpenSlide(self._get_path())
        self._properties = self._openslide.properties
        self._channels = 3 #supports only 3 channel rgb

        spacing = None
        properties = self._openslide.properties
        try:
            spacing = float(properties["openslide.mpp-x"])
        except KeyError as key_error:
            try:
                unit = {"cm": 10000, "centimeter": 10000}[
                    properties["tiff.ResolutionUnit"]
                ]
                res = float(properties["tiff.XResolution"])
                spacing = unit / res
            except KeyError as key_error:
                raise UnsupportedVendorError(
                    self._image_path, properties
                ) from key_error

        spacings = [
            spacing * self._openslide.level_downsamples[level]
            for level in range(self._openslide.level_count)
        ]

        return self._openslide.level_dimensions, self._openslide.level_downsamples, spacings

    def _close(self):
        self._openslide.close()

    def get_image_label(self, size=128):
        """ non-anonymized images can have an image label"""
        img = self._openslide.associated_images.get('label',None)
        if img is None:
            return None
        # img =  Image.fromarray(img_arr)
        res = img.thumbnail((size,size), Image.ANTIALIAS)
        if res is not None:
            img = res #old version?
        img = np.array(img)
        return img

class AsapReader(ImageReader):

    def get_patch(self, x: int, y: int, width: int, height: int, spacing: float,
        relative: bool = False,
    ) -> np.ndarray:

        level = self.level(spacing)
        if relative:
            downsampling = self.get_downsampling_from_spacing(spacing)
            x, y = int(x * downsampling), int(y * downsampling)

        return np.array(
            self._reader.getUCharPatch(int(x), int(y), int(width), int(height), int(level))
        )

    def _init_shapes_downsamplings_spacings(self):
        self._reader = MultiResolutionImageReader().open(self._get_path())
        self._reader.setCacheSize(0)
        # self.__dict__.update(MultiResolutionImageReader().open(image_path).__dict__)
        shapes = [
                tuple(self._reader.getLevelDimensions(level))
                for level in range(self._reader.getNumberOfLevels())
            ]
        downsamples = [self._reader.getLevelDownsample(level) for level in range(self._reader.getNumberOfLevels())]

        spacings = [
                self._reader.getSpacing()[0] * downsampling
                for downsampling in downsamples
            ]
        return shapes, downsamples, spacings

    def _close(self):
        try:
            self._reader.close()
        except:
            pass #old asap version have close method

def slide_infos(dir):
    pathes = PathUtils.list_pathes(dir, containing_or=['.tif','svs'])
    entries = []
    for i,p in enumerate(pathes):
        reader = OpenSlideReader(p)
        spacing = reader.spacings[0]
        w = reader.shapes[0][0]
        h = reader.shapes[0][1]
        entries.append(dict(name=p.stem, spacing=spacing, w=w, h=h))
        print(entries[-1])
        reader.close()

    df = pd.DataFrame(entries)
    print_df(df)

def create_reader(path, reader='asap', **kwargs):
    if reader=='asap':
        return AsapReader(path, **kwargs)
    elif reader=='openslide':
        return OpenSlideReader(path, **kwargs)
    else:
        raise ValueError('unknown reader %s' % reader)

def read_wsi_spacing_os(slide_path):
    reader = OpenSlideReader(slide_path)
    spacing = reader.spacings[0]
    return spacing


def example():
    path = "/Users/witali/Daten/Radboud/data/bach/photos_tif_new_asap/b001.tif"
    areader = AsapReader(path)
    oreader = OpenSlideReader(path)
    content = areader.content(spacing=1)
    from wsipack.utils.cool_utils import showim
    print(areader.shapes, areader.spacings)
    print(oreader.shapes, oreader.spacings)
    showim(content)

if __name__ == '__main__':
    example()