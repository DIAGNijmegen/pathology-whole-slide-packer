#Adapted from https://github.com/DIAGNijmegen/pathology-whole-slide-data version 0.0.16
import shutil

from pathlib import Path
import cv2
import abc
import multiresolutionimageinterface as mir
from multiresolutionimageinterface import MultiResolutionImageWriter
import numpy as np
from tqdm import tqdm

from wsipack.utils.cool_utils import mkdir, take_closest_smaller_number, showim
from wsipack.wsi.wsi_read import ImageReader, create_reader, AsapReader


class Writer(abc.ABC):
    def __init__(self, callbacks=None):
        self._callbacks = callbacks

class TileShapeError(Exception):
    pass

class CoordinateError(Exception):
    pass


class WholeSlideImageWriterBase(Writer, MultiResolutionImageWriter):
    def __init__(self, callbacks=()):
        Writer.__init__(self, callbacks)
        MultiResolutionImageWriter.__init__(self)

    def write_tile(self, tile, coordinates=None, mask=None):
        tile = self._apply_tile_callbacks(tile)
        tile = self._mask_tile(tile, mask)
        tile = self._crop_tile(tile)
        self._write_tile_to_image(tile, coordinates)

    def _write_tile_to_image(self, tile, coordinates):
        if coordinates:
            col, row = self._get_col_row(coordinates)
            if col is not None and row is not None:
                self.writeBaseImagePartToLocation(
                    tile.flatten().astype("uint8"), col, row
                )
        else:
            self.writeBaseImagePart(tile.flatten().astype("uint8"))

    def _get_col_row(self, coordinates):
        x, y = coordinates
        if x < self._dimensions[0] and x >= 0 and y < self._dimensions[1] and y >= 0:
            return x, y

        raise CoordinateError(
            f"Invalid coordinate x,y={x, y} with dimension setting {self._dimensions}"
        )

    def _apply_tile_callbacks(self, tile):
        for callback in self._callbacks:
            tile = callback(tile)
        return tile

    def _mask_tile(self, tile, mask):
        if mask is not None:
            if tile.ndim == 3 and mask.ndim == 2:
                mask = mask[..., None]   # expand to (H,W,1)
            tile *= mask
        return tile

    def _crop_tile(self, tile):
        if len(tile.shape) == 2:
            return tile[: self._tile_shape[1], : self._tile_shape[0]]

        if len(tile.shape) == 3:
            return tile[: self._tile_shape[1], : self._tile_shape[0], :]

        raise TileShapeError(
            f"Invalid tile shape: {tile.shape}, tile shape should contain at 2 or 3 dimensions"
        )

    def save(self):
        self.finishImage()


class AsapMaskWriter(WholeSlideImageWriterBase):
    def __init__(self, callbacks=(), suffix=".tif"):
        super().__init__(callbacks=callbacks)
        self._suffix = suffix

    def write(self, path, spacing, dimensions, tile_shape):
        self._path = str(path).replace(Path(path).suffix, self._suffix)
        # there is a small difference in the last digits of the spacing written by mir,
        # probably caused by rounding errors down the line
        self._spacing = float(spacing)
        self._dimensions = dimensions
        self._tile_shape = tile_shape

        print(f"Creating: {self._path}....")
        print(f"Spacing: {self._spacing}")
        print(f"Dimensions: {self._dimensions}")
        print(f"Tile_shape: {self._tile_shape}")

        self.openFile(self._path)
        self.setTileSize(self._tile_shape[0])

        try:
            self.setCompression(mir.Compression_LZW)
            self.setDataType(mir.DataType_UChar)
            self.setInterpolation(mir.Interpolation_NearestNeighbor)
            self.setColorType(mir.ColorType_Monochrome)
        except:
            self.setCompression(mir.LZW)
            self.setDataType(mir.UChar)
            self.setInterpolation(mir.NearestNeighbor)
            self.setColorType(mir.Monochrome)

        # set writing spacing
        pixel_size_vec = mir.vector_double()
        pixel_size_vec.push_back(self._spacing)
        pixel_size_vec.push_back(self._spacing)
        self.setSpacing(pixel_size_vec)
        self.writeImageInformation(self._dimensions[0], self._dimensions[1])


class AsapImageWriter(WholeSlideImageWriterBase):
    def __init__(self, callbacks=(), suffix=".tif"):
        super().__init__(callbacks=callbacks)
        self._suffix = suffix

    def write(self, path, spacing, dimensions, tile_shape, jpeg_quality=80, interpolation='nearest'):
        """ dimensions: (width, height) """
        self._path = str(path).replace(Path(path).suffix, self._suffix)
        self._spacing = spacing
        self._dimensions = dimensions
        self._tile_shape = tile_shape

        print(f"Creating: {self._path}....")
        print(f"Spacing: {self._spacing}")
        print(f"Dimensions: {self._dimensions}")
        print(f"Tile_shape: {self._tile_shape}")

        self.openFile(self._path)
        self.setTileSize(self._tile_shape[0])

        try:
            if interpolation=='nearest':
                self.setInterpolation(mir.Interpolation_NearestNeighbor)
            else:
                self.setInterpolation(mir.Interpolation_Linear)
            self.setDataType(mir.DataType_UChar)
            self.setColorType(mir.ColorType_RGB)
            self.setCompression(mir.Compression_JPEG)
        except:
            if interpolation=='nearest':
                self.setInterpolation(mir.NearestNeighbor)
            else:
                self.setInterpolation(mir.Linear)
            self.setDataType(mir.UChar)
            self.setColorType(mir.RGB)
            self.setCompression(mir.JPEG)

        self.setJPEGQuality(jpeg_quality)

        # set writing spacing
        pixel_size_vec = mir.vector_double()
        pixel_size_vec.push_back(self._spacing)
        pixel_size_vec.push_back(self._spacing)
        self.setSpacing(pixel_size_vec)
        self.writeImageInformation(self._dimensions[0], self._dimensions[1])


class ArrayImageWriter(object):
    def __init__(self, cache_dir=None, tile_size=512, suppress_mir_stdout=True, skip_empty=False, jpeg_quality=80, b2w=False):
        self.cache_dir = cache_dir
        self.tile_size = tile_size
        self.suppress_mir_stdout = suppress_mir_stdout
        self.skip_empty = skip_empty
        self.jpeg_quality = jpeg_quality
        self.b2w = b2w

    def write_array(self, arr, path, spacing):
        tile_size = self._adapt_tile_size(arr)
        shape = arr.shape[:2][::-1]
        writer = AsapWriter(path=path, spacing=spacing, tile_size=tile_size, shape=shape, jpeg_quality=self.jpeg_quality, cache_dir=self.cache_dir)

        if arr.dtype == bool:
            arr = arr.astype(np.uint8)

        mkdir(Path(path).parent)
        write_array_with_writer(arr, writer, tile_size=tile_size, b2w=self.b2w, close=True)

    def _adapt_tile_size(self, arr):
        tile_size = self.tile_size
        if min(arr.shape[:2]) < tile_size:
            try:
                tile_size = take_closest_smaller_number([8, 16, 32, 64, 128, 256], min(arr.shape[:2]))
            except:
                print('failed determining tile_size for array with shape %s' % str(arr.shape))
                raise
        return tile_size


class AsapWriter(object):
    def __init__(self, path, spacing, shape, tile_size, jpeg_quality=80, cache_dir=None):
        self.path = path
        self.spacing = spacing
        self.shape = shape
        self.tile_size = tile_size
        self.jpeg_quality = jpeg_quality
        self.writer = None
        self.cache_dir = cache_dir

    def write(self, patch, x, y):
        self._init_writer(patch)

        if len(patch.shape) == 2:
            patch = patch[:, :, None]

        self.writer.write_tile(patch, (x, y))

    def _init_writer(self, patch):
        path = self.path
        if str(self.cache_dir)!='None':
            mkdir(self.cache_dir)
            path = Path(self.cache_dir)/Path(path).name
        if self.writer is None:
            kwargs = {}
            if 'int' in str(patch.dtype):
                if len(patch.shape) == 3 and patch.shape[-1] == 3:
                    self.writer = AsapImageWriter()
                    kwargs['jpeg_quality'] = self.jpeg_quality
                    kwargs['interpolation'] = 'linear'
                else:
                    self.writer = AsapMaskWriter()
            else:
                self.writer = AsapMaskWriter()

            self.writer.write(path, dimensions=self.shape, spacing=self.spacing,
                              tile_shape=(self.tile_size, self.tile_size), **kwargs)

    def close(self):
        self.writer.finishImage()
        if str(self.cache_dir)!='None':
            print('copying %s to %s' % (self.writer._path, self.path))
            shutil.copyfile(self.writer._path, self.path)
            #delete the cache file
            Path(self.writer._path).unlink()

def write_arrayimage(array, path, out_spacing, cache_dir=None):
    writer = ArrayImageWriter(cache_dir)
    writer.write_array(array, path, out_spacing)


def write_array_with_writer(array, writer:AsapWriter, tile_size=512, close=True, b2w=False):
    if array.dtype==np.float16:
        array = array.astype(np.float32)

    shape = array.shape
    if len(shape)==2:
        n_channels = 1
    else:
        n_channels = shape[-1]

    for row in range(0, shape[0]+tile_size, tile_size): #+tile_size if array not divisible by tile_size
        if row >= shape[0]: continue
        for col in range(0, shape[1]+tile_size, tile_size):
            #print every 10th percent
            if col % (10*tile_size)==0:
                print('.', end='')
            if col >= shape[1]: continue
            tile = array[row:row+tile_size, col:col+tile_size]
            wtile = np.zeros((tile_size, tile_size, n_channels), dtype=array.dtype)
            # wtile = wtile.squeeze()
            if len(wtile.shape)!=len(tile.shape):
                wtile = wtile.squeeze()
                if len(tile.shape)==3: tile = tile.squeeze()
            tile = tile.copy()
            if b2w:
                # convert all-black pixels (0 in all thre dimensions) to white
                tile[(tile==0).all(axis=2)] = 255
            wtile[:tile.shape[0],:tile.shape[1]] = tile
            # print('writing tile from row', row, 'col', col, tile.shape, 'wtile', wtile.shape)
            writer.write(wtile, x=col, y=row)
            # print('done writing tile')
    if close:
        writer.close()


def write_tif_with_reader(reader:ImageReader, spacing, out_path, level=None, tile_size=512, overwrite_spacing=None, overwrite=False,
                          cache_dir=None):
    if Path(out_path).exists() and not overwrite:
        print('Skipping existing %s' % out_path)
        return
    if level is not None and spacing is not None:
        raise ValueError('Either level or spacing must be specified, not both.')
    if level is not None:
        spacing = reader.spacings[level]
    print('writing to %s with spacing %f and tile_size %d' % (out_path, spacing, tile_size))
    out_dir = Path(out_path).parent
    mkdir(out_dir)
    shape = reader.shape(spacing)
    write_spacing = spacing if overwrite_spacing is None else overwrite_spacing
    writer = AsapWriter(out_path, write_spacing, shape, tile_size=tile_size, cache_dir=cache_dir)
    spacing = reader.refine(spacing)

    print('getting tiles...')
    for y in tqdm(range(0, shape[1], tile_size)):
        for x in range(0, shape[0], tile_size):
            # Calculate actual readable dimensions
            # actual_height = min(tile_size, shape[1] - y)
            # actual_width = min(tile_size, shape[0] - x)
            patch = reader.read(spacing, x=x, y=y, width=tile_size, height=tile_size)
            writer.write(patch, x=x, y=y)

    writer.close()
    print(out_path)

def convert_slide(path, spacing, out_path, reader='asap', tile_size=512, level=None, overwrite_spacing=None, overwrite=False, cache_dir=None):
    """
    Convert a tif image to a new tif image with specified spacing and tile size.

    :param path: Path to the input tif image.
    :param reader: SlideReader instance for reading the image.
    :param spacing: Desired spacing for the output image.
    :param out_path: Path to save the converted tif image.
    :param tile_size: Size of the tiles to be used in the output image.
    """
    reader = create_reader(path, reader=reader)
    print(f'Converting {path} to %s with spacing {spacing}, level {level} and tile size {tile_size}')
    write_tif_with_reader(reader, spacing, out_path, tile_size=tile_size, level=level,
                          overwrite_spacing=overwrite_spacing, overwrite=overwrite, cache_dir=cache_dir)



def _example():
    path = "../../documentation/assets/example.tif"
    # path = "/Volumes/atb_hd/msi_mss/preprocessing/cptac_coad/packed/packed/05CO005-a23ef085-a2a1-4ffc-9d1a-56c0f0.tif"
    # path = "/Volumes/breast_her2/yale/her2/images_tif/Her2Neg_Case_66.tif"
    reader = AsapReader(path, downsample=False)
    print(reader.spacings)
    spacing = reader.spacings[0]
    spacing = 2
    content = reader.content(spacing)
    spacing = reader.refine(spacing)
    reader.close()
    # showim(content)

    out_path = './out/example_resaved.tif'
    tile_size = min(512, max(content.shape[:2]))
    print('writing content with shape %s to %s with tile_size %d' % (str(content.shape), out_path, tile_size))
    writer = ArrayImageWriter(tile_size=tile_size)
    writer.write_array(content, path=out_path, spacing=spacing)

    reader = AsapReader(out_path, downsample=True)
    print(reader.spacings)
    print(reader.shapes)
    showim(reader.content(reader.spacings[0]))


if __name__ == '__main__':
    _example()