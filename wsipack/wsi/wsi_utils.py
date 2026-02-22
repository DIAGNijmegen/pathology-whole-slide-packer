from PIL import Image
from scipy.ndimage import zoom

from wsipack.wsi.asap_writer import ArrayImageWriter
from wsipack.utils.asap_links import make_asap_link
from wsipack.utils.cool_utils import *
from wsipack.wsi.wsi_read import ImageReader, create_reader


def write_array_with_writer(array, writer, tile_size=512, close=True):
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
            if col >= shape[1]: continue
            tile = array[row:row+tile_size, col:col+tile_size]
            wtile = np.zeros((tile_size, tile_size, n_channels), dtype=array.dtype)
            # wtile = wtile.squeeze()
            if len(wtile.shape)!=len(tile.shape):
                wtile = wtile.squeeze()
                tile = tile.squeeze()
            wtile[:tile.shape[0],:tile.shape[1]] = tile.copy()
            # print('writing tile from row', row, 'col', col, tile.shape, 'wtile', wtile.shape)
            writer.write(tile=wtile, row=row, col=col)
            # print('done writing tile')
    if close:
        writer.close()


def write_wsi_heatmap(hm, out_path, slide_spacing, wsi, links_dir, shift, anno_path=None, overwrite=False):
    if Path(out_path).exists() and not overwrite:
        print('not overwriting %s' % str(out_path))
        return out_path

    out_dir = Path(out_path).parent
    print('writing results to %s' % Path(out_dir).absolute())
    ensure_dir_exists(out_dir)

    if str(links_dir)==str(out_dir):
        if 'likelihood_map' not in str(out_path):
            suffix = Path(out_path).suffix
            out_path = str(out_path)[:-len(suffix)]+'_likelihood_map.tif'
    else:
        ensure_dir_exists(links_dir)

    close = False
    if isinstance(wsi, (str, Path)):
        wsi = ImageReader(str(wsi))
        close = True
    slide_spacing = wsi.refine(slide_spacing)
    max_spacing = wsi.spacings[-1]
    out_spacing = slide_spacing * shift
    if out_spacing > (max_spacing*1.02):
        zoom_factor = out_spacing / max_spacing
        # zoom_factor = int(np.round(zoom_factor))
        # zoom_factor = take_closest_number([2, 4, 8, 16, 32, 64, 128], zoom_factor)
        hm = zoom(hm, zoom=zoom_factor, order=0, mode='nearest')
        # hm = upscale(hm, zoom_factor)
        print('zoomed by %d to %s' % (zoom_factor, str(hm.shape)))
        out_spacing = slide_spacing * (shift / zoom_factor)
        wsi_shape = wsi.shape(out_spacing)
        hm = hm[:wsi_shape[0], :wsi_shape[1]]

    if close:
        wsi.close()

    # if np.allclose(hm, 0): #for better visualization in asap
    #     hm[0, 0] = 1

    # cam_result = cam_result[1:, 1:]  # otherwise doesnt work in asap for some reason...
    print('writing array with spacing %f' % out_spacing)
    ArrayImageWriter().write_array(hm, out_path, out_spacing)
    if links_dir is not None:
        print('make asap links...')
        mask_path = out_path
        if str(links_dir)==str(out_dir):
            mask_path = None
        make_asap_link(wsi.path, anno_path, mask_path, links_dir, relative=True)

    # try:
    #     pyvips_dir = str(out_path.parent)+'_pyvips'
    #     ensure_dir_exists(pyvips_dir)
    #     pyvips_path = Path(pyvips_dir)/out_path.name
    #     writer = PyvipsWriter()
    #     writer.write_array(cam_result, pyvips_path, out_spacing)
    #     links_dir = str(links_dir)+'_pyvips'
    #     make_asap_link(wsi.path, anno_path, out_path, links_dir, relative=True)
    # except:
    #     print('writing pyvips failed:')
    #     print(sys.exc_info())
    return out_path

def write_salient_patches(wsi, sal_result, slide_spacing, out_dir, patch_size, shift, n_examples=9, n_cols=3, percentile=99.9,
                          read_spacing=0.5, sal_result2=None, overwrite=False):
    if n_examples==0: return

    # out_dir = str(Path(out_path).parent)+'_example_patches'
    if not 'patches' in str(out_dir):
        suffix = '_example_patches'
        if 'links' in str(out_dir):
            out_dir = str(out_dir).replace('_links',suffix)
        else:
            out_dir = str(out_dir)+suffix

    if sal_result2 is not None:
        if sal_result.shape != sal_result2.shape: raise ValueError('different sal_results shapes: %s and %s' % (str(sal_result.shape), str(sal_result2.shape)))
    ensure_dir_exists(out_dir)
    name = Path(wsi.path).stem
    # name = Path(out_path).stem
    img_out_path = Path(out_dir)/(f'{name}.jpg')

    if img_out_path.exists() and not overwrite:
        print('not overwriting %s' % str(img_out_path))
        return img_out_path

    patches = []
    names = []
    sal_strings = []

    sal_result_flat = sal_result.flatten()
    # salient_flat_ = np.argsort(-sal_result_flat)
    salient_flat = np.argsort(sal_result_flat)[::-1]
    max_coords = list(zip(*np.unravel_index(salient_flat, shape=sal_result.shape)))

    for x,y in max_coords:
        saliency = sal_result[x, y]
        xslide, yslide = int(x*shift), int(y*shift)
        try:
            patch = wsi.read(slide_spacing, xslide, yslide, patch_size, patch_size)
        except:
            print(f'failed reading salient ({saliency}) patch from {name} x={xslide}, y={yslide}, shift={shift}')
            raise
        patches.append(patch)
        sal_str = f'{saliency:.4f}'
        if sal_result2 is not None:
            sal_str+=f' ({sal_result2[x, y]:.4f})'
        # name = f'{x}_{y}_{sal_str}'
        sal_strings.append(sal_str)
        # sals.append(saliency)
        if len(patches)>=n_examples:
            break

    ### to get more diverse patches randomly select from 9x percentile
    # sal_result = deepcopy(sal_result)
    # slide_spacing = wsi.refine(slide_spacing)
    # # out_spacing = slide_spacing * shift
    #
    # for i in range(20):
    #     max_perc = np.percentile(sal_result[sal_result != 0], percentile)
    #     max_ind = sal_result >= max_perc
    #     if np.sum(max_ind)<n_examples:
    #         percentile -= 1
    #     else:
    #         break
    # max_x, max_y = np.where(max_ind)
    # sel_inds = np.arange(len(max_x))
    # np.random.shuffle(sel_inds)
    # sel_inds = sel_inds[:n_examples]
    #
    # for sel_ind in sel_inds:
    #     x, y = int(max_x[sel_ind]*shift), int(max_y[sel_ind]*shift)
    #     saliency = sal_result[max_x[sel_ind], max_y[sel_ind]]
    #     try:
    #         #### patch = wsi.read(slide_spacing, x-shift, y-shift, shift*3, shift*3)
    #         patch = wsi.read(slide_spacing, x, y, shift, shift)
    #         patches.append(patch)
    #         names.append(f'{x}_{y}_{saliency:.4f}')
    #         sals.append(saliency)
    #     except:
    #         print(f'Failed reading patch ({x}, {y}) at spacing {slide_spacing}')
    #
    # sel_sorted = np.argsort(sals)[::-1]
    #
    # patches, names, sals = lists_indexed(sel_sorted, patches, names, sals)


    n_rows = n_examples//n_cols
    # showgrid(patches, nrows_ncols=(n_examples//2, 2), figsize=(5,7.5), titles=names, show=False, save_path=img_out_path)
    # sal_strings = ['%.4f' % s for s in sals]
    showgrid(patches, rows=n_rows, cols=n_cols, figsize=(n_rows*2.5, n_cols*2.5), titles=sal_strings,
             show=False, save_path=img_out_path, wspace=0.01, hspace=0.01, title_pad=-10, title_bold=True)
    # im = Image.fromarray(patch)
    # im.save(str(img_out_path), "JPEG", quality=80, optimize=True)
    # print('saved %d most salient examples in %s' % (len(sel_inds), img_out_path))
    return img_out_path

def upscale(arr, factor):
    if factor % 2 !=0:
        print('warning! tried only for mod 2, not %d' % factor)
    nshape = [int(np.ceil(arr.shape[0]*factor)), int(np.ceil(arr.shape[1]*factor))]
    scaled = np.zeros(nshape, dtype=arr.dtype)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            scaled[i*factor:(i+1)*factor,j*factor:(j+1)*factor] = arr[i,j]
    return scaled


def find_thumbnail_level(shapes, max_pix=1024 * 1024):
    """ gets the level and shape with the largest resolution < 1500 in all dimensions,
    returns the level and the smallest shape"""
    n_levels = len(shapes)
    for i,ldim in enumerate(shapes):
        if ldim[0]*ldim[1]<=max_pix:
            return i
    #not found
    # smallest_shape = img.getLevelDimensions(n_levels-1)
    return n_levels-1
    # if smallest_shape[0]*smallest_shape[1]<5000000:
    #     return n_levels-1, smallest_shape
    # return None, None

def create_thumbnail(path, out_path, overwrite=False, format='JPEG', openslide=False, level=None):
    """ apparently, for some mrxs, turns red to blue... """
    if str(path)==str(out_path):
        raise ValueError('input=output=%s' % path)

    if Path(out_path).exists() and not overwrite:
        print('skipping, since thumbnail exists %s' % str(out_path))
        return False

    print(str(path))
    try:
        reader = create_reader(str(path), reader='openslide' if openslide else 'asap')
    except:
        print("failed opening %s: %s" % (str(path), str(sys.exc_info())))
        return False
    if level is None:
        level = find_thumbnail_level(reader.shapes)
    if level is None:
        print('skipping creating thumbnail, since didnt found sufficiently small level', reader.shapes)
        return False

    try:
        spacing = reader.spacings[level]
        thumbnail = reader.content(spacing)
        reader.close()
    except:
        print("failed loading %s: %s" % (str(path), str(sys.exc_info())))
        return False

    return write_arr_as_image(thumbnail, out_path, format=format)


def write_arr_as_image(arr, out_path, format='JPEG'):
    print(out_path)

    if arr.shape[-1]==1:
        arr = arr.squeeze()
        if 'uint' in str(arr.dtype):
            arr*=(255 // arr.max())
        else:
            print('dtype of mask not uint, but %s!' % str(arr.dtype))
        #mask
    im = Image.fromarray(arr).convert('RGB')
    im.save(str(out_path), format, quality=80)
    return True

def read_patch_from_arr(arr, row, col, height, width):
    """ simulate reading a patch form the slide, but using its content array (handles border-cases)
        assumes hwc, pads with zeros on the lower/top for left/top-border (negative row/col) and
        on right/bottom for right/bottom-border.
    """
    arr_height = arr.shape[0]; arr_width = arr.shape[1]
    if row < 0 or col < 0 or row+height > arr_height or col+width > arr_width:
        if len(arr.shape)>2:
            patch = np.zeros((height, width, arr.shape[2]), dtype=arr.dtype)
        else:
            patch = np.zeros((height, width), dtype=arr.dtype)
        left_pad = 0
        if row < 0:
            left_pad = -row
        top_pad = 0
        if col < 0:
            top_pad = -col
        read_patch = arr[row+left_pad:row+height, col+top_pad:col+width]
        patch[left_pad:left_pad+read_patch.shape[0],top_pad:top_pad+read_patch.shape[1]] = read_patch
    else:
        patch = arr[row:row+height, col:col+width]
    return patch

if __name__ == '__main__':
    # arr = np.array([[1, 2, 4],[5, 3, 6]])
    # scaled = upscale(arr, 2)
    # showims(arr, scaled)
    pass