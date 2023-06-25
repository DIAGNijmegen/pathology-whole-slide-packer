from wsipack.utils.cool_utils import *
from wsipack.utils.path_utils import *

def make_link(src, link, relative=False, ssh_node=None, verbose=False, dry_run=False, exist_ok=False):
    if exist_ok and Path(link).exists():
        return
    if relative:
        src = os.path.relpath(str(src), Path(link).parent)
    if ssh_node is None or len(ssh_node)==0:
        os.symlink(src, str(link))
    else:
        src_str = str(src)
        dst_str = str(link)
        cmd = f"ssh {ssh_node} ln -s {src_str} {dst_str}"
        if dry_run:
            print('dry_run:',cmd)
            output = None
        else:
            output = subprocess.getoutput(cmd)
        if verbose:
            print(output)
        return output

def make_asap_links(img_dir, anno_dir, mask_dir, links_dir, only_with_mask=False, only_with_anno=False, mask_ending='tif',
                    anno_ending='xml', relative=True, take_shortest=False, ssh_node=None, open_file_browser=False,
                    verbose=False, same_name=False, filter_names=None, mask_containing=None, **kwargs):
    """ only_with_anno: makes links only if there is an annotation """
    if anno_dir is None and mask_dir is None:
        raise ValueError('either annotations or masks must be given!')

    print('reading pathes in %s' % str(img_dir))
    img_pathes = PathUtils.list_pathes(img_dir, type='file', **kwargs)
    if filter_names is not None:
        n_old = len(img_pathes)
        img_pathes = [p for p in img_pathes if p.stem in filter_names]
        print('%d/%d pathes, rest filtered' % (len(img_pathes), n_old))
    else:
        print('found %d slides' % len(img_pathes))
    # else:
    #     print('found %d images in %s' % (len(img_pathes), str(img_dir)))

    if anno_dir is None:
        all_anno_pathes = []
    else:
        all_anno_pathes = _get_anno_pathes_from_anno_dir(anno_dir, anno_ending=anno_ending)

    all_mask_pathes = []
    if mask_dir is not None:
        if is_iterable(mask_dir) and Path(mask_dir[0]).exists():
            all_mask_pathes = mask_dir
        else:
            all_mask_pathes = PathUtils.list_pathes(mask_dir, ending=mask_ending, containing_or=mask_containing)
            print('found %d masks in %s' % (len(all_mask_pathes), str(mask_dir)))

    anno_pathes = []; mask_pathes = []
    for img_path in tqdm(img_pathes):
        name = img_path.stem
        anno_path = None
        if anno_dir is not None:
            anno_path = get_path_named_like(name, all_anno_pathes, take_shortest=take_shortest, same_name=same_name)
        anno_pathes.append(anno_path)
        mask_path = None
        if mask_dir is not None:
            mask_path = get_path_named_like(name, all_mask_pathes, take_shortest=take_shortest)
        mask_pathes.append(mask_path)

    return _make_asap_links_pathes(img_pathes, anno_pathes, mask_pathes, links_dir, only_with_anno=only_with_anno,
                                   only_with_mask=only_with_mask, relative=relative, ssh_node=ssh_node,
                                   open_file_browser=open_file_browser, verbose=verbose)



def make_asap_link(img_path, anno_path, mask_path, links_dir, relative=True, verbose=False, ssh_node=None, **kwargs):
    """ makes links so that ASAP opens the mask or annotation automatically when opening the slide
    relative: if True makes relative links, can be a string containing any of img, mask or anno,
    e.g. 'img,anno' will create only relative links for the slide and the annotation.
    """
    img_path = Path(img_path).absolute()
    if mask_path is not None:
        mask_path = Path(mask_path).absolute()
    if anno_path is not None:
        anno_path = Path(anno_path).absolute()

    links_dir = Path(str(links_dir)).absolute()
    links_path = links_dir/img_path.name
    if path_exists(links_path):
        if verbose: print('skipping %s, link exists' % str(links_path))
        # return False
    else:
        ensure_dir_exists(links_dir)
        make_link(img_path, links_path, relative=(relative==True or 'img' in str(relative)), ssh_node=ssh_node, **kwargs)
        # if relative:
        #     img_path = Path(os.path.relpath(str(img_path), str(links_dir)))
        # else:
        #     os.symlink(img_path, links_dir/img_path.name)
    img_name = img_path.stem

    link_created = False
    #if there is an directory for the image data, link that too
    img_dir = img_path.parent/img_name
    links_path = links_dir / img_name
    if path_exists(img_dir) and not links_path.exists():
        make_link(img_dir, links_path, relative=(relative==True or 'img' in str(relative)),
                  ssh_node=ssh_node, verbose=verbose, **kwargs)
        # os.symlink(img_dir, links_dir/img_name)
        link_created = True

    #### Anno ###
    links_path = links_dir/(img_name+'.xml')
    if anno_path is not None and not links_path.exists():
        make_link(anno_path, links_path, relative=(relative==True or 'anno' in str(relative)),
                  ssh_node=ssh_node, verbose=verbose, **kwargs)
        # os.symlink(anno_path, links_dir/(img_name+'.xml'))
        link_created = True

    ### Mask ####
    if mask_path is not None:
        mask_link_name = img_name + '_likelihood_map' + mask_path.suffix
        links_path = links_dir/mask_link_name
        if not links_path.exists():
            make_link(mask_path, links_path, relative=(relative==True or 'mask' in str(relative)),
                      ssh_node=ssh_node, verbose=verbose, **kwargs)
            # os.symlink(mask_path, links_dir/mask_link_name)
            link_created = True
    return link_created


def _make_asap_links_pathes(img_pathes, anno_pathes, mask_pathes, links_dir, only_with_anno=False, only_with_mask=False,
                            open_file_browser=True, verbose=False, ssh_node=None, **kwargs):
    info_string=''
    if ssh_node is not None: info_string+=' ssh_node %s' % str(ssh_node)
    print('making links for %d files%s...' % (len(img_pathes), info_string))

    counter = 0
    for i,img_path in enumerate(tqdm(img_pathes)):
        anno_path = None
        if anno_pathes is not None:
            anno_path = anno_pathes[i]
        if anno_path is None:
            if verbose: print('No annotation for %s' % img_path)
            if only_with_anno:
                continue
        mask_path = None
        if mask_pathes is not None:
            mask_path = mask_pathes[i]
        if mask_path is None and only_with_mask:
            if verbose: print('No mask for %s, skipping...' % img_path)
            continue
        if mask_path is None and anno_path is None:
            if verbose: print('skipping %s without mas and anno' % str(img_path))
            continue
        else:
            if counter==0: ensure_dir_exists(links_dir)
            counter+= make_asap_link(img_path, anno_path, mask_path, links_dir, ssh_node=ssh_node, verbose=verbose, **kwargs)
    print('%d/%d links saved in %s' % (counter, len(img_pathes), links_dir))
    if open_file_browser:
        os.system('nautilus %s' % links_dir)


def _get_anno_pathes_from_anno_dir(anno_dir, anno_ending='xml'):
    """ supports a list of anno_dirs, the first more important then the last """
    if is_string(anno_dir) and ',' in anno_dir:
        anno_dir = anno_dir.split(',')

    if is_iterable(anno_dir):
        counter = 0
        all_anno_names = []
        all_anno_pathes = []
        for adi in anno_dir:
            anno_pathes_i = pathes_in(adi, recursive=False, ending=anno_ending, sort=True)
            counter += len(anno_pathes_i)
            for api in anno_pathes_i:
                if api.stem not in all_anno_names:
                    all_anno_pathes.append(api)
                    all_anno_names.append(api.stem)
        print('found overall unique %d/%d annotations from %d directories' % (
        len(all_anno_pathes), counter, len(anno_dir)))
    else:
        all_anno_pathes = pathes_in(anno_dir, recursive=False, ending=anno_ending, sort=True)
    return all_anno_pathes