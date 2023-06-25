from __future__ import annotations
from copy import deepcopy

from shapely.geometry import CAP_STYLE

from wsipack.utils.cool_utils import *
from wsipack.utils.df_utils import print_df
from wsipack.utils.path_utils import *
import functools
from wsipack.wsi.contour_utils import dist_to_px

print = functools.partial(print, flush=True)

from wsipack.wsi.wsd_image import ImageReader
import xml.etree.ElementTree as ET

try:
    from lxml import etree
except:
    print('lxml not installed, anno_utils might fail')

from shapely import geometry

def rgb2hex(r,g,b):
    return '#{:02x}{:02x}{:02x}'.format(r,g,b)

def print_el(el):
    print(etree.tostring(el, pretty_print=True, encoding="unicode"))

class AnnoGroup(object):
    default_color = "#FF0000"
    #<AnnotationGroups>
    #<Group Name="Tumor" PartOfGroup="None" Color="#FF0000">
    def __init__(self, name, parent='None', color=default_color):
        self.name = name
        self.parent = str(parent)
        if color is None:
            color = AnnoGroup.default_color
        if is_iterable(color):
            color = rgb2hex(*color)
        self.color = color

    @staticmethod
    def from_el(group_el):
        return AnnoGroup(name=group_el.get('Name'), parent=group_el.get('PartOfGroup'), color=group_el.get('Color'))

    def to_el(self, parent_el):
        etree.SubElement(parent_el, 'Group', Name=self.name, PartOfGroup=self.parent, Color=self.color)

    def __repr__(self):
        return 'Group name=%s, parent=%s, color=%s' % (self.name, self.parent, self.color)

def _get_single_child(el, name):
    """ raises an error if there is no single child of that name"""
    children = _get_children(el, name)
    if len(children)!=1:
        raise ValueError('%d!=1 children of name %s in %s' % (len(children), name, str(el)))
    return children[0]

def _get_children(el, name):
    children = []
    for child in el.iter():
        if child.tag==name:
            children.append(child)
    return children

class Anno(object):
    #<Annotation Name="Annotation 0" Type="Polygon" PartOfGroup="Tumor" Color="#F4FA58">
    #<Coordinates>
    #<Coordinate Order="0" X="36483.8008" Y="36816.1992" />

    def __init__(self, name, type, group, coords, color='#000000'):
        self.name = name
        self.type = type
        self.group = str(group)
        if color is None:
            color='#000000'
        if is_iterable(color):
            color = rgb2hex(*color)
        self.color = color
        self.coords = coords

    def check(self):
        if self.type == 'Polygon' and len(self.coords)==1:
            print('Polygon anno %s with only 1 coord!' % self.name)

    @staticmethod
    def from_el(anno_el):
        coords_el = _get_single_child(anno_el, 'Coordinates')
        coords = {}
        for coord_el in coords_el:
            coords[int(coord_el.get('Order'))] = (int(round(float(coord_el.get('X').replace(',','.')))),
                                                  int(round((float(coord_el.get('Y').replace(',','.'))))))

        coords = dict_sorted_values_by_key(coords)
        coords = np.array(coords)
        return Anno(name=anno_el.get('Name'), group=anno_el.get('PartOfGroup'),
                                         type=anno_el.get('Type'), color=anno_el.get('Color'), coords=coords)

    def to_el(self, parent_el):
        anno_el = etree.SubElement(parent_el, 'Annotation', Name=self.name, Type=self.type, Color=self.color,
                                   PartOfGroup=str(self.group))
        coords_el = etree.SubElement(anno_el, 'Coordinates')
        for i,coord in enumerate(self.coords):
            etree.SubElement(coords_el, 'Coordinate', Order=str(i), X=str(coord[0]), Y=str(coord[1]))
        return anno_el

    def shift(self, w=0, h=0):
        self.coords[:,0]+=w
        self.coords[:,1]+=h

    def scale(self, factor=1, round=False):
        self.coords=self.coords*factor
        if round:
            self.coords = np.around(self.coords)

    def __repr__(self):
        return 'Anno name=%s, type=%s, group=%s, color=%s' %\
               (self.name, self.type, self.group, self.color)

    def __eq__(self, other):
        return str(self)==str(other) and self.coords==other.coords

    @staticmethod
    def create_rectangle_from_xywh(name, group, x, y, w, h, color='#000000'):
        coords = np.array([[x,y],[x+w,y],[x+w,y+h],[x, y+h]])
        return Anno(name=name, group=group,
                               type='Rectangle', coords=coords, color=color)


class AsapAnno(object):
    def __init__(self, path=None):
        self.annos = []
        self.groups = []
        if path is None:
            self.path = None
        else:
            self.path = Path(path)

        if path is not None:
            tree = etree.parse(str(self.path))
            root = tree.getroot()
            group_els = tree.findall('.//Group')
            for gr in group_els:
                # print(etree.tostring(gr, pretty_print=True, encoding="unicode"))
                group = AnnoGroup.from_el(gr)
                self.groups.append(group)

            anno_els = list(root.iter('Annotation'))
            for anno_el in anno_els:
                anno = Anno.from_el(anno_el)
                self.annos.append(anno)

            # for anno in root:
            #     print(child.tag, child.attrib)
            # asap_anno = AsapAnno()

    def check(self):
        for anno in tqdm(self.annos):
            anno.check()

    def add_group(self, name, color="#FF0000"):
        group = AnnoGroup(name, color=color)
        self.groups.append(group)

    def get_group(self, name):
        for group in self.groups:
            if group.name == name:
                return group
        return None

    def get_annos_overlapping_with(self, anno:Anno):
        if len(anno.coords)==1:
            raise ValueError('todo implement point overlap')
        found = []
        roip = geometry.Polygon(anno.coords)

        for a in self.annos:
            coords = a.coords
            if len(coords)==1:
                point = geometry.Point(*coords)
                if roip.contains(point):
                    found.append(a)
            else:
                ap = geometry.Polygon(coords)
                if roip.intersects(ap):
                    found.append(a)
        return found

    def get_annos_in_group(self, group):
        annos = [a for a in self.annos if a.group==group]
        return annos

    def scale(self, factor=1, round=False):
        """ scales the coordinates by the factor"""
        for anno in self.annos:
            anno.scale(factor, round=round)

    def print(self):
        tree = self._create_tree()
        print_el(tree)

    def _create_tree(self):

        page = etree.Element('ASAP_Annotations')

        anno_els = etree.SubElement(page, 'Annotations')
        for a in self.annos:
            a.to_el(anno_els)
        group_els = etree.SubElement(page, 'AnnotationGroups')
        for group in self.groups:
            group.to_el(group_els)

        return page

    def delete_groups_except(self, group_names):
        if group_names is None: return
        to_delete = [group.name for group in self.groups if group.name not in group_names]
        self.delete_groups(to_delete)

    def delete_group(self, name):
        if name is None: return
        return self.delete_groups([name])

    def delete_groups(self, group_names):
        if group_names is None: return
        self.annos = [a for a in self.annos if a.group not in group_names]
        self.groups = [g for g in self.groups if g.name not in group_names]

    def remove_annos(self, anno_items, by_ref=False):
        if not is_iterable(anno_items):
            anno_items = [anno_items]

        if by_ref:
            inds = [self.annos.index(item) for item in anno_items]
        else:
            inds = []
            for i,anno in enumerate(self.annos):
                for a in anno_items:
                    if anno.name==a.name:
                        inds.append(i)
                        break

        inds = sorted(inds, reverse=True)
        for ind in inds:
            self.annos.pop(ind)

    def group_names(self):
        return [g.name for g in self.groups]

    def rename_groups(self, rename_map):
        for old_name, new_name in rename_map.items():
            for group in self.groups:
                if group.name == old_name:
                    group.name = new_name
            for anno in self.annos:
                if anno.group == old_name:
                    anno.group = new_name
        self._remove_duplicate_groups()

    def add_annotation_prefix_suffix(self, prefix='', suffix=''):
        for anno in self.annos:
            if prefix is not None and len(prefix)>0:
                anno.name = prefix + anno.name
            if suffix is not None and len(suffix)>0:
                anno.name += suffix

    def update_group_colors(self, group_color_map):
        for group_name, color in group_color_map.items():
            for group in self.groups:
                if group.name==group_name:
                    group.color = color

    def _remove_duplicate_groups(self):
        known = []
        to_delete_inds = []
        for i,group in enumerate(self.groups):
            if group.name in known:
                to_delete_inds.append(i)
            else:
                known.append(group.name)

        for i in to_delete_inds[::-1]:
            del self.groups[i]

    def save(self, path, overwrite=False):
        if path is None:
            raise ValueError('saving not possible, path is None!')
        if Path(path).exists() and not overwrite:
            raise ValueError('not overwriting %s' % str(path))
        Path(path).parent.mkdir(exist_ok=True)
        page = self._create_tree()
        doc = etree.ElementTree(page)
        ensure_dir_exists(Path(path).parent)
        out_file = open(str(path), 'wb')
        doc.write(out_file, xml_declaration=True, pretty_print=True, encoding="utf-8")

    def convert_to_points(self, groups=None, types=['Rectangle', 'Polygon'], replace=False, name_suffix='_dot'):
        """ converts the annotations in groups (all if None) if of type (default: rectangles and polygons) to dots.
         if replace, replaces the original annotations, otherwise adds them with the suffix """
        annos = [anno for anno in self.annos if anno.type in types]
        for anno in annos:
            if groups is not None:
                if anno.group not in groups:
                    continue
            poly = geometry.Polygon(anno.coords)
            center = poly.centroid
            coords = np.array(list(center.coords))
            if replace:
                anno.type = 'Dot'
                anno.coords = coords
            else:
                self.annos.append(Anno(name=anno.name+name_suffix, group=anno.group,
                                       type='Dot', coords=coords, color=anno.color))

    def add_point(self, name, group, coords, color="#FF0000"):
        self.annos.append(Anno(name=name, group=group,
                               type='Dot', coords=coords, color=color))

    def convert_points_to_boxes(self, distance, groups=None, replace=False, name_suffix='_box'):
        annos = [anno for anno in self.annos if anno.type=='Dot']
        for anno in annos:
            if groups is not None:
                if anno.group not in groups:
                    continue
            #to do: refactor all conversions - only these lines are conversion-specific
            dot = anno.coords.flatten()
            poly = geometry.Point(dot)
            box = poly.buffer(distance, cap_style=CAP_STYLE.square)
            coords = np.array(list(box.exterior.coords))[:4,:]
            if replace:
                anno.type = 'Rectangle'
                anno.coords = coords
            else:
                self.annos.append(Anno(name=anno.name+name_suffix, group=anno.group,
                                       type='Rectangle', coords=coords, color=anno.color))

    def add_rectangle(self, name, group, coords, color='#000000'):
        if group not in self.group_names():
            raise ValueError('group %s not in groups %s' % (group, str(self.group_names())))
        if len(coords)!=4:
            raise ValueError('Rectangles should have 4 coordinates, not %d' % len(coords))
        self.annos.append(Anno(name=name, group=group,
                               type='Rectangle', coords=coords, color=color))

    def add_polygon(self, name, group, coords, color='#000000'):
        if str(group)!='None' and group not in self.group_names():
            raise ValueError('group %s not in groups %s' % (group, str(self.group_names())))
        self.annos.append(Anno(name=name, group=group,
                               type='Polygon', coords=coords, color=color))

    def convert_points_to_boxes_um(self, distance, wsi_path, **kwargs):
        reader = ImageReader(wsi_path)
        spacing = reader.spacings[0]
        reader.close()
        px = dist_to_px(distance, spacing)
        return self.convert_points_to_boxes(px, **kwargs)

    def get_color(self, anno:Anno):
        if str(anno.group) == 'None':
            return anno.color
        group = self.get_group(anno.group)
        return group.color

    def get_anno_by_name(self, name):
        """ returns the first anno by this name """
        for ann in self.annos:
            if ann.name == name:
                return ann
        return None

    def shift_annos(self, anno_items, x, y):
        if not is_iterable(anno_items):
            anno_items = [anno_items]
        for anno_item in anno_items:
            for index in range(len(anno_item.coords)):
                coordinate = anno_item.coords[index]
                anno_item.coords[index] = (int(round(coordinate[0] + x)), int(round(coordinate[1] + y)))

    def scale_annos(self, anno_items, factor):
        """ anno_items: anno-dictionary from Annotation containing e.g. coorindates, name,... """
        if not is_iterable(anno_items):
            anno_items = [anno_items]
        for anno_item in anno_items:
            for index in range(len(anno_item.coords)):
                coordinate = anno_item.coords[index]
                anno_item.coords[index] = (int(round(coordinate[0]*factor)), int(round(coordinate[1]*factor)))

    @classmethod
    def merge(cls, a1:AsapAnno, a2:AsapAnno, anno_prefix1='', anno_prefix2='', anno_suffix1='', anno_suffix2=' new',
              group1_color_map={}, group1_rename_map={}, group2_color_map={}, group2_rename_map={}):
        a1 = deepcopy(a1)
        a1.rename_groups(group1_rename_map)
        a1.update_group_colors(group1_color_map)
        a1.add_annotation_prefix_suffix(anno_prefix1, anno_suffix1)
        a2 = deepcopy(a2)
        a2.rename_groups(group2_rename_map)
        a2.update_group_colors(group2_color_map)
        a2.add_annotation_prefix_suffix(anno_prefix2, anno_suffix2)
        for group in a2.groups:
            if a1.get_group(group.name) is None:
                a1.groups.append(group)
        for anno in a2.annos:
            a1.annos.append(anno)

        return a1

    def infos(self):
        entries = []
        for anno in self.annos:
            entries.append(dict(type=anno.type, name=anno.name, n_coords=len(anno.coords)))
        df = pd.DataFrame(entries)
        print_df(df.sort_values('n_coords'))
        return df

    def check(self):
        for anno in self.annos:
            anno.check()

def anno_same_coords(a, b):
    acoords = a['coordinates']
    bcoords = b['coordinates']
    same = acoords == bcoords
    return same

def save_anno_rounded(anno_path, out_path):
    tree = ET.parse(str(anno_path))
    root = tree.getroot()
    coordinates = root.findall('.//Coordinate')
    for coord in coordinates:
        x = float(coord.get('X').replace(',', '.'))
        y = float(coord.get('Y').replace(',', '.'))
        xnew = int(np.round(x))
        ynew = int(np.round(y))
        coord.set('X', str(xnew))
        coord.set('Y', str(ynew))
        print('x', x, 'xnew', xnew, 'y',y, 'ynew', ynew)
    tree.write(out_path)
    print('%s saved' % out_path)

def scale_anno_coords(anno_path, out_path, factor=None, wsi_path=None, spacing=None):
    """ scales the annotations to be valid for the given spacing; takes the xml-spacing from the wsi_path """
    if factor is None and spacing is None:
        raise ValueError('requires either factor or spacing')
    if spacing is not None:
        if wsi_path is None:
            raise ValueError('requires wsi_path if spacing is to be used')
        if not Path(wsi_path).exists():
            raise ValueError('wsi %s doesnt exist' % str(wsi_path))
    reader = ImageReader(wsi_path)
    spacing = reader.refine(spacing)
    factor = reader.spacings[0]/spacing
    reader.close()
    if not (factor*100)//2:
        print('warn: factor %.3f for %s not //2' % (factor, Path(wsi_path).stem))

    ensure_dir_exists(Path(out_path).parent)
    anno_path = Path(anno_path)
    tree = ET.parse(str(anno_path))
    root = tree.getroot()
    coordinates = root.findall('.//Coordinate')
    for coord in coordinates:
        x = float(coord.get('X').replace(',', '.'))
        y = float(coord.get('Y').replace(',', '.'))
        xnew = int(round(x*factor))
        ynew = int(round(y*factor))
        coord.set('X', str(xnew))
        coord.set('Y', str(ynew))
    tree.write(str(out_path))
    print('%s saved' % out_path)

def scale_anno_coords_all(anno_dir, wsi_dir=None, factor=None, spacing=None, out_dir=None, overwrite=False):
    if factor is None and spacing is None:
        raise ValueError('requires either factor or spacing')
    anno_pathes = PathUtils.list_pathes(anno_dir, ending='.xml')
    if wsi_dir is not None:
        wsi_pathes = PathUtils.list_pathes(wsi_dir, not_containing=['.xml','.db','csv', '.json', '.yaml', '.txt'])
        wsi_pathes = get_corresponding_pathes_all(anno_pathes, wsi_pathes, take_shortest=True)
    if out_dir is None:
        out_dir = str(anno_dir)+f'_x{factor}'
    out_dir = Path(out_dir)
    ensure_dir_exists(out_dir)
    for i,anno_path in enumerate(anno_pathes):
        out_path = out_dir/anno_path.name
        if can_open_file(out_path, ls_parent=True) and not overwrite:
            print('skipping existing %s' % str(out_path))
            continue
        wsi_path = None
        if wsi_dir is not None:
            wsi_path = wsi_pathes[i]
        scale_anno_coords(anno_path, out_path, factor=factor, wsi_path=wsi_path, spacing=spacing)

    print('Done converting %d annos' % len(anno_pathes))

def shift_anno_coords(anno_path, out_path, xy):
    ensure_dir_exists(Path(out_path).parent)
    anno_path = Path(anno_path)
    tree = ET.parse(str(anno_path))
    root = tree.getroot()
    coordinates = root.findall('.//Coordinate')
    for coord in coordinates:
        x = float(coord.get('X').replace(',', '.'))
        y = float(coord.get('Y').replace(',', '.'))
        xnew = int(np.round(x)) - int(np.round(xy[0]))
        ynew = int(np.round(y)) - int(np.round(xy[1]))
        coord.set('X', str(xnew))
        coord.set('Y', str(ynew))
    tree.write(str(out_path))
    print('%s saved' % out_path)


def unshift_anno_coords(anno_path, out_dir, suffix_type='__', overwrite=False):

    anno_path = Path(anno_path)
    if suffix_type=='__':
        dparts = anno_path.stem.split('__')
        name = '__'.join(dparts[:-1])
        parts = dparts[-1].split('_')
        first_ind = 0
    else:
        if suffix_type=='tl_br': #top_left_bottom_right
            first_ind = -4
        elif suffix_type == 'tl': #top left only
            first_ind = -2
        else: raise ValueError('unknown annotation crop naming suffix type %s' % suffix_type)
        parts = anno_path.stem.split('_')
        parts = [p for p in parts if len(p)>0] #skimp emtpy lines
        name = '_'.join(parts[:-first_ind])

    coord = np.array([int(parts[first_ind]), int(parts[first_ind+1])])
    out_path = Path(out_dir)/(name+'.xml')

    if out_path.exists() and not overwrite:
        print('skipping existing %s' % out_path)
    else:
        shift_anno_coords(anno_path, out_path, -coord)


def unshift_anno_coords_all(anno_dir, out_dir, suffix_type='__', **kwargs):
    """ unshift annos using the coordinate in the name, eg. slide_12_150 """
    annos = PathUtils.list_pathes(anno_dir, ending='xml')
    print('unshift %d annos in %s' % (len(annos), out_dir))
    for anno in tqdm(annos):
        unshift_anno_coords(anno, out_dir, suffix_type=suffix_type, **kwargs)
    print('Done!')

def rename_anno(anno_path, out_path, name_map, strip=True, overwrite=False):
    if Path(out_path).exists() and not overwrite:
        print('skipping already existing %s' % str(out_path))
        return
    tree = ET.parse(str(anno_path))
    root = tree.getroot()
    annotations_root = root.find('Annotations')
    groups_root = root.find('AnnotationGroups')

    annotations = annotations_root.findall('Annotation')
    count = 0; group_count = 0
    for anno in annotations:
        group_name = anno.get('PartOfGroup')
        if strip:
            group_name_new = group_name.strip()
        else:
            group_name_new = group_name
        group_name_new = name_map.get(group_name_new, group_name_new)
        if group_name_new!=group_name:
            anno.set('PartOfGroup', group_name_new)
            count+=1

    for group in groups_root.findall('Group'):
        group_name = group.get('Name')
        if strip:
            group_name_new = group_name.strip()
        else:
            group_name_new = group_name
        group_name_new = name_map.get(group_name_new, group_name_new)
        if group_name_new!=group_name:
            group.set('Name', group_name_new)
            group_count+=1

    tree.write(str(out_path))
    print('%d group and %d anno changes saved in: %s' % (group_count, count, out_path))

def rename_annos_in(anno_dir, out_dir, name_map, **kwargs):
    ensure_dir_exists(out_dir)
    anno_pathes = pathes_in(anno_dir, ending='xml')
    print('renaming %d annotations' % len(anno_pathes))
    for ap in tqdm(anno_pathes):
        rename_anno(ap, Path(out_dir) / ap.name, name_map, **kwargs)
    print('Done!')

def auto_add_groups(anno):
    groups = anno.groups
    if groups is None: groups = {}
    for a in anno.annotations:
        group = a['group']
        if str(group)!='None':
            group = groups[group]
            group['color'] = a['color']

