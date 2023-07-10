from pathlib import Path

import itertools

import functools

import numpy as np
import pandas as pd

from wsipack.utils.cool_utils import is_string, write_lines, is_iterable, mkdir, is_list_or_tuple
from wsipack.utils.sparse_utils import entropy

import builtins as __builtin__


def print_df(df, n=None, **kwargs):
    print(df_text(df, n=n, **kwargs))

def printdf(df, n=None, **kwargs):
    return print_df(df, n=n, **kwargs)

def df_getna(df):
    """ returns df with rows where at least one value is na """
    df = df[df.isna().any(axis=1)]
    return df

def df_remove_ending(dfcol):
    """ removes last .ending, e.g. x.h5 -> x """
    return dfcol.str.split('.').str[:-1].str.join('.') #remove ending


def df_text(df, n=None, tablefmt='psql', **kwargs):
    if n is not None:
        df = df.head(n=n)
    from tabulate import tabulate
    return tabulate(df, headers='keys', tablefmt=tablefmt, **kwargs)

def df_textwrap(df, cols, width):
    import textwrap
    rows = df.to_dict('records')
    for row in rows:
        for col in cols:
            text = row[col]
            row[col] = '\n'.join(textwrap.wrap(text, width=width))
    df = pd.DataFrame(rows)
    return df

def df_text_save(df, out_path, **kwargs):
    text = df_text(df, **kwargs)
    with open(str(out_path), 'w') as out:
        out.write(text)

def df_strip_columns(df):
    columns={str(col):str(col).strip() for col in df.columns}
    columns = {c:cs for c,cs in columns.items() if len(c)!=len(cs)}
    if len(columns)>0:
        df.rename(columns=columns, inplace=True)
    return df

def df_autorename_columns(df):
    df_strip_columns(df)
    rename = {}
    for col in df.columns:
        name = col.replace('&','').replace('#','').lower().strip().replace(' ','_')
        if name in rename.values(): raise ValueError('duplicate column name %s for %s' % (name, col))
        rename[str(col)] = name
    df.rename(columns=rename, inplace=True)
    return df

def df_dropna(df, col, verbose=False, inplace=False, **kwargs):
    n_before = len(df)
    if not is_iterable(col):
        col = [col]
    result = df.dropna(subset=col, inplace=inplace, **kwargs)
    if inplace:
        n_after = len(df)
    else:
        result = result.copy()
        n_after = len(result)
    if verbose:
        print('%d/%d %s not na' % (n_after, n_before, str(col)))
    return result

def df_count(df, *cols, verbose=False):
    entries = []
    for col in cols:
        uniques = df[col].sort_values().unique()
        for uq in uniques:
            if isinstance(uq,float) and np.isnan(uq):
                nu = df[col].isna().sum() #equals doesnt work for nans
            else:
                nu = (df[col]==uq).sum()
            entries.append(dict(col=col, val=uq, count=nu))
    dfr = pd.DataFrame(entries)
    if verbose:
        print_df(dfr)
    return dfr

def df_stats(df, *cols, verbose=True):
    entries = []
    for col in cols:
        entry = dict(var=col)
        entry['mean'] = df[col].mean()
        entry['min'] = df[col].min()
        entry['max'] = df[col].max()
        entry['std'] = df[col].std()
        entries.append(entry)
    df = pd.DataFrame(entries)
    if verbose:
        print_df(df)
    return df

def df_columns_starting_with(df, stri, not_starting=None):
    cols = []
    for c in df.columns:
        if str(c).startswith(stri) and (not_starting is None or not(str(c).startswith(not_starting))):
            cols.append(str(c))
    return sorted(cols)

def df_columns_move(df, cols):
    cols = [c for c in cols if c in df.columns]
    rest_cols = [str(c) for c in df.columns if str(c) not in cols]
    df = df[cols+rest_cols]
    return df

def df_to_trac_format(df, float_formatter='%.4f', float_formatter_small='%.1e', index_name=None, mark_best=None, mark_significant=False,
                      mark_bold=[], print_lines=False, p_col='p', column_formatter={}):
    """ returns lines ||col||..."""
    #todo: index=True
    lines = []
    if mark_best is not None:
        ascending = 'err' in mark_best.lower() or 'loss' in mark_best.lower()
        df = df.sort_values(mark_best, ascending=ascending)
        best_ind = df.index[0]
    if mark_significant:
        df = df.sort_values(p_col, ascending=True)

    df = df.replace(np.nan, '-', regex=True)
    columns = df.columns
    stri = ''
    if index_name is not None:
        if not is_string(index_name):
            index_name = df.index.name
        stri=f"||'''{index_name}'''"
    for col in columns:
        stri+=f"||'''{str(col)}'''"
    lines.append(stri)
    for ind, row in df.iterrows():
        stri = ''
        if index_name is not None:
            stri = '||'+ind
        is_significant = False
        if mark_significant:
            is_significant = row[p_col] <= 0.05
        for col in columns:
            val_str = str(row[col])
            if isinstance(row[col], (float)):
                val_str = float_formatter % row[col]
                e_str = float_formatter_small % row[col]
                if col in column_formatter:
                    val_str = column_formatter[col] % row[col]
                if float(val_str)==0 and float(e_str)!=0:
                    val_str = e_str

            if (mark_best is not None and ind == best_ind) or is_significant:
                val_str = '**'+val_str+'**'
            stri+='||'+val_str
        lines.append(stri)

    for i in range(len(lines)):
        for mbold in mark_bold:
            lines[i] = lines[i].replace(mbold, f'**{mbold}**')

    if print_lines:
        print('\n'.join(lines))
    return lines


def df_to_trac(df, path, index_name=None, **kwargs):
    lines = df_to_trac_format(df, index_name=index_name, **kwargs)
    write_lines(path, lines)

def df_two_cols_dict(df, key, val):
    return dict(zip(df[key].values,df[val].values))

def df_object_col(df, col):
    return 'obj' in str(df.dtypes[col])

def df_merge(dfl, dfr, left, right=None, check=None, silent=False, left_title='x', right_title='y', **kwargs):
    nl = len(dfl)
    nr = len(dfr)
    if right is None:
        right = left
    df = pd.merge(dfl, dfr, left_on=left, right_on=right, suffixes=('_'+str(left_title), '_'+str(right_title)), **kwargs)
    if check in ['left','both'] and len(df)!=nl:
        print_df(dfl.head(2))
        print_df(dfr.head(2))
        raise ValueError('After merge of %d entries with %d entries only %d' % ((nl, nr, len(df))))
    if check in ['right','both'] and len(df)!=nr:
        print_df(dfl.head(2))
        print_df(dfr.head(2))
        raise ValueError('After merge of %d entries with %d entries only %d' % ((nl, nr, len(df))))
    # dfl_surp = dfl[~dfl[left].isin(df[left])]
    # dfr_surp = dfr[~dfr[right].isin(df[right])]
    if not silent: print('%d %s, %d %s, %d merged' % (len(dfl), str(left_title),
                                                      len(dfr), str(right_title), len(df)))

    return df

def df_delete_cols(df, cols):
    if not is_iterable(cols):
        cols = cols.split(',')
        cols = [c.strip() for c in cols]
    for col in cols:
        if col in df:
            del df[col]

def df_merge_check(dfl, dfr, left, right=None, left_title=None, right_title=None, verbose=False, silent=False,
                   n=10, **kwargs):
    if silent: verbose = False
    if right is None:
        right = left
    suffixes=kwargs.pop('suffixes',("_x", "_y"))
    # if left_title is not None and right_title is not None:
    #     suffixes = ('_'+left_title, '_'+right_title)
    dfm = pd.merge(dfl, dfr, left_on=left, right_on=right, suffixes=suffixes, **kwargs)
    dfl_surp = dfl[~dfl[left].isin(dfm[left])]
    merged_right = right if right in dfm else right+suffixes[1]
    dfr_surp = dfr[~dfr[right].isin(dfm[merged_right])]
    if not silent: print('%d %s, %d %s, %d merged' % (len(dfl), left if left_title is None else left_title,
                                       len(dfr), right if right_title is None else right_title, len(dfm)))
    if len(dfl_surp)>0:
        if left_title is None: left_title = left
        if not silent: print('%d only in %s' % (len(dfl_surp), left_title))
        if verbose:
            print_df(dfl_surp.head(n))
    if len(dfr_surp)>0:
        if right_title is None: right_title = right
        if not silent: print('%d only in %s' % (len(dfr_surp), right_title))
        if verbose:
            print_df(dfr_surp.head(n))
    return dfm, dfl_surp, dfr_surp

def df_concat(*dfs, **kwargs):
    return pd.concat(dfs, ignore_index=True, **kwargs)

def df_check_group_ids(df, group_cols, verbose=False, n=10):
    """ assumes df contains groups with multiple group ids and checks for each combination of group ids
        that there is only one other group id (e.g. use for sanity-checking pa-tnumbers and studynumbers wenn pseudonymizing)
    """
    pairs = list(itertools.permutations(group_cols, 2))
    print('checking group combinations', pairs)
    all_duplicates = []
    for g1, g2 in pairs:
        dfgcount = df.groupby(g1)[g2].nunique()
        dfbad = dfgcount[dfgcount>1]
        dfg_dupls = df[df[g1].isin(list(dfbad.index.values))].sort_values(g1)
        all_duplicates.append(dfg_dupls)
        print('%d/%d %s duplicates in %s groups, %d %s unique, %d %s unique' %\
              (len(dfg_dupls), len(df), g2, g1, dfg_dupls[g1].nunique(), g1, dfg_dupls[g2].nunique(), g2))
        if verbose and len(dfg_dupls)>0:
            print_df(dfg_dupls.sort_values(g1).head(n))
    return all_duplicates

def df_duplicates_check(df, col, verbose=False, n=None, raise_error=False):
    df = df[~df[col].isna()]
    df_dupl = df[df[col].duplicated(keep=False)]
    if verbose:
        # print('checking %d entries for %s-duplicates' % (len(df), col))
        if len(df_dupl)>0:
            if not raise_error:
                print('%d/%d duplicates:' % (len(df_dupl),len(df)))
            print_df(df_dupl, n=n)
            if raise_error:
                raise ValueError('%d/%d duplicates:' % (len(df_dupl),len(df)))
        else:
            pass
            # print('no duplicates in %s' % col)
    return df_dupl

def df_cols_to_lowercase(df):
    rename_map = {str(col):str(col).lower() for col in df.columns}
    df.rename(columns=rename_map, inplace=True)
    return df

def df_to_excel(df, path, sheet_name='Sheet1', index=False, max_width=60, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise ValueError('%s already exists' % str(path))
    mkdir(path.parent)
    writer = pd.ExcelWriter(str(path), engine='openpyxl')
    df.to_excel(writer, sheet_name=sheet_name, index=index)

    for column in df:
        try:
            column_length = max(df[column].astype(str).map(len).max(), len(column))
            column_length = min(column_length, max_width)
            col_idx = df.columns.get_loc(column)
            writer.sheets[sheet_name].set_column(col_idx, col_idx, column_length)
        except:
            pass
    try:
        writer.save()
    except:
        #new version doesnt seem to have save, instead just close
        writer.close()

def df_read(path):
    path = str(path)
    if path.endswith('csv'):
        df = pd.read_csv(path)
    elif path.endswith('xlsx'):
        df = pd.read_excel(path)
    else:
        raise ValueError('unknown suffix %s' % Path(path).suffix)
    df_strip_columns(df)
    return df

def df_to_csv(df, path, index=None, overwrite=False, **kwargs):
    mkdir(Path(path).parent)
    if not overwrite and Path(path).exists():
        raise ValueError('not overwriting already existing %s' % str(path))
    df.to_csv(str(path), index=index)

def df_save(df, path, verbose=False, *args, **kwargs):
    if verbose: print(str(path))
    if Path(path).suffix in ['.csv','csv']:
        df_to_csv(df, path, *args, **kwargs)
    elif Path(path).suffix in ['.xlsx','xlsx']:
        df_to_excel(df, path, *args, **kwargs)
    else:
        raise ValueError('unknown suffix to save df: %s' % str(path))

def df_update(df, dfo, index_col, other_cols=True):
    df = df.set_index(index_col)
    dfo = dfo.set_index(index_col)
    same_type_cols = [c for c in dfo.columns if c in df and df.dtypes[c]==dfo.dtypes[c]]
    if other_cols:
        other_cols = [c for c in dfo.columns if c not in df]
        for oc in other_cols:
            df[oc] = None
    df.update(dfo)
    df.reset_index(inplace=True, drop=False)
    #update can change dtype from int to float
    for sc in same_type_cols:
        if df.dtypes[sc]!=dfo.dtypes[sc]:
            df[sc] = df[sc].astype(dfo.dtypes[sc])
    return df

def print_df_trac(df, index_name=None, **kwargs):
    lines = df_to_trac_format(df, index_name=index_name, **kwargs)
    print('---------------------- TRAC --------------------------')
    for line in lines:
        print(line)
    print('------------------------------------------------------')
    return '\n'.join(lines)

def _df_concat_exp():
    df1 = pd.DataFrame({'A':[3,4], 'B':['bb','bbb']})
    df2 = pd.DataFrame({'A':[1,2], 'B':['cc','ccc']})
    df = pd.concat([df1, df2], ignore_index=True, axis=0)
    print_df(df)

def _my_sum(series):
    return np.sum(series)


def agg_entropy(series):
    return entropy(series.values)

def max_vote(series):
    counts = series.value_counts()
    max_vote = counts.idxmax()
    return max_vote

def unique_one(series, allowna=True):
    """ expects and returns the single unique value in the series, throws an error otherwise"""
    if is_list_or_tuple(series):
        sname = 'list'
    else:
        sname = series.name
    vals = list(set(series))
    vals_notna = [v for v in vals if str(v)!='nan']
    if not allowna and len(vals)!=len(vals_notna):
        raise ValueError(f'series {sname} contains nans: {vals}')

    if len(vals_notna)>1: #multiple values
        raise ValueError('%d>1 unique values %s in series %s!' % \
                        (len(vals), str(vals), sname))
    elif len(vals_notna)==1:
        return vals_notna[0] #single vbalue
    else:
        return vals[0] # nan

    # if len(vals)>1:
    #     all_nans = True
    #     non_nan_values = []
    #     for val in vals:
    #         if str(val)!='nan':
    #             all_nans = False
    #             break
    #     if all_nans and allowna:
    #         pass
    #     else:
    #
    #         raise ValueError('%d>1 unique values %s in series %s!' %\
    #                          (len(vals), str(vals), sname))
    # return vals[0]

def agg_join(series, conc=', ', unique=False, as_str=False, sort=False):
    # if series.isna().all():
    #     return None
    vals = list(series)
    vals = [v for v in vals if v is not None]
    if len(vals)==0:
        return None
    if as_str:
        vals = [str(v) for v in vals]
    if not is_string(vals[0]):
        raise ValueError("can't join non-strings in series %s: %s" % (series.name, str(vals)))
    if unique:
        vals = list(set(vals))
    if sort:
        vals = sorted(vals)
    return conc.join(vals)

agg_join_str = functools.partial(agg_join, unique=False, sort=True, as_str=True)
agg_join_unique = functools.partial(agg_join, unique=True, sort=True)
agg_join_unique_str = functools.partial(agg_join, unique=True, sort=True, as_str=True)

def df_value_count(df, cols, **kwargs):
    return df_value_counts(df, cols, **kwargs)

def df_value_counts(df, cols, result_value_col='value', dropna=False):
    vals = []
    if is_string(cols):
        cols = [cols]
    for col in cols:
        dfc = df[col].value_counts(dropna=dropna)
        vals.extend(dfc.index)

    vals = list(set(vals))

    col_map = {}
    col_map[result_value_col] = vals
    for col in cols:
        dfc = df[col].value_counts(dropna=dropna)
        col_values = []
        for val in vals:
            if val in dfc:
                col_values.append(dfc[val])
            else:
                col_values.append(0)
        col_map[col] = col_values

    dfr = pd.DataFrame(col_map)

    dfr.set_index(result_value_col, inplace=True)
    dfr.loc['col_sum']= dfr.sum(numeric_only=True, axis=0)
    dfr.loc[:,'row_sum'] = dfr.sum(numeric_only=True, axis=1)

    return dfr

def df_cols(df):
    return [str(c) for c in df.columns]

def df_col_to_str(df, col):
    df[col] = df[col].astype('str')

def df_move_col_to(df, col, after):
    """ moves the column after the given column """
    cols = df_cols(df)
    if is_string(after):
        after = cols.index(after)+1
    cols.remove(col)
    cols.insert(after, col)
    df = df[cols]
    return df

def df_save_excel_sheets(dfs, names, out_path, overwrite=False, csv=False):
    """ if csv, saves also cvs per df """
    if out_path is None:
        print('None out_path!'); return
    out_path = Path(out_path)
    if out_path.suffix!='xlsx':
        out_path = out_path.parent/(out_path.stem+'.xlsx')
    if out_path.exists() and not overwrite:
        print('not overwriting %s' % str(out_path))
        return

    out_path.parent.mkdir(exist_ok=True)
    writer = pd.ExcelWriter(str(out_path), engine='xlsxwriter')
    for df, name in zip(dfs, names):
        df.to_excel(writer, sheet_name=name, index=None)
        if csv:
            csv_out_path = str(out_path).replace('.xlsx','_'+name+'.csv')
            if overwrite or not Path(csv_out_path).exists():
                df.to_csv(csv_out_path, index=None)
    writer.save()
    print('%s' % str(Path(out_path).absolute()))