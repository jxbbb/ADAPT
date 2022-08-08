
def print_csv_table(r, rows, cols):
    '''
    r: r[row][col]
    '''
    all_line = []
    all_line.append(',' + ','.join(map(str, cols)))
    def quote_if_comma(p):
        if ',' in p:
            return '"' + p + '"'
        else:
            return p
    for row in rows:
        parts = []
        parts.append(row)
        for col in cols:
            parts.append(r[row][col])
        all_line.append(','.join(map(quote_if_comma, map(str, parts))))
    return '\n'.join(all_line)

def cartesian_index(sizes):
    index = [0] * len(sizes)
    yield index

    def is_end(index, sizes):
        for i in range(len(index)):
            if index[i] != sizes[i] - 1:
                return False
        return True

    while not is_end(index, sizes):
        found = -1
        for i in range(len(index) - 1, -1, -1):
            if index[i] == sizes[i] - 1:
                index[i] = 0
            else:
                found = i
                break
        if found != -1:
            index[found] = index[found] + 1
            yield index

def _spans(sizes):
    span = []
    for i, s in enumerate(sizes):
        x = 1
        for j in range(i + 1, len(sizes)):
            x = x * sizes[j]
        span.append(x)
    return span

def _dup(sizes):
    dup = []
    k = 1
    for i in range(len(sizes)):
        dup.append(k)
        k = k * sizes[i]
    return dup

def _extract(r, names, index):
    x = r
    for i in range(len(index)):
        k = names[i][index[i]]
        if k not in x:
            return ''
        x = x[k]
    return x

def print_latex_table(r, all_rows, all_cols, **kwargs):
    return print_m_table(r, all_rows, all_cols, **kwargs)

def print_simple_latex_table(all_a2b, keys, caption=None,
        label=None, span_two=False,
        interval=None):
    lines = []
    if span_two:
        lines.append('\\begin{table*}')
    else:
        lines.append('\\begin{table}')
    lines.append('\\centering')
    if caption:
        lines.append('\\caption{{{}}}'.format(caption))
    if label:
        lines.append('\\label{{{}}}'.format(label))
    num_cols = 0
    for k in keys:
        if isinstance(k, str):
            num_cols += 1
        else:
            assert isinstance(k, dict) and len(k) == 1
            num_cols += len(k[list(k.keys())[0]])
    if interval:
        assert len(interval) == num_cols - 1
        column_config = ''.join(['c@{{{}}}'.format('~' * i) if i is not None
            else 'c' for i in interval])
        column_config += 'c'
    else:
        column_config = 'c' * num_cols
    line = '\\begin{{tabular}}{{{}}}'.format(column_config )
    lines.append(line)
    lines.append('\\toprule')

    if any(isinstance(k, dict) for k in keys):
        def get_first(k):
            if isinstance(k, str):
                return ''
                #return '\multirow{{2}}{{*}}{{{}}}'.format(k)
            else:
                sub_num_cols = len(k[list(k.keys())[0]])
                return '\multicolumn{{{}}}{{c}}{{{}}}'.format(sub_num_cols,
                        list(k.keys())[0])
        def get_seconds(k):
            if isinstance(k, str):
                return [k]
            else:
                return k[list(k.keys())[0]]
        line = ' & '.join([get_first(k) for k in keys])
        line = line + '\\\\'
        lines.append(line)
        start = 1
        for k in keys:
            if isinstance(k, dict):
                sub_len = len(k[list(k.keys())[0]])
                line = '\\cmidrule(lr){{{}-{}}}'.format(start,
                        start + sub_len - 1)
                lines.append(line)
                start += sub_len
            else:
                start += 1

        line = ' & '.join(sk for k in keys for sk in get_seconds(k))
        line = line + '\\\\'
        lines.append(line)
        keys2 = []
        for k in keys:
            if isinstance(k, str):
                keys2.append(k)
            else:
                for s_k, s_v in k.items():
                    for a in s_v:
                        keys2.append(s_k + '$' + a)
        keys = keys2
    else:
        line = ' & '.join(keys)
        line = line + '\\\\'
        lines.append(line)
    lines.append('\\midrule')

    for a2b in all_a2b:
        line = ' & '.join(map(str, [a2b.get(k, '') for k in keys]))
        line = line + '\\\\'
        lines.append(line)
        if a2b.get('__add_line_after'):
            lines.append('\\midrule')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    if span_two:
        lines.append('\\end{table*}')
    else:
        lines.append('\\end{table}')

    return '\n'.join(lines)

def print_m_table(r, all_rows, all_cols, caption=None,
        compact=False):
    sizes_cols = list(map(len, all_cols))
    cols_dup = _dup(sizes_cols)
    cols_span = _spans(sizes_cols)
    num_cols = cols_span[0] * sizes_cols[0] + len(all_rows)

    lines = []
    lines.append('\\begin{table*}')
    lines.append('\\centering')
    if caption:
        lines.append('\\caption{{{}}}'.format(caption))
        lines.append('\\label{{{}}}'.format(caption.replace(' ', '_')))
    if compact:
        c = '@{~}c'
    else:
        c = 'c'
    line = '\\begin{{tabular}}{{{}@{{}}}}'.format(c * num_cols)
    lines.append(line)
    lines.append('\\toprule')

    for i in range(len(all_cols)):
        line = ''
        for j in range(len(all_rows) - 1):
            line = line + '&'
        s = cols_span[i]
        for j in range(cols_dup[i]):
            for k in range(len(all_cols[i])):
                if s == 1:
                    line = line + '&{}'.format(all_cols[i][k])
                else:
                    line = line + '&\multicolumn{{{0}}}{{c}}{{{1}}}'.format(s,
                            all_cols[i][k])
        line = line + '\\\\'
        lines.append(line)
        lines.append('\\midrule')
    sizes_rows = list(map(len, all_rows))
    rows_span = _spans(sizes_rows)
    digit_format = '&{}'
    for index in cartesian_index(sizes_rows):
        line = ''
        for i in range(len(index)):
            prefix = '' if i == 0 else '&'
            if all(v == 0 for v in index[i + 1: ]):
                if rows_span[i] == 1:
                    line = '{}{}{}'.format(line, prefix,
                            all_rows[i][index[i]])
                else:
                    line = line + prefix + \
                            '\multirow{{{0}}}{{*}}{{{1}}}'.format(rows_span[i],
                                            all_rows[i][index[i]])
            else:
                if rows_span[i] == 1:
                    line = '{}{}{}'.format(line, prefix, all_rows[i][index[i]])
        for col_index in cartesian_index(sizes_cols):
            value = _extract(_extract(r, all_rows, index), all_cols, col_index)
            line = line + digit_format.format(value)
        line = line + '\\\\'
        lines.append(line)
        is_end_first_index = True
        for i in range(1, len(index)):
            if index[i] != sizes_rows[i] - 1:
                is_end_first_index = False
                break
        if is_end_first_index:
            if index[0] != sizes_rows[0] - 1:
                lines.append('\\midrule')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table*}')

    return '\n'.join(lines)

def print_table(r, rows, cols):
    return print_m_table(r, [rows], [cols])

def test_print_m_table():
    r = {}
    r['dog'] = {}
    r['dog']['dog1'] = {}
    r['dog']['dog1']['s'] = {}
    r['dog']['dog1']['s']['s1'] = 0
    r['dog']['dog1']['s']['s2'] = 1
    r['dog']['dog2'] = {}
    r['dog']['dog2']['s'] = {}
    r['dog']['dog2']['s']['s1'] = 2
    r['dog']['dog2']['s']['s2'] = 3

    import logging
    logging.info(print_m_table(r, [['dog'], ['dog1', 'dog2']], [['s'], ['s1',
        's2']]))


