# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import errno
import os
import os.path as op
import re
import logging
import numpy as np
import torch
import random
import shutil
from .comm import is_main_process
import yaml
import json
from .basic_utils import save_json
from easydict import EasyDict as edict
from .logger import LOGGER as logger
from pprint import pformat


def ensure_directory(path):
    if path == '' or path == '.':
        return
    if path != None and len(path) > 0:
        assert not op.isfile(path), '{} is a file'.format(path)
        if not os.path.exists(path) and not op.islink(path):
            try:
                os.makedirs(path)
            except:
                if os.path.isdir(path):
                    # another process has done makedir
                    pass
                else:
                    raise


def get_user_name():
    import getpass
    return getpass.getuser()


def acquireLock(lock_f='/tmp/lockfile.LOCK'):
    ''' acquire exclusive lock file access '''
    import fcntl
    locked_file_descriptor = open(lock_f, 'w+')
    fcntl.lockf(locked_file_descriptor, fcntl.LOCK_EX)
    return locked_file_descriptor


def releaseLock(locked_file_descriptor):
    ''' release exclusive lock file access '''
    locked_file_descriptor.close()


def hash_sha1(s):
    import hashlib
    if type(s) is not str:
        s = pformat(s)
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


def print_trace():
    import traceback
    traceback.print_exc()


def limited_retry_agent(num, func, *args, **kwargs):
    for i in range(num):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.info('fails with \n{}: tried {}-th time'.format(
                e,
                i + 1))
            import time
            print_trace()
            if i == num - 1:
                raise
            time.sleep(5)


def exclusive_open_to_read(fname, mode='r'):
    disable_lock = os.environ.get('QD_DISABLE_EXCLUSIVE_READ_BY_LOCK')
    if disable_lock is not None:
        disable_lock = int(disable_lock)
    if not disable_lock:
        user_name = get_user_name()
        lock_fd = acquireLock(op.join('/tmp',
            '{}_lock_{}'.format(user_name, hash_sha1(fname))))
    #try:
    # in AML, it could fail with Input/Output error. If it fails, we will
    # use azcopy as a fall back solution for reading
    fp = limited_retry_agent(10, open, fname, mode)
    #except:
        #if 'FILE_OPEN_AZCOPY_BLOB_ACCOUNT_PATH' in os.environ:
            #return azcopy_read(fname)
        #else:
            #raise
    if not disable_lock:
        releaseLock(lock_fd)
    return fp


class NoOp(object):
    """ useful for distributed training No-Ops """
    def __getattr__(self, name):
        return self.noop

    def noop(self, *args, **kwargs):
        return


def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def mkdir(path):
    # if it is the current folder, skip.
    # otherwise the original code will raise FileNotFoundError
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_config(cfg, path):
    if is_main_process():
        with open(path, 'w') as f:
            f.write(cfg.dump())


def config_iteration(output_dir, max_iter):
    save_file = os.path.join(output_dir, 'last_checkpoint')
    iteration = -1
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            fname = f.read().strip()
        model_name = os.path.basename(fname)
        model_path = os.path.dirname(fname)
        if model_name.startswith('model_') and len(model_name) == 17:
            iteration = int(model_name[-11:-4])
        elif model_name == "model_final":
            iteration = max_iter
        elif model_path.startswith('checkpoint-') and len(model_path) == 18:
            iteration = int(model_path.split('-')[-1])
    return iteration


def get_matching_parameters(model, regexp, none_on_empty=True):
    """Returns parameters matching regular expression"""
    if not regexp:
        if none_on_empty:
            return {}
        else:
            return dict(model.named_parameters())
    compiled_pattern = re.compile(regexp)
    params = {}
    for weight_name, weight in model.named_parameters():
        if compiled_pattern.match(weight_name):
            params[weight_name] = weight
    return params


def freeze_weights(model, regexp):
    """Freeze weights based on regular expression."""
    for weight_name, weight in get_matching_parameters(model, regexp).items():
        weight.requires_grad = False
        logger.info("Disabled training of {}".format(weight_name))


def unfreeze_weights(model, regexp, backbone_freeze_at=-1,
        is_distributed=False):
    """
    WARNING: This is not fully tested and may have issues. Now it is not used 
    during training but keep it here for future reference. 
    Unfreeze weights based on regular expression.
    This is helpful during training to unfreeze freezed weights after
    other unfreezed weights have been trained for some iterations.
    """
    for weight_name, weight in get_matching_parameters(model, regexp).items():
        weight.requires_grad = True
        logger.info("Enabled training of {}".format(weight_name))
    if backbone_freeze_at >= 0:
        logger.info("Freeze backbone at stage: {}".format(backbone_freeze_at))
        if is_distributed:
            model.module.backbone.body._freeze_backbone(backbone_freeze_at)
        else:
            model.backbone.body._freeze_backbone(backbone_freeze_at)


def delete_tsv_files(tsvs):
    for t in tsvs:
        if op.isfile(t):
            try_delete(t)
        line = op.splitext(t)[0] + '.lineidx'
        if op.isfile(line):
            try_delete(line)


def concat_files(ins, out):
    mkdir(op.dirname(out))
    out_tmp = out + '.tmp'
    with open(out_tmp, 'wb') as fp_out:
        for i, f in enumerate(ins):
            logging.info('concating {}/{} - {}'.format(i, len(ins), f))
            with open(f, 'rb') as fp_in:
                shutil.copyfileobj(fp_in, fp_out, 1024*1024*10)
    os.rename(out_tmp, out)


def concat_tsv_files(tsvs, out_tsv):
    concat_files(tsvs, out_tsv)
    sizes = [os.stat(t).st_size for t in tsvs]
    sizes = np.cumsum(sizes)
    all_idx = []
    for i, t in enumerate(tsvs):
        for idx in load_list_file(op.splitext(t)[0] + '.lineidx'):
            if i == 0:
                all_idx.append(idx)
            else:
                all_idx.append(str(int(idx) + sizes[i - 1]))
    with open(op.splitext(out_tsv)[0] + '.lineidx', 'w') as f:
        f.write('\n'.join(all_idx))


def load_list_file(fname):
    with open(fname, 'r') as fp:
        lines = fp.readlines()
    result = [line.strip() for line in lines]
    if len(result) > 0 and result[-1] == '':
        result = result[:-1]
    return result


def try_once(func):
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.info('ignore error \n{}'.format(str(e)))
    return func_wrapper


@try_once
def try_delete(f):
    os.remove(f)


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def print_and_run_cmd(cmd):
    print(cmd)
    os.system(cmd)


def write_to_yaml_file(context, file_name):
    with open(file_name, 'w') as fp:
        yaml.dump(context, fp, encoding='utf-8')


def load_from_yaml_file(yaml_file):
    with open(yaml_file, 'r') as fp:
        return yaml.load(fp, Loader=yaml.CLoader)


def parse_yaml_file(yaml_file):
    r = re.compile('.*fea.*lab.*.yaml')
    temp = op.basename(yaml_file).split('.')
    split_name = temp[0]
    if r.match(yaml_file) is not None:
        fea_folder = '.'.join(temp[temp.index('fea') + 1 : temp.index('lab')])
        lab_folder = '.'.join(temp[temp.index('lab') + 1 : -1])
    else:
        fea_folder, lab_folder = None, None
    return split_name, fea_folder, lab_folder


def check_yaml_file(yaml_file):
    # check yaml file, generate if possible
    if not op.isfile(yaml_file):
        try:
            split_name, fea_folder, lab_folder = parse_yaml_file(yaml_file)
            if fea_folder and lab_folder:
                base_yaml_file = op.join(op.dirname(yaml_file), split_name + '.yaml')
                if op.isfile(base_yaml_file):
                    data = load_from_yaml_file(base_yaml_file)
                    data['feature'] = op.join(fea_folder, split_name + '.feature.tsv')
                    data['label'] = op.join(lab_folder, split_name + '.label.tsv')
                    assert op.isfile(op.join(op.dirname(base_yaml_file), data['feature']))
                    assert op.isfile(op.join(op.dirname(base_yaml_file), data['label']))
                    if is_main_process():
                        write_to_yaml_file(data, yaml_file)
                        print("generate yaml file: {}".format(yaml_file))
        except:
            raise ValueError("yaml file: {} does not exist and cannot create it".format(yaml_file))


