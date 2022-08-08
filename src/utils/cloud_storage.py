import time
import sys
from .qd_common import query_all_opened_file_in_system
from .qd_common import ensure_remove_file
from .qd_common import has_handle
from .qd_common import get_file_size
from .qd_common import hash_sha1
from .qd_common import acquireLock, hash_sha1, releaseLock
from .qd_common import write_to_file
import os.path as op
import glob
from pprint import pformat
from .qd_common import load_from_yaml_file
from .qd_common import cmd_run
from .qd_common import parse_iteration
import logging
from tqdm import tqdm
import os
from .qd_common import ensure_directory
from deprecated import deprecated
try:
    from azure.storage.common.storageclient import logger
    logger.propagate = False
except Exception as e:
    print(f"Cannot import azure.storage {e}")


def create_cloud_storage(x=None, config_file=None, config=None):
    if config is not None:
        return CloudStorage(config)
    if config_file is None:
        config_file = './aux_data/configs/{}blob_account.yaml'.format(x)
    config = load_from_yaml_file(config_file)
    c = CloudStorage(config)
    return c

def azcopy_upload(src, dest_url, dest_key):
    cmd = [get_azcopy(), '--source',
            src,
            '--destination',
            dest_url,
            '--exclude-older',
            '--dest-key',
            "{}".format(dest_key),
            '--quiet',
            '--parallel-level',
            '32']
    resume_file = '/tmp/azure.' + hash_sha1([src, dest_url]) + '.jnl'
    cmd.append('--resume')
    cmd.append(resume_file)
    if op.isdir(src):
        cmd.append('--recursive')
    cmd_run(cmd, shell=True)

def blob_upload_qdoutput(src_path, dest_path, client):
    to_copy = get_to_copy_file_for_qdoutput(src_path, dest_path)
    for f, d in to_copy:
        client.az_upload2(f, d)

def get_to_copy_file_for_qdoutput(src_path, dest_path):
    # upload all the files under src_path
    to_copy = []
    all_src_file = glob.glob(op.join(src_path, '*'))
    exclude_suffix = ['txt', 'zip']
    for f in all_src_file:
        if op.isfile(f) and not any(f.endswith(s) for s in exclude_suffix):
            to_copy.append((f, op.join(dest_path, op.basename(f))))

    # for the model and the tested files, only upload the best
    all_src_file = glob.glob(op.join(src_path, 'snapshot',
        'model_iter_*.caffemodel'))
    all_src_file.extend(glob.glob(op.join(src_path, 'snapshot',
        'model_iter_*.pt')))
    all_iter = [parse_iteration(f) for f in all_src_file]
    max_iters = max(all_iter)
    need_copy_files = [f for f, i in zip(all_src_file, all_iter) if i == max_iters]
    dest_snapshot = op.join(dest_path, 'snapshot')
    for f in need_copy_files:
        to_copy.append((f, op.join(dest_snapshot, op.basename(f))))
        if f.endswith('.caffemodel'):
            f = f.replace('.caffemodel', '.solverstate')
            to_copy.append((f, op.join(dest_snapshot, op.basename(f))))
    return to_copy

def blob_upload(src, dst, c=None):
    if c is None:
        c = create_cloud_storage('vig')
    c.az_upload2(src, dst)

def get_root_all_full_expid(full_expid_prefix, all_blob_name):
    # full_expid_prefix can be the folder of full_expid; or with some prefix so
    # that we can filter
    if all(b.startswith(full_expid_prefix + '/') for b in all_blob_name):
        root = full_expid_prefix
    else:
        root = op.dirname(full_expid_prefix)
    all_full_expid = set(b[len(root) + 1:].split('/')[0] for b in
            all_blob_name if b.startswith(root))
    return root, all_full_expid

def get_azcopy():
    # this is v10
    azcopy = op.expanduser('~/code/azcopy/azcopy')
    if not op.isfile(azcopy):
        azcopy = 'azcopy'
    return azcopy

def get_leaf_names(all_fname):
    # build the tree first
    from ete3 import Tree
    root = Tree()
    for fname in all_fname:
        components = fname.split('/')
        curr = root
        for com in components:
            currs = [c for c in curr.children if c.name == com]
            if len(currs) == 0:
                curr = curr.add_child(name=com)
            else:
                assert len(currs) == 1
                curr = currs[0]
    result = []
    for node in root.iter_leaves():
        ans = [s.name for s in node.get_ancestors()[:-1]]
        ans.insert(0, node.name)
        result.append('/'.join([a for a in ans[::-1]]))
    return result

def blob_download_qdoutput(src_path, target_folder):
    c = create_cloud_storage('vig')
    c.blob_download_qdoutput(src_path, target_folder)

def create_cloud_fuse(config=None):
    if config is None:
        if 'QD_CLOUD_FUSE_CONFIG_FILE' in os.environ:
            fname = os.environ['QD_CLOUD_FUSE_CONFIG_FILE']
        else:
            fname = 'aux_data/configs/cloud_fuse.yaml'
        config = load_from_yaml_file(fname)
    local_to_remote_cache = config['local_to_remote_cache']
    cloud_account = config['storage_account']
    cloud = create_cloud_storage(cloud_account)
    return CloudFuse(local_to_remote_cache, cloud)

def garbage_collection_for_cloud_fuse_loop(local_folders, total_size_limit):
    while True:
        try:
            garbage_collection_for_cloud_fuse(local_folders, total_size_limit)
        except:
            from .qd_common import print_trace
            print_trace()
        time.sleep(10)

def garbage_collection_for_cloud_fuse(local_folders, total_size_limit):
    infos = []
    for local_folder in local_folders:
        for root, _, files in os.walk(local_folder):
            for f in files:
                if f.startswith('.') or f.endswith('.tmp'):
                    continue
                f = op.join(root, f)
                if op.isfile(f):
                    info = {
                        'fname': f,
                        'size_in_bytes': get_file_size(f),
                    }
                    infos.append(info)
    total = sum([i['size_in_bytes'] for i in infos])
    #logging.info('scanned {} files'.format(len(infos)))
    if total < total_size_limit:
        #logging.info('{}G < {}G. No need to delete files'.format(
            #total / 1024 ** 3, total_size_limit / 1024 ** 3))
        return
    # we need to delete
    need_to_delete = total - total_size_limit
    opened = set(query_all_opened_file_in_system())
    logging.info('queried {} opened files in system'.format(len(opened)))
    for i in infos:
        i['has_handle'] = has_handle(i['fname'], opened)
        i['last_access_time'] = op.getatime(i['fname'])
    infos = sorted(infos, key=lambda i: (i['has_handle'], i['last_access_time']))
    deleted = []
    for i in infos:
        ensure_remove_file(i['fname'])
        need_to_delete -= i['size_in_bytes']
        deleted.append(i['fname'])
        if need_to_delete < 0:
            break
    logging.info('deleted {}'.format(deleted))

class CloudFuse(object):
    def __init__(self, local_to_remote_cache, cloud):
        self.local_to_remote_cache = local_to_remote_cache
        self.local_to_remote_cache = dict([(op.abspath(l), x) for l, x in self.local_to_remote_cache.items()])
        self.cloud = cloud
        self.invoked_collector = False

    def isfile(self, fname):
        remote_dir, cache_dir, sub_name = self.get_remote_cache(fname)
        remote_file = op.join(remote_dir, sub_name)
        if remote_file is None:
            return op.isfile(fname)
        return self.cloud.file_exists(remote_file)

    def get_file_size(self, fname):
        remote_dir, cache_dir, sub_name = self.get_remote_cache(fname)
        if sub_name is None:
            return get_file_size(fname)
        remote_file = op.join(remote_dir, sub_name)
        return self.cloud.query_info(remote_file)['size_in_bytes']

    def ensure_cache(self, fname_or_fs):
        if isinstance(fname_or_fs, str):
            fname = fname_or_fs
            remote_dir, cache_dir, sub_name = self.get_remote_cache(fname)
            remote_file = op.join(remote_dir, sub_name)
            cache_file = op.join(cache_dir, sub_name)
            if remote_file is None:
                return
            self.ensure_remote_to_cache(remote_file, cache_file)
        elif isinstance(fname_or_fs, (tuple, list)) and len(fname_or_fs) > 0:
            # for multiple files, we can download them at the same time rather
            # than one by one
            fnames = fname_or_fs
            remote_cache_infos = [self.get_remote_cache(f) for f in fnames]
            remote_cache_infos = [(rd, cd, sn) for rd, cd, sn in remote_cache_infos if not op.isfile(op.join(cd, sn))]
            if len(remote_cache_infos) > 0:
                lock_fd = self.acquireLock()
                # we need to check again
                remote_cache_infos = [self.get_remote_cache(f) for f in fnames]
                remote_cache_infos = [(rd, cd, sn) for rd, cd, sn in remote_cache_infos if not op.isfile(op.join(cd, sn))]
                remote_cache_infos = [((rd, cd), sn) for rd, cd, sn in remote_cache_infos]
                from .qd_common import list_to_dict
                rd_cd_to_sn = list_to_dict(remote_cache_infos, 0)
                for (rd, cd), sns in rd_cd_to_sn.items():
                    file_list = '/tmp/{}'.format(hash_sha1(pformat(sns)))
                    write_to_file('\n'.join(sns), file_list)
                    self.cloud.az_download(
                        rd,
                        cd,
                        is_folder=False,
                        file_list=file_list,
                        tmp_first=False,
                        sync=False,
                        retry=10,
                    )
                    for s in sns:
                        assert op.isfile(op.join(cd, s)), op.join(cd, s)
                releaseLock(lock_fd)

    def open(self, fname, mode):
        assert mode in ['r', 'rb'], 'no support for writing'
        remote_dir, cache_dir, sub_name = self.get_remote_cache(fname)
        if remote_dir is None:
            return open(fname, mode)
        remote_file = op.join(remote_dir, sub_name)
        cache_file = op.join(cache_dir, sub_name)
        self.ensure_remote_to_cache(remote_file, cache_file)
        return open(cache_file, mode)

    def ensure_del_cache(self, fname):
        remote_dir, cache_dir, sub_name = self.get_remote_cache(fname)
        if remote_dir is None:
            return
        cache_file = op.join(cache_dir, sub_name)
        #if has_handle(cache_file):
            #logging.warning('{} is used by some process'.format(
                #cache_file))
            #return False
        ensure_remove_file(cache_file)
        #return True

    def ensure_invoke_garbage_collect(self):
        # if the current occupied files take more than 0.5. Then, remove the
        # oldest files
        if self.invoked_collector:
            return
        self.invoked_collector = True
        local_caches = [c for r, c in self.local_to_remote_cache.values()]
        local_caches = list(set(local_caches))
        import threading
        # 200 GB
        limit = 200 * 1024 * 1024 * 1024
        t = threading.Thread(
            target=garbage_collection_for_cloud_fuse_loop, args=(local_caches, limit),
            daemon=True,
        )
        t.start()
        self.garbage_collector = t

    #--------------------------------------------------------------
    def acquireLock(self, x=None):
        # all files share one lock
        lock_fd = acquireLock(op.join('/tmp', 'lock_azcopy_fuse'))
        return lock_fd

    def get_remote_cache(self, fname):
        fname = op.abspath(fname)
        local_folders = [l for l in self.local_to_remote_cache if fname.startswith(l + '/')]
        if len(local_folders) == 0:
            return None, None, None
        local_folder = local_folders[0]
        remote_folder, cache_folder = self.local_to_remote_cache[local_folder]
        sub_name = fname[len(local_folder) + 1:]
        d = op.dirname(sub_name)
        sub = op.basename(sub_name)
        return op.join(remote_folder, d), op.join(cache_folder, d), sub

    def remote_to_cache(self, remote_file, cache_file):
        if op.isfile(cache_file):
            return
        lock_fd = self.acquireLock()
        if not op.isfile(cache_file):
            self.cloud.az_download(
                remote_file, cache_file, sync=False, is_folder=False)
            # the following is slower
            #self.cloud.download_to_path(
                #remote_file, cache_file, max_connections=10)
        releaseLock(lock_fd)

    def ensure_remote_to_cache(self, remote_file, cache_file):
        if not op.isfile(cache_file):
            self.remote_to_cache(remote_file, cache_file)

class CloudStorage(object):
    def __init__(self, config=None):
        if config is None:
            config_file = 'aux_data/configs/azure_blob_account.yaml'
            config = load_from_yaml_file(config_file)
        account_name = config['account_name']
        account_key = config.get('account_key')
        self.sas_token = config.get('sas_token')
        self.container_name = config['container_name']
        self.account_name = account_name
        self.account_key = account_key
        self._block_blob_service = None

    def __repr__(self):
        return 'CloudStorage(account={}, container={})'.format(
            self.account_name,
            self.container_name,
        )

    @property
    def block_blob_service(self):
        if self._block_blob_service is None:
            from azure.storage.blob import BlockBlobService
            self._block_blob_service = BlockBlobService(
                    account_name=self.account_name,
                    account_key=self.account_key,
                    sas_token=self.sas_token)
        return self._block_blob_service

    def list_blob_names(self, prefix=None, creation_time_larger_than=None):
        if creation_time_larger_than is not None:
            creation_time_larger_than = creation_time_larger_than.timestamp()
            return (b.name for b in self.block_blob_service.list_blobs(
                self.container_name,
                prefix=prefix)
                    if b.properties.creation_time.timestamp() > creation_time_larger_than)
        else:
            return self.block_blob_service.list_blob_names(self.container_name, prefix=prefix)

    def rm_prefix(self, prefix):
        all_path = self.list_blob_names(prefix)
        for p in all_path:
            logging.info('deleting {}'.format(p))
            self.rm(p)

    def rm(self, path):
        self.block_blob_service.delete_blob(self.container_name, path)

    def iter_blob_info(self, prefix=None, creation_time_larger_than=None):
        def valid(b):
            c1 = creation_time_larger_than is None or b.properties.creation_time.timestamp() > creation_time_larger_than.timestamp()
            c2 = b.name.startswith(prefix)
            return c1 and c2
        for b in self.block_blob_service.list_blobs(self.container_name, prefix=prefix):
            if valid(b):
                yield {
                    'name': b.name,
                    'size_in_bytes': b.properties.content_length,
                    'creation_time': b.properties.creation_time,
                }

    def list_blob_info(self, prefix=None, creation_time_larger_than=None):
        return list(self.iter_blob_info(prefix, creation_time_larger_than))

    def get_url(self, blob_name):
        from azure.storage.blob.models import BlobPermissions
        permission = BlobPermissions(read=True)
        import datetime
        expiry = datetime.datetime.now() + datetime.timedelta(days=30)
        sas = self.block_blob_service.generate_blob_shared_access_signature(
            self.container_name, blob_name,
            permission=permission,
            expiry=expiry,
        )
        url = self.block_blob_service.make_blob_url(self.container_name,
                                                    blob_name)
        return '{}?{}'.format(url, sas)

    def upload_stream(self, s, name, force=False):
        if not force and self.block_blob_service.exists(self.container_name,
                name):
            return self.block_blob_service.make_blob_url(
                    self.container_name,
                    name)
        else:
            if sys.version_info.major == 3 and type(s) is bytes:
                self.block_blob_service.create_blob_from_bytes(
                        self.container_name,
                        name,
                        s)
            else:
                self.block_blob_service.create_blob_from_stream(
                        self.container_name,
                        name,
                        s)
            return self.block_blob_service.make_blob_url(
                    self.container_name,
                    name)

    def upload_folder(self, folder, target_prefix):
        def remove_tailing(x):
            if x.endswith('/') or x.endswith('\\'):
                x = x[:-1]
            return x
        folder = remove_tailing(folder)
        target_prefix = remove_tailing(target_prefix)
        for root, dirs, files in os.walk(folder):
            for f in files:
                src_file = op.join(root, f)
                assert src_file.startswith(folder)
                target_file = src_file.replace(folder, target_prefix)
                self.upload_file(src_file, target_file)
            for d in dirs:
                self.upload_folder(op.join(root, d),
                        op.join(target_prefix, d))

    def upload_file(self, src_file, target_file):
        logging.info('uploading {} to {}'.format(src_file, target_file))
        #import time
        #start_time = time.time()
        bar = [None]
        last = [0]
        def upload_callback(curr, total):
            if total < 1024 ** 3:
                return
            if bar[0] is None:
                bar[0] = tqdm(total=total, unit_scale=True)
            bar[0].update(curr - last[0])
            last[0] = curr
        if target_file.startswith('/'):
            logging.info('remove strarting slash for {}'.format(target_file))
            target_file = target_file[1:]
        self.block_blob_service.create_blob_from_path(
                self.container_name,
                target_file,
                src_file,
                max_connections=8,
                progress_callback=upload_callback)

    def az_upload(self, src_dir, dest_dir):
        # this is using the old version of azcopy. prefer to use az_upload2
        dest_url = op.join('https://{}.blob.core.windows.net'.format(self.account_name),
                self.container_name, dest_dir)
        if self.account_key:
            azcopy_upload(src_dir, dest_url, self.account_key)
        else:
            raise Exception

    @deprecated('use upload()')
    def az_sync(self, src_dir, dest_dir):
        self.upload(src_dir, dest_dir)

    def upload(self, src_dir, dest_dir, from_blob=None):
        if from_blob is None:
            self.upload_from_local(src_dir, dest_dir)
        else:
            self.upload_from_another(src_dir, dest_dir, from_blob)

    def upload_from_another(self, src_dir, dest_dir, from_blob):
        assert self.sas_token
        cmd = []
        cmd.append(get_azcopy())
        if from_blob.dir_exists(src_dir) and not self.dir_exists(dest_dir):
            # in this case, azcopy will copy the local folder under the
            # destination folder, and thus we have to use the folder of
            # dest_dir as the dest_dir.
            assert op.basename(src_dir) == op.basename(dest_dir)
            dest_dir = op.dirname(dest_dir)
            cmd.append('cp')
        else:
            if self.exists(dest_dir):
                cmd.append('sync')
            else:
                cmd.append('cp')
        url = 'https://{}.blob.core.windows.net'.format(from_blob.account_name)
        url = op.join(url, from_blob.container_name, src_dir)
        assert self.sas_token.startswith('?')
        from_url = url
        url = url + from_blob.sas_token
        cmd.append(url)

        url = 'https://{}.blob.core.windows.net'.format(self.account_name)
        if dest_dir.startswith('/'):
            dest_dir = dest_dir[1:]
        url = op.join(url, self.container_name, dest_dir)
        assert self.sas_token.startswith('?')
        data_url = url
        url = url + self.sas_token
        cmd.append(url)

        if from_blob.dir_exists(src_dir):
            cmd.append('--recursive')
        if from_url == data_url:
            logging.info('no need to sync data as url is exactly the same')
            return data_url, url
        cmd_run(cmd)
        return data_url, url

    def upload_from_local(self, src_dir, dest_dir):
        assert self.sas_token
        cmd = []
        cmd.append(get_azcopy())
        if op.isdir(src_dir) and not self.dir_exists(dest_dir):
            # in this case, azcopy will copy the local folder under the
            # destination folder, and thus we have to use the folder of
            # dest_dir as the dest_dir.
            assert op.basename(src_dir) == op.basename(dest_dir)
            dest_dir = op.dirname(dest_dir)
            cmd.append('cp')
        else:
            if self.exists(dest_dir):
                cmd.append('sync')
            else:
                cmd.append('cp')
        cmd.append(op.realpath(src_dir))
        url = 'https://{}.blob.core.windows.net'.format(self.account_name)
        if dest_dir.startswith('/'):
            dest_dir = dest_dir[1:]
        url = op.join(url, self.container_name, dest_dir)
        assert self.sas_token.startswith('?')
        data_url = url
        url = url + self.sas_token
        cmd.append(url)
        if op.isdir(src_dir):
            cmd.append('--recursive')
        cmd_run(cmd)
        return data_url, url

    @deprecated('use upload')
    def az_upload2(self, src_dir, dest_dir, sync=False):
        return self.upload(src_dir, dest_dir)
        #assert self.sas_token
        #cmd = []
        #cmd.append(get_azcopy())
        #if sync:
            #cmd.append('sync')
        #else:
            #cmd.append('cp')
        #cmd.append(op.realpath(src_dir))
        #url = 'https://{}.blob.core.windows.net'.format(self.account_name)
        #if dest_dir.startswith('/'):
            #dest_dir = dest_dir[1:]
        #url = op.join(url, self.container_name, dest_dir)
        #assert self.sas_token.startswith('?')
        #data_url = url
        #url = url + self.sas_token
        #cmd.append(url)
        #if op.isdir(src_dir):
            #cmd.append('--recursive')
        #cmd_run(cmd)
        #return data_url, url

    def query_info(self, path):
        p = self.block_blob_service.get_blob_properties(self.container_name, path)
        result = {
            'size_in_bytes': p.properties.content_length,
            'creation_time': p.properties.creation_time,
        }
        return result

    def az_download_all(self, remote_path, local_path):
        all_blob_name = list(self.list_blob_names(remote_path))
        all_blob_name = get_leaf_names(all_blob_name)
        for blob_name in all_blob_name:
            target_file = blob_name.replace(remote_path, local_path)
            if not op.isfile(target_file):
                self.az_download(blob_name, target_file,
                        sync=False)

    def is_folder(self, remote_path):
        is_folder = False
        for x in self.list_blob_names(remote_path + '/'):
            is_folder = True
            break
        return is_folder

    def az_download_each(self, remote_path, local_path):
        # if it is a folder, we will download each file individually
        if remote_path.startswith('/'):
            remote_path = remote_path[1:]
        if remote_path.endswith('/'):
            remote_path = remote_path[:-1]
        is_folder = self.is_folder(remote_path)
        if is_folder:
            all_remote_file = self.list_blob_names(remote_path + '/')
            all_local_file = [op.join(local_path, r[len(remote_path) + 1:])
                              for r in all_remote_file]
        else:
            all_remote_file = [remote_path]
            all_local_file = [local_path]
        for r, l in zip(all_remote_file, all_local_file):
            self.az_download(r, l, sync=True)

    def az_download(self,
                    remote_path,
                    local_path,
                    sync=True,
                    is_folder=None,
                    tmp_first=True,
                    file_list=None,
                    retry=1,
                    ):
        from .qd_common import limited_retry_agent
        limited_retry_agent(retry, self.az_download_once,
                            remote_path, local_path, sync,
                            is_folder, tmp_first, file_list)

    def az_download_once(self,
                    remote_path,
                    local_path,
                    sync=True,
                    is_folder=None,
                    tmp_first=True,
                    file_list=None,
                    ):
        if remote_path.startswith('/'):
            remote_path = remote_path[1:]
        if remote_path.endswith('/'):
            remote_path = remote_path[:-1]
        if is_folder is None:
            is_folder = self.is_folder(remote_path)
        if sync and tmp_first:
            logging.info('if sync, no need to save to temp first')
            tmp_first = False
        if is_folder:
            if not sync and op.isdir(local_path) and tmp_first:
                if len(os.listdir(local_path)) > 0:
                    logging.error('ignore to download from {} to {}'
                                  ' since destination is not empty'.format(
                                      remote_path,
                                      local_path,
                                  ))
                    return
                from .qd_common import ensure_remove_dir
                ensure_remove_dir(local_path)
        else:
            if sync:
                if tmp_first:
                    sync = False
                elif not op.isfile(local_path):
                    sync = False
        ensure_directory(op.dirname(local_path))
        origin_local_path = local_path
        if tmp_first:
            local_path = local_path + '.tmp'
        ensure_directory(op.dirname(local_path))
        assert self.sas_token
        cmd = []
        cmd.append(get_azcopy())
        if sync:
            cmd.append('sync')
        else:
            cmd.append('cp')
        url = 'https://{}.blob.core.windows.net'.format(self.account_name)
        url = '/'.join([url, self.container_name, remote_path])
        assert self.sas_token.startswith('?')
        data_url = url
        url = url + self.sas_token
        cmd.append(url)
        if file_list:
            # requirements from file_list
            assert op.basename(local_path) == op.basename(remote_path)
            cmd.append(op.dirname(op.realpath(local_path)))
        else:
            cmd.append(op.realpath(local_path))
        if is_folder:
            cmd.append('--recursive')
            if sync:
                # azcopy's requirement
                ensure_directory(local_path)
        else:
            ensure_directory(op.dirname(local_path))
        if file_list:
            cmd.append('--list-of-files')
            cmd.append(file_list)
        cmd_run(cmd)
        if tmp_first:
            os.rename(local_path, origin_local_path)
        return data_url, url

    def download_to_path(self, blob_name, local_path,
            max_connections=2):
        dir_path = op.dirname(local_path)
        if op.isfile(dir_path) and get_file_size(dir_path) == 0:
            os.remove(dir_path)
        ensure_directory(dir_path)
        tmp_local_path = local_path + '.tmp'
        pbar = {}
        def progress_callback(curr, total):
            if len(pbar) == 0:
                pbar['tqdm'] = tqdm(total=total, unit_scale=True)
                pbar['last'] = 0
                pbar['count'] = 0
            pbar['count'] += 1
            if pbar['count'] > 100:
                pbar['tqdm'].update(curr - pbar['last'])
                pbar['last'] = curr
                pbar['count'] = 0
        self.block_blob_service.get_blob_to_path(self.container_name,
                blob_name, tmp_local_path,
                progress_callback=progress_callback,
                max_connections=max_connections)
        os.rename(tmp_local_path, local_path)

    def download_to_stream(self, blob_name, s, max_connections=2):
        pbar = {}
        def progress_callback(curr, total):
            if len(pbar) == 0:
                pbar['tqdm'] = tqdm(total=total, unit_scale=True)
                pbar['last'] = 0
                pbar['count'] = 0
            pbar['count'] += 1
            if pbar['count'] > 1:
                pbar['tqdm'].update(curr - pbar['last'])
                pbar['last'] = curr
                pbar['count'] = 0
        self.block_blob_service.get_blob_to_stream(self.container_name,
                                                   blob_name,
                                                   s,
                                                   max_connections=max_connections,
                                                   progress_callback=progress_callback,
                                                   )

    def exists(self, path):
        return self.file_exists(path) or self.dir_exists(path)

    def file_exists(self, path):
        fp = acquireLock('/tmp/{}.lock'.format(hash_sha1(path)))
        result = self.block_blob_service.exists(self.container_name, path)
        releaseLock(fp)
        return result

    def dir_exists(self, dir_path):
        dir_path = op.normpath(dir_path)
        for x in self.list_blob_names(prefix=dir_path + '/'):
            return True
        return False

    def blob_download_qdoutput(
        self,
        src_path,
        target_folder,
        latest_only=True,
        creation_time_larger_than=None,
        too_large_limit_in_gb=None,
        ignore_base_fname_patterns=None,
        dry_run=False,
    ):
        def is_in_snapshot(b):
            parts = list(b.split('/'))
            return 'snapshot' in parts and parts.index('snapshot') < len(parts) - 1
        all_blob_name = list(self.list_blob_names(
            src_path,
            creation_time_larger_than))
        # remove snapshot/model_iter/abc.bin
        clean = []
        for b in all_blob_name:
            parts = list(b.split('/'))
            if 'snapshot' in parts and \
                    parts.index('snapshot') < len(parts) - 2:
                continue
            else:
                clean.append(b)
        all_blob_name = clean

        all_blob_name = get_leaf_names(all_blob_name)
        in_snapshot_blobs = [b for b in all_blob_name if is_in_snapshot(b)]
        not_in_snapshot_blobs = [b for b in all_blob_name if not is_in_snapshot(b)]
        try:
            not_in_snapshot_blobs.remove(src_path)
        except:
            pass
        try:
            not_in_snapshot_blobs.remove(src_path + '/snapshot')
        except:
            pass
        need_download_blobs = []
        need_download_blobs.extend(not_in_snapshot_blobs)
        iters = [parse_iteration(f) for f in in_snapshot_blobs]
        if len(iters) > 0 and latest_only:
            max_iters = max(iters)
            need_download_blobs.extend([f for f, i in zip(in_snapshot_blobs, iters) if i ==
                    max_iters or f.endswith('.report')])
        need_download_blobs.extend([f for f, i in zip(in_snapshot_blobs, iters) if
                i == -2])
        to_remove = []
        for i, b1 in enumerate(need_download_blobs):
            for b2 in need_download_blobs:
                if b1 != b2 and b2.startswith(b1) and b2.startswith(b1 + '/'):
                    to_remove.append(b1)
                    break
        for t in to_remove:
            need_download_blobs.remove(t)
        need_download_blobs = [t for t in need_download_blobs
            if not t.endswith('.tmp')]
        f_target_f = [(f, f.replace(src_path, target_folder)) for f in
            need_download_blobs]
        f_target_f = [(f, target_f) for f, target_f in f_target_f if not op.isfile(target_f)]
        f_target_f = [(f, target_f) for f, target_f in f_target_f if len(f) > 0]

        if ignore_base_fname_patterns is not None:
            import re
            result = []
            for f, target_f in f_target_f:
                if any(re.match(p, f) is not None for p in
                       ignore_base_fname_patterns):
                    logging.info('ignore {} due to reg pattern matching'.format(f))
                else:
                    result.append((f, target_f))
            f_target_f = result

        if too_large_limit_in_gb is not None:
            logging.info('before size filtering = {}'.format(len(f_target_f)))
            f_target_f = [(f, target_f) for f, target_f in f_target_f if
                          self.query_info(f)['size_in_bytes'] / 1024. ** 3 <=
                          too_large_limit_in_gb]
            logging.info('after size filtering = {}'.format(len(f_target_f)))
        for f, target_f in tqdm(f_target_f):
            logging.info('download {} to {}'.format(f, target_f))
            try:
                if not dry_run:
                    self.az_download(f, target_f)
            except:
                pass
                    #self.download_to_path(f, target_f)

if __name__ == '__main__':
    from .qd_common import init_logging
    from .qd_common import parse_general_args
    init_logging()
    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)
