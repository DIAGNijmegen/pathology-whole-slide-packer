import math
import sys
import os

def is_docker():
    path = '/proc/self/cgroup'
    return (
            os.path.exists('/.dockerenv') or
            os.path.isfile(path) and any('docker' in line for line in open(path))
    )

def count_docker_cpus(quota_file=None, period_file=None, log_fct=print, default=0):
    print('count_docker_cpus..')
    try:
        with open(quota_file or "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", 'r') as content_file:
            cfs_quota_us = int(content_file.read())
        with open(period_file or "/sys/fs/cgroup/cpu/cpu.cfs_period_us", 'r') as content_file:
            cfs_period_us = int(content_file.read())
        if cfs_quota_us > 0 and cfs_period_us > 0:
            n_cpus = int(math.ceil(cfs_quota_us / cfs_period_us))
            print('%d cpus accoriding to quota' % n_cpus)
            return n_cpus

    except Exception:
        log_fct("Getting number of cpus from cfs_quota failed; using multiprocessing.cpu_count", sys.exc_info())
    print('count failed, return default %d' % default)
    return default
    # return multiprocessing.cpu_count()
