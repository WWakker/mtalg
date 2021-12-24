import os


def check_threads(threads):
    if ('MAKE_CORES_DANCE_LIKE_CHICKENS' not in os.environ and
        'CDSW_NODE_NAME' in os.environ                     and 
        'DISC_KRB_REALM' in os.environ                     and 
        'CDSW_CPU_MILLICORES' in os.environ                and 
        os.name == 'posix'                                 and 
        os.environ['DISC_KRB_REALM'] == 'MIT01.ECB.DE'):
        threads = min(threads, int(os.environ['CDSW_CPU_MILLICORES']) // 1000)
    return threads
