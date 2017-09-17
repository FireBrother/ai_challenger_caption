import time

import numpy as np
import redis

global_redis = redis.Redis(host='162.105.86.208', port=6379, db=1)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print '%r %2.2f sec' % \
              (method.__name__, te - ts)
        return result

    return timed


@timeit
def load_challenger_ai_data(path='./', split='train'):
    valid_split = ['train']
    if split not in valid_split:
        raise ValueError('split should be in %r, but %r got.' % (valid_split, split))


def get_feture_by_id(image_id):
    value = global_redis.get('image_feature:{}:value'.format(image_id))
    info = global_redis.get('image_feature:{}:info'.format(image_id))
    d_type = info.split('|')[0]
    size = [int(x) for x in info.split('|')[1:]]
    return np.fromstring(value, d_type).reshape(size)


if __name__ == '__main__':
    # load_challenger_ai_data(split='train')
    print get_feture_by_id('8f00f3d0f1008e085ab660e70dffced16a8259f6.jpg')
