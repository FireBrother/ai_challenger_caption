import code
import codecs
import json
from collections import Counter

import cPickle

import redis
import scipy
from scipy import ndimage

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.models import vgg19

from utils import timeit
from config import *


@timeit
def process_caption_data(caption_path):
    with open(caption_path, 'r') as f:
        caption_data = json.load(f)

    annotations = []
    for item in caption_data:
        for caption in item['caption']:
            annotations.append({'caption': caption, 'image_id': item['image_id']})

    return annotations

    # max_len = 0
    # sent = ''
    # for item in caption_data:
    #     for c in item['caption']:
    #         if len(c) > max_len:
    #             max_len = len(c)
    #             sent = c
    # print max_len, sent


@timeit
def build_vocab(annotations, thresh=1):
    counter = Counter()
    for i, item in enumerate(annotations):
        for w in item['caption']:
            counter[w] += 1

    vocab = [w for w in counter if counter[w] >= thresh]
    print 'total words {}, {} remaining with threshold {}'.format(len(counter), len(vocab), thresh)

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2, u'<UNK>': 3}
    idx = 4
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    return word_to_idx


@timeit
def build_caption_tokens(annotations, word_to_idx, max_length=50):
    print 'building caption tokens'
    caption_tokens = np.ndarray((len(annotations), max_length + 2)).astype(np.int32)

    for i, item in enumerate(annotations):
        caption_token = []
        caption_token.append(word_to_idx['<START>'])

        for w in item['caption']:
            caption_token.append(word_to_idx.get(w, word_to_idx['<UNK>']))

        caption_token.append(word_to_idx['<END>'])
        caption_token.extend([word_to_idx['<NULL>']] * (max_length + 2 - len(caption_token)))

        print item['caption']
        print caption_token

        caption_tokens[i, :] = np.asarray(caption_token)

    return caption_tokens


def build_image_features(caption_data_path, image_dir):
    print 'building image features'
    with open(caption_data_path, 'r') as f:
        caption_data = json.load(f)
    n_examples = len(caption_data)
    batch_size = 64
    print 'n_examples:{}, batch_size:{}'.format(n_examples, batch_size)

    model = vgg19(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # model.classifier._modules['6'] = nn.Linear(4096, 4096, bias=False)
    new_classifier = nn.Sequential(*list(model.classifier.children())[:-3])
    model.classifier = new_classifier
    if torch.cuda.is_available():
        print 'using cuda'
        model = model.cuda()
    r = redis.Redis(host='162.105.86.208', port=6379, db=1)

    # image_features = np.ndarray([n_examples, 512, 7, 7], dtype=np.float32)
    for start, end in zip(range(0, n_examples, batch_size),
                          range(batch_size, n_examples + batch_size, batch_size)):
        print start
        image_batch_file = [x['image_id'] for x in caption_data[start:end]]
        img_batch = torch.from_numpy(np.array(
            map(lambda x: scipy.misc.imresize(ndimage.imread(image_dir + '/' + x, mode='RGB'), (224, 224)),
                image_batch_file)).astype(np.float32))
        img_batch.transpose_(1, 2)
        img_batch.transpose_(1, 3)
        tensor = Variable(img_batch).cuda()
        # print model.features(tensor)
        features = model.features(tensor)
        np_array = features.cpu().data.numpy()
        b, l, w, h = np_array.shape
        for i in range(b):
            key = 'image_feature:{}'.format(caption_data[start+i]['image_id'])
            value = np_array[i, :, :, :].ravel().tostring()
            info = '{}|{}|{}|{}'.format(str(np_array.dtype), l, w, h)
            r.set(key+':info', info)
            r.set(key+':value', value)
        # code.interact(banner='', local=locals())
        # image_features[start:end, :] = features


def main():
    # train_annotations = process_caption_data(train_annotations_path)
    # cPickle.dump(train_annotations, open('train_annotations.pkl', 'w'))

    # word_to_idx = build_vocab(train_annotations)
    # cPickle.dump(word_to_idx, open('word_to_idx.pkl', 'w'))

    # train_annotations = cPickle.load(open('train_annotations.pkl'))
    # word_to_idx = cPickle.load(open('word_to_idx.pkl'))
    # build_caption_tokens(train_annotations, word_to_idx, 20)

    build_image_features(train_annotations_path, train_images_path)


if __name__ == '__main__':
    main()
