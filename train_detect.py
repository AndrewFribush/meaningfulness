#!/usr/bin/python

import math
import pickle

accepted_chars = 'abcdefghijklmnopqrstuvwxyz '
trainer = 'trainer.txt'
good = 'good.txt'
bad = 'bad.txt'

pos = dict([(char, idx) for idx, char in enumerate(accepted_chars)])

def normalize(line):
    return [c.lower() for c in line if c.lower() in accepted_chars]

def ngram(n, l):
    filtered = normalize(l)
    for start in range(0, len(filtered) - n + 1):
        yield ''.join(filtered[start:start + n])

def train():
    k = len(accepted_chars)
    counts = [[10 for i in xrange(k)] for i in xrange(k)]

    for line in open(trainer): #put training
        for a, b in ngram(2, line):
            counts[pos[a]][pos[b]] += 1

    for i, row in enumerate(counts):
        s = float(sum(row))
        for j in xrange(len(row)):
            row[j] = math.log(row[j] / s)

    good_probs = [avg_transition_prob(l, counts) for l in open(good)] #put good
    bad_probs = [avg_transition_prob(l, counts) for l in open(bad)] #put bad

    assert min(good_probs) > max(bad_probs)

    thresh = (min(good_probs) + max(bad_probs)) / 2
    pickle.dump({'mat': counts, 'thresh': thresh}, open('gib_model.pki', 'wb'))

def avg_transition_prob(l, log_prob_mat):
    log_prob = 0.0
    transition_ct = 0
    for a, b in ngram(2, l):
        log_prob += log_prob_mat[pos[a]][pos[b]]
        transition_ct += 1
    return math.exp(log_prob / (transition_ct or 1))

if __name__ == '__main__':
    train()