#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generates lyrics based on sample input."""

import collections
import io
import random

defaultdict = collections.defaultdict

LUO_DAYOU_LYRICS_FILE = 'data/luo_dayou_lyrics.txt'
NUM_CHARS_PER_SONG = 200

def NormalizeFileLines(path):
  normalized_lines = []
  for line in io.open(path, mode='r', encoding='utf-8').readlines():
    line = line.strip()
    if not line or (u'罗大佑' in line):
      continue

    prefix = ''
    for ch in line:
      if ord(ch) < 128 or ch in u'　！…、—○':  # Treat non-Chinese as separater.
        if prefix:
          normalized_lines.append(prefix)
          prefix = ''
      else:
        prefix += ch
    if prefix:
      normalized_lines.append(prefix)
      prefix = ''
  return normalized_lines
      
def BuildUnigramFrequencyMap(lines):
  map = defaultdict(lambda: 0)
  for line in lines:
    for ch in line:
      map[ch] += 1
    map[''] += 1
  return map

def BuildBigramFrequencyMap(lines):
  map = defaultdict(lambda: defaultdict(lambda: 0))
  for line in lines:
    ch0 = ''
    for ch1 in line:
      map[ch0][ch1] += 1
      ch0 = ch1
    map[ch1][''] += 1
  return map

def BuildTrigramFrequencyMap(lines):
  map = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
  for line in lines:
    ch0 = ''
    ch1 = ''
    for ch2 in line:
      map[ch0][ch1][ch2] += 1
      ch0 = ch1
      ch1 = ch2
    map[ch0][ch1][''] += 1
    map[ch1][''][''] += 1
  return map

def PrintFrequencyMap(freq_map):
  freq_list = []
  for ch, count in freq_map.items():
    freq_list.append((count, ch))
  freq_list = sorted(freq_list, reverse=True)

  for count, ch in freq_list:
    print('%s: %d' % (ch, count))

def WeightedSample(freq_map):
  total_count = sum(freq_map.values())
  i = random.randrange(total_count)
  start = 0
  for x, count in freq_map.items():
    if start <= i and i < start + count:
      return x
    start += count
  return ''
    
random.seed()
lines = NormalizeFileLines(LUO_DAYOU_LYRICS_FILE)
uni_freq_map = BuildUnigramFrequencyMap(lines)
bi_freq_map = BuildBigramFrequencyMap(lines)
tri_freq_map = BuildTrigramFrequencyMap(lines)

lyrics = ''
for _ in range(NUM_CHARS_PER_SONG):
  ch = WeightedSample(uni_freq_map)
  if ch:
    lyrics += ch
  else:
    lyrics += '\n'
print('----')
print(lyrics)

lyrics = ''
ch = ''
for _ in range(NUM_CHARS_PER_SONG):
  freq_map = bi_freq_map[ch]
  ch = WeightedSample(freq_map)
  if ch:
    lyrics += ch
  else:
    lyrics += '\n'
print('----')
print(lyrics)

lyrics = ''
ch0 = ''
ch1 = ''
for _ in range(NUM_CHARS_PER_SONG):
  freq_map = tri_freq_map[ch0][ch1]
  if len(freq_map) <= 1:
    freq_map = bi_freq_map[ch1]
  ch2 = WeightedSample(freq_map)
  if ch2:
    lyrics += ch2
  else:
    lyrics += '\n'
  ch0 = ch1
  ch1 = ch2
print('----')
print(lyrics)
  
  

