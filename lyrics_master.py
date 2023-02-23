#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generates lyrics based on sample input."""

from typing import Dict, List, Tuple

import collections
import io
import random

defaultdict = collections.defaultdict

LUO_DAYOU_LYRICS_FILE = "data/luo_dayou_lyrics.txt"
NUM_CHARS_PER_SONG = 200


def NormalizeFileLines(path: str) -> List[str]:
    normalized_lines: List[str] = []
    for line in io.open(path, mode="r", encoding="utf-8").readlines():
        line = line.strip()
        if not line or ("罗大佑" in line):
            continue

        prefix = ""
        for ch in line:
            if ord(ch) < 128 or ch in "　！…、—○":  # Treat non-Chinese as separater.
                if prefix:
                    normalized_lines.append(prefix)
                    prefix = ""
            else:
                prefix += ch
        if prefix:
            normalized_lines.append(prefix)
            prefix = ""
    return normalized_lines


def BuildUnigramFrequencyMap(lines: List[str]) -> Dict[str, int]:
    map: Dict[str, int] = defaultdict(lambda: 0)
    for line in lines:
        for ch in line:
            map[ch] += 1
        map[""] += 1
    return map


def BuildBigramFrequencyMap(lines: List[str]) -> Dict[str, Dict[str, int]]:
    map: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
    for line in lines:
        ch0 = ""
        ch1 = ""
        for ch1 in line:
            map[ch0][ch1] += 1
            ch0 = ch1
        map[ch1][""] += 1
    return map


def BuildTrigramFrequencyMap(lines: List[str]) -> Dict[str, Dict[str, Dict[str, int]]]:
    map: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: 0))
    )
    for line in lines:
        ch0 = ""
        ch1 = ""
        for ch2 in line:
            map[ch0][ch1][ch2] += 1
            ch0 = ch1
            ch1 = ch2
        map[ch0][ch1][""] += 1
        map[ch1][""][""] += 1
    return map


def BuildQuadgramFrequencyMap(
    lines: List[str],
) -> Dict[str, Dict[str, Dict[str, Dict[str, int]]]]:
    map: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    )
    for line in lines:
        ch0 = ""
        ch1 = ""
        ch2 = ""
        ch3 = ""
        for ch3 in line:
            map[ch0][ch1][ch2][ch3] += 1
            ch0 = ch1
            ch1 = ch2
            ch2 = ch3
        map[ch1][ch2][ch3][""] += 1
        map[ch2][ch3][""][""] += 1
        map[ch3][""][""][""] += 1
    return map


def PrintFrequencyMap(freq_map: Dict[str, int]) -> None:
    freq_list: List[Tuple[int, str]] = []
    for ch, count in freq_map.items():
        freq_list.append((count, ch))
    freq_list = sorted(freq_list, reverse=True)

    for count, ch in freq_list:
        print("%s: %d" % (ch, count))


def WeightedSample(freq_map: Dict[str, int]) -> str:
    total_count = sum(freq_map.values())
    i = random.randrange(total_count)
    start = 0
    for x, count in freq_map.items():
        if start <= i and i < start + count:
            return x
        start += count
    return ""


random.seed()
lines = NormalizeFileLines(LUO_DAYOU_LYRICS_FILE)
uni_freq_map = BuildUnigramFrequencyMap(lines)
bi_freq_map = BuildBigramFrequencyMap(lines)
tri_freq_map = BuildTrigramFrequencyMap(lines)
quad_freq_map = BuildQuadgramFrequencyMap(lines)

lyrics = ""
for _ in range(NUM_CHARS_PER_SONG):
    ch = WeightedSample(uni_freq_map)
    if ch:
        lyrics += ch
    else:
        lyrics += "\n"
print("----")
print(lyrics)

lyrics = ""
ch = ""
for _ in range(NUM_CHARS_PER_SONG):
    freq_map: Dict[str, int] = bi_freq_map[ch]
    ch = WeightedSample(freq_map)
    if ch:
        lyrics += ch
    else:
        lyrics += "\n"
print("----")
print(lyrics)

lyrics = ""
ch0 = ""
ch1 = ""
for _ in range(NUM_CHARS_PER_SONG):
    freq_map: Dict[str, int] = tri_freq_map[ch0][ch1]
    if len(freq_map) <= 1:
        freq_map = bi_freq_map[ch1]
    ch2 = WeightedSample(freq_map)
    if ch2:
        lyrics += ch2
    else:
        lyrics += "\n"
    ch0 = ch1
    ch1 = ch2
print("----")
print(lyrics)

lyrics = ""
ch0 = ""
ch1 = ""
ch2 = ""
for _ in range(NUM_CHARS_PER_SONG):
    freq_map: Dict[str, int] = quad_freq_map[ch0][ch1][ch2]
    if len(freq_map) <= 1:
        freq_map = tri_freq_map[ch1][ch2]
    if len(freq_map) <= 1:
        freq_map = bi_freq_map[ch2]
    ch3 = WeightedSample(freq_map)
    if ch3:
        lyrics += ch3
    else:
        lyrics += "\n"
    ch0 = ch1
    ch1 = ch2
    ch2 = ch3
print("----")
print(lyrics)
