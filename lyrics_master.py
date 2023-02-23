#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generates lyrics based on sample input."""

from typing import Dict, List, Tuple, TypeVar

import collections
import io
import math
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
        ch2 = ""
        for ch2 in line:
            map[ch0][ch1][ch2] += 1
            ch0 = ch1
            ch1 = ch2
        map[ch1][ch2][""] += 1
        map[ch2][""][""] += 1
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


T = TypeVar("T")


def PickTopP(xs: List[T], top_p: float) -> List[T]:
    num_to_keep = math.ceil(top_p * len(xs))
    if num_to_keep <= 0:
        num_to_keep = 1
    return xs[:num_to_keep]


def WeightedSample(
    freq_map: Dict[str, int], temperature: float = 0.5, top_p: float = 0.3
) -> str:
    """Picks one char randomly.

    Args:
        freq_map: map from a char to its frequency
        temperature: a float in [0, 1] that determines how wild the pick can be.
            0 means that we will always pick the char with the highest frequency.
            0.5 means that the probability of a char being picked is proportional
            to its frequency in the map.
            1 means that all chars in `freq_map` are considered with equal probability.
        top_p: a float in [0, 1] that determines how tolerant the pick is.
            0 means that only the best choice is considered.
            0.5 means the best half of the valid choices are considered.
            1 means that all valid choices are considered.
    """

    assert 0 <= temperature
    assert temperature <= 1
    assert 0 <= top_p
    assert top_p <= 1

    # Sort the entries by frequency, descending.
    sorted_list = sorted(
        freq_map.items(), key=lambda ch_and_freq: ch_and_freq[1], reverse=True
    )
    candidates = PickTopP(sorted_list, top_p)
    filtered_freq_map = {ch: freq for ch, freq in candidates}
    total_count = sum(filtered_freq_map.values())
    i = random.randrange(total_count)
    start = 0
    for x, count in filtered_freq_map.items():
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
