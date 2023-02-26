#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generates lyrics based on sample input."""

from typing import Dict, List, Tuple, TypeVar

import argparse
import collections
import io
import json
import math
import os
import random
import requests
import sys


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
            if ord(ch) < 128 or ch in "　！…、—○《》":  # Treat non-Chinese as separater.
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


def AdjustWeightByTemperature(
    freq_map: Dict[str, int], temperature: float
) -> Dict[str, float]:

    assert 0 <= temperature
    assert temperature <= 1

    # map: 0 => 10, 0.5 => 1, 1 => 0
    #   0.5 - t: 0 => 0.5, 0.5 => 0, 1 => -0.5
    #   1 - 2t: 0 => 1, 0.5 => 0, 1 => -1
    #   10^(1 - 2t): 0 => 10, 0.5 => 1, 1 => 0.1
    return {
        ch: math.pow(freq, math.pow(10, 1 - 2 * temperature))
        for ch, freq in freq_map.items()
    }


def WeightedSample(freq_map: Dict[str, int], temperature: float, top_p: float) -> str:
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
    filtered_freq_map = AdjustWeightByTemperature(
        {ch: freq for ch, freq in candidates}, temperature
    )
    total_weight: float = sum(filtered_freq_map.values())
    r = random.random() * total_weight
    start = 0
    for x, weight in filtered_freq_map.items():
        if start <= r and r < start + weight:
            return x
        start += weight
    return ""


def FloatFrom0To1(text: str) -> float:
    try:
        x = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text} is not a floating-point literal.")

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{text} is not in the range [0.0, 1.0].")
    return x


def GetChar(text: str, index: int) -> str:
    """Get the char at the given index, or "" if the index is invalid."""

    try:
        return text[index]
    except IndexError:
        return ""


def GenerateLyricsByGpt3(start: str, temperature: float, top_p: float) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit(
            "Please set the OPENAI_API_KEY environment variable to your API key first."
        )

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = json.dumps(
        {
            "model": "text-davinci-003",
            "prompt": "用罗大佑的风格写一首歌词。" + (f"用“{start}”开头，有副歌，不超过200字。" if start else ""),
            "max_tokens": 500,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
    )
    completion_endpoint = "https://api.openai.com/v1/completions"
    result = requests.post(completion_endpoint, headers=headers, data=data)
    lyrics = result.json()["choices"][0]["text"]
    print(json.loads(data)["prompt"])
    print(lyrics)


def main():
    # Parse the flags.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-g",
        "--gpt3",
        help="Use the GPT-3 API to generate lyrics.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        help="How wild the generator is (a float in [0, 1]).",
        type=FloatFrom0To1,
        default=0.7,
    )
    parser.add_argument(
        "-p",
        "--top_p",
        help="What ratio of the candidates are considered (a float in [0, 1]).",
        type=FloatFrom0To1,
        default=1,
    )
    parser.add_argument(
        "start",
        nargs="?",
        help="The start of the lyrics (the first several characters).",
        default="",
    )
    args = parser.parse_args()

    if args.gpt3:
        GenerateLyricsByGpt3(args.start, temperature=args.temperature, top_p=args.top_p)
        return

    random.seed()
    lines = NormalizeFileLines(LUO_DAYOU_LYRICS_FILE)
    uni_freq_map = BuildUnigramFrequencyMap(lines)
    bi_freq_map = BuildBigramFrequencyMap(lines)
    tri_freq_map = BuildTrigramFrequencyMap(lines)
    quad_freq_map = BuildQuadgramFrequencyMap(lines)

    lyrics = args.start
    for _ in range(NUM_CHARS_PER_SONG):
        ch = WeightedSample(
            uni_freq_map, temperature=args.temperature, top_p=args.top_p
        )
        if ch:
            lyrics += ch
        else:
            lyrics += "\n"
    print("----")
    print(lyrics)

    lyrics = args.start
    ch = GetChar(lyrics, -1)
    for _ in range(NUM_CHARS_PER_SONG):
        freq_map: Dict[str, int] = bi_freq_map[ch]
        ch = WeightedSample(freq_map, temperature=args.temperature, top_p=args.top_p)
        if ch:
            lyrics += ch
        else:
            lyrics += "\n"
    print("----")
    print(lyrics)

    lyrics = args.start
    ch0 = GetChar(lyrics, -2)
    ch1 = GetChar(lyrics, -1)
    for _ in range(NUM_CHARS_PER_SONG):
        freq_map: Dict[str, int] = tri_freq_map[ch0][ch1]
        if len(freq_map) <= 1:
            freq_map = bi_freq_map[ch1]
        ch2 = WeightedSample(freq_map, temperature=args.temperature, top_p=args.top_p)
        if ch2:
            lyrics += ch2
        else:
            lyrics += "\n"
        ch0 = ch1
        ch1 = ch2
    print("----")
    print(lyrics)

    lyrics = args.start
    ch0 = GetChar(lyrics, -3)
    ch1 = GetChar(lyrics, -2)
    ch2 = GetChar(lyrics, -1)
    for _ in range(NUM_CHARS_PER_SONG):
        freq_map: Dict[str, int] = quad_freq_map[ch0][ch1][ch2]
        if len(freq_map) <= 1:
            freq_map = tri_freq_map[ch1][ch2]
        if len(freq_map) <= 1:
            freq_map = bi_freq_map[ch2]
        ch3 = WeightedSample(freq_map, temperature=args.temperature, top_p=args.top_p)
        if ch3:
            lyrics += ch3
        else:
            lyrics += "\n"
        ch0 = ch1
        ch1 = ch2
        ch2 = ch3
    print("----")
    print(lyrics)


if __name__ == "__main__":
    main()
