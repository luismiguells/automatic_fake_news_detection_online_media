# -*- coding: utf-8 -*-
"""
Data splitting module for extracting words, emoticons, hashtags, mentions, links, and abbreviations.
This module processes a corpus text file and extracts various components into separate files.
"""

import re
import sys
import emoji
from pathlib import Path
from nltk.tokenize.casual import EMOTICON_RE as emo_re

# Add the parent directory of this script to sys.path to support utils import
sys.path.append(str(Path(__file__).resolve().parent.parent))

# URL matching regex
URL_REGEX_PATTERN = r"""
  (?:
  https?:
    (?:
      /{1,3}
      |
      [a-z0-9%]
    )
    |
    [a-z0-9.\-]+[.]
    (?:[a-z]{2,13})
    /
  )
  (?:
    [^\s()<>{}\[\]]+
    |
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\)
    |
    \([^\s]+?\)
  )+
  (?:
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\)
    |
    \([^\s]+?\)
    |
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]
  )
  |
  (?:
  	(?<!@)
    [a-z0-9]+
    (?:[.\-][a-z0-9]+)*
    [.]
    (?:[a-z]{2,13})
    \b
    /?
    (?!@)
  )
"""

def clean_text(line, hashtags, mentions, links):
    """
    Removes hashtags, mentions, and links from a line of text.

    Args:
        line (str): The original text line.
        hashtags (list): List of extracted hashtags.
        mentions (list): List of extracted mentions.
        links (list): List of extracted links.

    Returns:
        str: The cleaned text.
    """
    for link in links:
        line = line.replace(link, '')
    for hashtag in hashtags:
        line = line.replace('#' + hashtag, '').replace('＃' + hashtag, '')
    for mention in mentions:
        line = line.replace('@' + mention, '').replace('＠' + mention, '')
    return line

def read_abbreviations(file_path):
    """
    Reads abbreviations from a file into a dictionary for fast lookup.

    Args:
        file_path (Path): Path to the abbreviations file.

    Returns:
        dict: A dictionary of lowercase abbreviations.
    """
    abvs_dict = {}
    if not file_path.exists():
        return abvs_dict
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip().lower()
            if token:
                abvs_dict[token] = 1
    return abvs_dict

def process_corpus(corpus_file, abvs_list_file, output_dir):
    """
    Processes the corpus file and splits it into different components.

    Args:
        corpus_file (Path): Path to the input corpus file.
        abvs_list_file (Path): Path to the abbreviations list file.
        output_dir (Path): Directory where output files will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    abv_d = read_abbreviations(abvs_list_file)
    url_re = re.compile(URL_REGEX_PATTERN, re.VERBOSE | re.I | re.UNICODE)
    hashtag_re = re.compile('(?:^|\s)[＃#]{1}(\w+)', re.UNICODE)
    mention_re = re.compile('(?:^|\s)[＠@]{1}(\w+)', re.UNICODE)
    word_re = re.compile('[a-záéíóúñàèìòù][a-záéíóúñàèìòù_-]+')
    abv_find_re = re.compile('[a-z0-9ñáéíóú+/]+')

    files = {
        'words': open(output_dir / 'words.txt', 'w', encoding='utf-8'),
        'emo': open(output_dir / 'emoticons.txt', 'w', encoding='utf-8'),
        'hash': open(output_dir / 'hashtags.txt', 'w', encoding='utf-8'),
        'at': open(output_dir / 'ats.txt', 'w', encoding='utf-8'),
        'link': open(output_dir / 'links.txt', 'w', encoding='utf-8'),
        'abv': open(output_dir / 'abvs.txt', 'w', encoding='utf-8')
    }

    # Compatibility check for emoji library versions
    emoji_list = getattr(emoji, 'EMOJI_DATA', getattr(emoji, 'UNICODE_EMOJI_ENGLISH', {}))

    try:
        with open(corpus_file, 'r', encoding='utf-8') as text_r:
            for i, line in enumerate(text_r, 1):
                line = line.strip().lower()
                hashs = hashtag_re.findall(line)
                ats = mention_re.findall(line)
                links = url_re.findall(line)
                
                cleaned_line = clean_text(line, hashs, ats, links)
                
                emoticons = emo_re.findall(cleaned_line)
                emojis = [char for char in cleaned_line if char in emoji_list]
                words = word_re.findall(cleaned_line)
                abvs = [w for w in abv_find_re.findall(cleaned_line) if w in abv_d and len(w) > 1]

                files['words'].write(' '.join(words) + '\n')
                files['abv'].write(' '.join(abvs) + '\n')
                files['emo'].write(' '.join(emoticons + emojis) + '\n')
                files['hash'].write(' '.join(hashs) + '\n')
                files['at'].write(' '.join(ats) + '\n')
                files['link'].write(' '.join(links) + '\n')
                
                if i % 100 == 0:
                    print(f"Processed {i} lines...")
    finally:
        for f in files.values():
            f.close()

if __name__ == "__main__":
    # Example configuration for LIAR dataset valid split
    data_root = Path(__file__).resolve().parent.parent.parent / 'data'
    WORKING_DIR = data_root / 'liar_dataset'
    CORPUS_PATH = WORKING_DIR / 'corpus_valid.txt'
    ABVS_PATH = data_root / 'abvs' / 'abvs_english.txt'
    OUT_DIR = WORKING_DIR / 'split_valid'

    if CORPUS_PATH.exists():
        process_corpus(CORPUS_PATH, ABVS_PATH, OUT_DIR)
    else:
        print(f"Corpus file not found at {CORPUS_PATH}")
