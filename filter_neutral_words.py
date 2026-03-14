#!/usr/bin/env python3
"""
Filter a word list to neutral, simple words: length, VADER sentiment, word
frequency (wordfreq), max syllables (pysyllables), and optionally nouns only
(NLTK WordNet). Writes one word per line to words.txt.

Requires: pip install vaderSentiment wordfreq pysyllables nltk
First run with --nouns-only: python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
"""
import argparse
import sys
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordfreq import zipf_frequency
from pysyllables import get_syllable_count

NEUTRAL_LOW = -0.5
NEUTRAL_HIGH = 0.5
DEFAULT_MIN_LENGTH = 3
DEFAULT_MAX_LENGTH = 9
DEFAULT_MIN_ZIPF = 4.0
DEFAULT_MAX_SYLLABLES = 2


def load_vader_lexicon() -> dict[str, float]:
    """Load VADER lexicon once: word -> mean sentiment score (-4 to +4)."""
    analyzer = SentimentIntensityAnalyzer()
    return getattr(analyzer, "lexicon", {})


def is_neutral(word: str, lexicon: dict[str, float]) -> bool:
    """True if word is not in VADER lexicon, or its score is in the neutral band."""
    w = word.lower()
    if w not in lexicon:
        return True
    score = lexicon[w]
    return NEUTRAL_LOW <= score <= NEUTRAL_HIGH


def ensure_wordnet() -> None:
    """Download NLTK wordnet and omw-1.4 if not already present."""
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)


def is_noun(word: str) -> bool:
    """True if word has at least one noun synset in WordNet."""
    from nltk.corpus import wordnet as wn
    w = word.lower()
    return len(wn.synsets(w, wn.NOUN)) > 0


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "english-words" / "words_alpha.txt"
    default_output = script_dir / "words.txt"

    parser = argparse.ArgumentParser(
        description="Filter word list to neutral, simple words (length, VADER, wordfreq, syllables, nouns)."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=default_input,
        help=f"Input word list (one word per line). Default: {default_input}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_output,
        help=f"Output file. Default: {default_output}",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=DEFAULT_MIN_LENGTH,
        help=f"Minimum word length (default: {DEFAULT_MIN_LENGTH})",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Maximum word length (default: {DEFAULT_MAX_LENGTH})",
    )
    parser.add_argument(
        "--min-zipf",
        type=float,
        default=DEFAULT_MIN_ZIPF,
        help=f"Minimum Zipf frequency for common words (default: {DEFAULT_MIN_ZIPF})",
    )
    parser.add_argument(
        "--max-syllables",
        type=int,
        default=DEFAULT_MAX_SYLLABLES,
        help=f"Maximum syllable count (default: {DEFAULT_MAX_SYLLABLES})",
    )
    parser.add_argument(
        "--nouns-only",
        action="store_true",
        default=True,
        help="Keep only words that are nouns in WordNet (default: True)",
    )
    parser.add_argument(
        "--no-nouns-only",
        action="store_false",
        dest="nouns_only",
        help="Do not filter by part of speech",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.nouns_only:
        ensure_wordnet()

    lexicon = load_vader_lexicon()
    kept: list[str] = []
    seen: set[str] = set()

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if not word:
                continue
            if len(word) < args.min_length or len(word) > args.max_length:
                continue
            if not is_neutral(word, lexicon):
                continue
            if zipf_frequency(word, "en") < args.min_zipf:
                continue
            syll = get_syllable_count(word)
            if syll is None or syll > args.max_syllables:
                continue
            if args.nouns_only and not is_noun(word):
                continue
            if word in seen:
                continue
            seen.add(word)
            kept.append(word)

    kept.sort()
    args.output.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"Wrote {len(kept)} neutral simple words to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
