# cognitive-shuffle

Speaks a random word from `words.txt` every 5–6 seconds using `pyttsx3`, which talks through the system speech engine directly.

**Requirements:** your normal `python3` is fine on this machine. No `ffmpeg`, `ffprobe`, or PortAudio setup is needed.

```bash
pip install -r requirements.txt
python3 cognitive_shuffle.py
```

Generate `words.txt` with `python3 filter_neutral_words.py` (see script for options). Filter script uses the other dependencies in `requirements.txt`; the main script uses `pyttsx3`.
