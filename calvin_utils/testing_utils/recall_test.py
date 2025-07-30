#!/usr/bin/env python3
import random, time, sys, os, argparse, concurrent.futures, multiprocessing, _thread
from pathlib import Path
import threading
# ---------------------------------------------------------------------
# constants
DEFAULT_LIST_LEN  = 12
DEFAULT_BLOCK_SEC = 30
WORD_FILE         = Path(__file__).with_name("nouns.txt")   # nouns.txt in same dir
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Delayed free recall CLI")
    p.add_argument("-l", "--length", type=int, default=DEFAULT_LIST_LEN,
                   help=f"Words per list (default {DEFAULT_LIST_LEN})")
    p.add_argument("-t", "--time", type=int, default=DEFAULT_BLOCK_SEC,
                   help=f"Seconds per block (default {DEFAULT_BLOCK_SEC})")
    p.add_argument("--word-file", type=Path, default=WORD_FILE,
                   help="Path to word pool text file")
    return p.parse_args()

# ---------- I/O helpers ------------------------------------------------
def say(msg: str = "done"):
    """Cross platform text to speech."""
    if sys.platform.startswith("darwin"):
        os.system(f'say "{msg}"')
    elif sys.platform.startswith("linux"):
        os.system(f'espeak "{msg}"')        # or spd‑say
    elif sys.platform.startswith("win"):
        import pyttsx3
        e = pyttsx3.init();  e.say(msg);  e.runAndWait()

def load_word_column(path: Path, delim: str = '\t') -> list[str]:
    with path.open() as f:
        return [ln.split(delim, 1)[0].strip().upper()
                for ln in f
                if ln and ln[0].isalpha()]      # skip headers / commentary

def sample_words(pool: list[str], n: int) -> list[str]:
    return random.sample(pool, n)
# ---------- distractor helpers -----------------------------------------
def question():
    stop = False
    while stop==False:
        try:
            a, b, c = (random.randint(2, 9) for _ in range(3))
            answer = input(f"{a}+{b}+{c}=? ")
        except EOFError:
            print("EOFError encountered. Generating synthetic tests (cannot accept inputs): ")
            for i in range(10):
                a, b, c = (random.randint(2, 9) for _ in range(3))
                print(f"{a}+{b}+{c}=? ")
            break  # gracefully exit when stdin is unavailable (as in some terminals)

# ---------- task blocks ------------------------------------------------
def show_words(words: list[str], sec: int):
    print("\n*** ENCODING PERIOD ***")
    print(" ".join(words))
    print(f"(encoding for {sec}s)")
    time.sleep(sec)
    say("time")

def arithmetic_distractor(sec: int = 30):
    print("\n*** CONSOLIDATION PERIOD ***")    
    process = multiprocessing.Process(target = question)
    process.start()
    time.sleep(sec)
    process.terminate()
    process.join()
    say("time")                               # audible cue

def get_timed_input(words_list: list[str], sec: int) -> int:
    print("\n*** RECALL PERIOD ***")
    print("Presented words:\n", " ".join(words_list))
    print(f"(recall window {sec}s)")
    time.sleep(sec)
    say("time")
    try:
        return int(input("How many correct? > ").strip() or 0)
    except ValueError:
        return 0

# ---------- trial loop -------------------------------------------------
def run_single_trial(pool: list[str], list_len: int, block_sec: int):
    words = sample_words(pool, list_len)
    show_words(words, block_sec)
    arithmetic_distractor(block_sec)
    recalled = get_timed_input(words, block_sec)
    print(f"Patient recalled {recalled}/{list_len}")

def main():
    args   = parse_args()
    pool   = load_word_column(args.word_file)
    while True:
        run_single_trial(pool, args.length, args.time)
        if input("\nAnother list?  <Enter=Yes | q=Quit> ").lower().startswith('q'):
            break

if __name__ == "__main__":
    main()