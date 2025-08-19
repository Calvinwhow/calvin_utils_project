#!/usr/bin/env python3
import random, time, sys, os, argparse, multiprocessing, csv
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------
# constants
DEFAULT_LIST_LEN  = 12
DEFAULT_BLOCK_SEC = 5
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

# ---------- file logging -----------------------------------------------
def prompt_session_meta(args) -> tuple[str, str, Path, Path]:
    """Get patient/date/outdir; create file; return (patient, date, outdir, csv_path)."""
    patient = input("Patient name: ").strip()
    session_no = int(input("Session Number [1/2/3]: "))
    date_str = datetime.now().strftime("%Y-%m-%d")
    outdir = Path(input("Output directory: ").strip() or "./outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    # simple slug for filename
    safe_patient = "_".join(patient.split())
    csv_path = outdir / f"{date_str}_{safe_patient}_ses-{session_no}_delayed_recall.csv"

    # create file + header if new
    if not csv_path.exists():
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "trial_number","list_len","recalled_n","block_sec", "words_presented"
            ])
    return csv_path

def append_row(csv_path: Path, row: dict):
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            row["trial_number"],
            row["list_len"],
            row["recalled_n"],
            row["block_sec"],
            " ".join(row["words_presented"]),  # space-delimited for readability
        ])

# ---------- trial loop -------------------------------------------------
def run_single_trial(pool: list[str], list_len: int, block_sec: int):
    words = sample_words(pool, list_len)
    show_words(words, block_sec)
    arithmetic_distractor(block_sec)
    recalled = get_timed_input(words, block_sec)
    print(f"Patient recalled {recalled}/{list_len}")
    return recalled, list_len, words

def main():
    args   = parse_args()
    pool   = load_word_column(args.word_file)
    csv_path = prompt_session_meta(args)
    print(f"\nLogging to: {csv_path}")
    trial_no = 1
    while True:
        recalled, list_len, words = run_single_trial(pool, args.length, args.time)
        row = {"trial_number": trial_no,
               "list_len": list_len,
               "recalled_n": recalled,
               "block_sec": args.time,
               "words_presented": words}
        append_row(csv_path, row)
        trial_no += 1
        if input("\nAnother list?  <Enter=Yes | q=Quit> ").lower().startswith('q'):
            break

if __name__ == "__main__":
    main()