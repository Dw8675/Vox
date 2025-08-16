# generate_duo_intro_with_music.py
# Two-host (female + male) TTS with background drone + intro sting + auto ducking.
# Output: truecrime_intro_duo_master.wav

import numpy as np
import soundfile as sf
from TTS.api import TTS

# -----------------------
# Settings
# -----------------------
SAMPLE_RATE = 22050
PAUSE_SECS = 0.6          # gap between lines
DUCK_GAIN_SPEECH = 0.35   # music level under speech (0.0 = silent, 1.0 = no duck)
DUCK_FADE_MS = 120        # smooth edges of duck (ms)
MASTER_PEAK_DBFS = -1.0   # final peak normalize target
DRONE_GAIN = 0.15         # overall drone loudness
STING_GAIN = 0.25         # intro sting loudness
STING_LEN = 2.0           # seconds

MODEL_ID = "tts_models/en/vctk/vits"  # multi-speaker English
FEMALE_SPEAKER = "p273"
MALE_SPEAKER   = "p240"

SCRIPT_LINES = [
    ("host1", "Welcome back to the show where the only thing thinner than the alibi is my patience. Tonight’s case proves two eternal truths: people are terrible at secrets, and small towns are excellent at gossip. Our story has a quiet street, a creaky porch light, and just enough bad decisions to qualify as a group project."),
    ("host2", "You know it’s bad when even the dog won’t bark—like it took one look at the suspect and said, 'Nope, I’m clocking out.' This is a story with more red flags than a clearance rack at a closing store, and somehow everyone still missed them."),
    ("host1", "The timeline? Sloppy. The motives? Piled higher than the detective’s inbox. Witnesses heard a scream, saw a shadow, forgot their glasses, and suddenly remembered everything the moment a reward was announced. It’s like the universe said, 'Let’s run a social experiment,' and the town said, 'Say less.'"),
    ("host2", "Our person of interest had the subtlety of a marching band—changing stories, midnight yard work, and a sudden fascination with bleach. If you’re scrubbing concrete at 2 a.m., you’re either fixing a murder or auditioning for the world’s saddest cleaning commercial."),
    ("host1", "And the investigation? Part police work, part improv. Evidence wandered off. Files were misplaced. At one point the case map looked less like a strategy and more like a conspiracy theory made by yarn and anxiety."),
    ("host2", "But buried under the noise—there is a pattern. Someone who knew routines, knew when the street went quiet, and knew exactly how long it takes for fear to sound like silence. This wasn’t random. It was practiced."),
    ("host1", "Tonight, we’ll pull the thread—timeline, suspects, the weird phone call that didn’t make the news, and why one receipt might matter more than a dozen interviews."),
    ("host2", "So lock your doors, dim the lights, and tell yourself that shadow in the hallway is just bad interior design. Because this case has jokes—dark ones—but the punchline is brutal: somebody didn’t come home, and somebody thinks they got away with it."),
    ("host1", "Cue the theme. And if you hear footsteps behind you… it’s probably just us, walking you through a crime scene one bad decision at a time. Welcome to tonight’s episode.")
]

# -----------------------
# Audio helpers
# -----------------------
def db_to_lin(db):
    return 10 ** (db / 20.0)

def lin_to_db(lin):
    lin = max(lin, 1e-12)
    return 20.0 * np.log10(lin)

def fade_in(x, ms, sr):
    n = int(ms * 1e-3 * sr)
    if n <= 1: return x
    ramp = np.linspace(0.0, 1.0, n)
    x[:n] *= ramp
    return x

def fade_out(x, ms, sr):
    n = int(ms * 1e-3 * sr)
    if n <= 1: return x
    ramp = np.linspace(1.0, 0.0, n)
    x[-n:] *= ramp
    return x

def normalize_peak(x, target_dbfs=-1.0):
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak < 1e-6:
        return x
    target_lin = db_to_lin(target_dbfs)
    return x * (target_lin / peak)

def seconds_to_samples(sec, sr):
    return int(round(sec * sr))

# -----------------------
# Music generators
# -----------------------
def make_drone(duration_s, sr, gain=0.15):
    t = np.linspace(0, duration_s, seconds_to_samples(duration_s, sr), endpoint=False)
    # Low, detuned sines for unease (A2-ish)
    f1, f2 = 110.0, 111.2
    # Add a faint higher harmonic/air
    f3 = 220.0
    drone = (
        0.7 * np.sin(2*np.pi*f1*t) +
        0.7 * np.sin(2*np.pi*f2*t) +
        0.25 * np.sin(2*np.pi*f3*t)
    )
    # Gentle slow amplitude wobble
    lfo = 0.85 + 0.15 * np.sin(2*np.pi*0.08*t)
    out = (drone * lfo).astype(np.float32)
    out = normalize_peak(out, -8.0) * gain
    # long fades to avoid clicks
    fade_in(out, 800, sr)
    fade_out(out, 1200, sr)
    return out

def make_sting(duration_s, sr, gain=0.3):
    # Minor chord hit -> A minor: A3 (220), C4 (261.63), E4 (329.63)
    t = np.linspace(0, duration_s, seconds_to_samples(duration_s, sr), endpoint=False)
    freqs = [220.0, 261.63, 329.63]
    env = np.exp(-3.0 * t)  # percussive decay
    sig = sum(np.sin(2*np.pi*f*t) for f in freqs) / len(freqs)
    # touch of noise for grit
    noise = (np.random.randn(sig.size).astype(np.float32) * 0.02)
    out = (sig.astype(np.float32) * env + noise)
    out = normalize_peak(out, -6.0) * gain
    fade_in(out, 10, sr)
    fade_out(out, 400, sr)
    return out

def smooth_duck_envelope(length, sr, regions, duck_gain=0.35, edge_ms=120):
    """
    length: total samples
    regions: list of (start_sample, end_sample) to duck
    returns per-sample gain array
    """
    env = np.ones(length, dtype=np.float32)
    edge = seconds_to_samples(edge_ms/1000.0, sr)
    for (s, e) in regions:
        s0 = max(0, s - edge)
        e0 = min(length, e + edge)
        env[s0:e0] = duck_gain
        # crossfade edges
        if s0 < s:
            seg = s - s0
            env[s0:s] = np.linspace(1.0, duck_gain, seg, dtype=np.float32)
        if e < e0:
            seg = e0 - e
            env[e:e0] = np.linspace(duck_gain, 1.0, seg, dtype=np.float32)
    return env

# -----------------------
# TTS synthesis
# -----------------------
def synth_script(tts, script, sr, female_id, male_id, pause_secs):
    pause = np.zeros(seconds_to_samples(pause_secs, sr), dtype=np.float32)
    chunks = []
    speech_regions = []  # list of (start_sample, end_sample) for ducking
    cursor = 0

    for role, line in script:
        speaker = female_id if role == "host1" else male_id
        wav = tts.tts(text=line, speaker=speaker)
        wav = np.array(wav, dtype=np.float32)
        # mark region on the timeline
        start = cursor
        end = start + wav.size
        speech_regions.append((start, end))
        chunks.append(wav)
        chunks.append(pause)
        cursor = end + pause.size

    if chunks:
        voice = np.concatenate(chunks)
    else:
        voice = np.zeros(1, dtype=np.float32)

    return voice, speech_regions

# -----------------------
# Main
# -----------------------
def main():
    print("Loading TTS model… (CPU mode)")
    tts = TTS(model_name=MODEL_ID, progress_bar=False, gpu=False)

    print("Synthesizing dialogue…")
    voice, regions = synth_script(
        tts, SCRIPT_LINES, SAMPLE_RATE,
        FEMALE_SPEAKER, MALE_SPEAKER, PAUSE_SECS
    )

    total_sec = voice.size / SAMPLE_RATE
    print(f"Dialogue length: {total_sec:.2f}s")

    print("Generating background drone…")
    drone = make_drone(total_sec, SAMPLE_RATE, gain=DRONE_GAIN)

    print("Generating intro sting…")
    sting = make_sting(STING_LEN, SAMPLE_RATE, gain=STING_GAIN)

    # Build music bed timeline (drone full-length + sting at start 0-2s)
    music = drone.copy()
    L = music.size
    slen = sting.size
    music[:min(L, slen)] += sting[:min(L, slen)]

    # Auto-duck music under speech
    print("Applying ducking…")
    env = smooth_duck_envelope(L, SAMPLE_RATE, regions,
                               duck_gain=DUCK_GAIN_SPEECH, edge_ms=DUCK_FADE_MS)
    music_ducked = music * env

    # Mix: voice (foreground) + music_ducked (background)
    print("Mixing + normalizing…")
    mix = np.zeros(max(L, voice.size), dtype=np.float32)
    mix[:voice.size] += voice
    mix[:music_ducked.size] += music_ducked

    # Gentle master peak normalize to target
    out = normalize_peak(mix, MASTER_PEAK_DBFS)

    outfile = "truecrime_intro_duo_master.wav"
    sf.write(outfile, out, SAMPLE_RATE)
    print(f"Done: {outfile}")

if __name__ == "__main__":
    main()
