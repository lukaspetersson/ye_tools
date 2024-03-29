##
import music21 as m21
import numpy as np
from collections import Counter, defaultdict
import glob
import time
import os
import matplotlib.pyplot as plt
import random
from scipy.stats import pearsonr, spearmanr

##
score = m21.converter.parse("~/ye_tools/datasets/maestro-v3.0.0/2006/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_04_Track04_wav.midi")

##
# Test to filter out notes and display them

def include_note(note):
        return float(note.duration.quarterLength) <= 4 and not (float(note.duration.quarterLength) != 4 and isinstance(note, m21.note.Rest))

stream = m21.stream.Stream()
for part in score.parts:
    s = m21.stream.Stream()
    for note in part.notesAndRests:
        if include_note(note) or True:
            s.append(note)
    stream.append(s)

stream.show()
#stream.write()

##
# Dataset has lmd has 180k songs
# Maestro has 1276

len(glob.glob("/home/lukas/ye_tools/datasets/maestro-v3.0.0/**/*.midi", recursive=True))

##
# Display what python objects are in the score

types = set()
for el in score.recurse():
    types.add(type(el))
types

## 
# analyze key signature
# use analyze() even if it is slow, other strategy does not allways work

#method 1
t0 = time.time()
k1 = score.analyze('key')
print(time.time()-t0)

#method 2
t0 = time.time()
k2 = None
for el in score.flatten():
    if isinstance(el, m21.key.KeySignature):
        k2 = el
        break
print(time.time()-t0)

print(k1, k2)

##
# Load 30 songs to do analysis on

#lmd:
# parse is SOMETIMES really slow (range: 0.1 - 60s)
# TODO: timeout ceirtain songs

# maestro:
# parse mean = 9s, max=20 (No need for filter) 
# parsing should be 3h without parallel
# key and krumhansl allways below 1s

# second try with maestro:
# mean = 0.8, cashe?

# TODO: parallelize data analysis

t_parse = []
scores = []
for i, file in enumerate(glob.glob("/home/lukas/ye_tools/datasets/maestro-v3.0.0/**/*.midi", recursive=True)):
    if i > 30: break

    t0=time.time()
    s = m21.converter.parse(file)
    t_parse.append(time.time()-t0)
    scores.append(s)

print(t_parse)

##

# teqniques give different resutls
# however, not relevant if we anyway transpose the song as data augmentation

t_krum = []
t_key= []
for s in scores:
    t0=time.time()
    k1 = s.analyze('Krumhansl')
    t_krum.append(time.time()-t0)

    t0=time.time()
    k2 = s.analyze('key')
    t_key.append(time.time()-t0)

    print(k1, k2)
    print(k1 == k2)

print(t_krum)
print(t_key)

##
for s in scores:
    for note in s.parts[0].flatten().notes:
        print(note.duration.quarterLength)
        break
##
# investigate if most songs has acceptable durations
# 1/2 has only acceptable, 1/4 has many 1/3 notes, 1/4 has a few >4 notes 
# 75% should be useable

tot_parts = 0
useable_parts = 0

acceptable_durations = set([i/12 for i in range(12*4+1)])
for s in scores:
    #maestro
    part = s.parts[0]
    tot_parts += 1
    c = 0
    for note in s.parts[0].flatten().notes:
        if float(note.duration.quarterLength) not in acceptable_durations:
            c += 1
    print(c)
    if c < 20:
        useable_parts += 1

    #lmd
    #for part in s.parts:
     #   tot_parts += 1
      #  c = 0
       # for note in part.notes:
        #    print(note.duration.quarterLength)
         #   if float(note.duration.quarterLength) not in acceptable_durations:
          #      c += 1
        #if c < 20:
         #   useable_parts += 1
print(useable_parts, tot_parts)

##
#If the value is an integer, the transposition is treated in half steps
score = scores[0]
print(score.parts[0].flatten().notes[0].pitch.midi)
score = score.transpose(3)
print(score.parts[0].flatten().notes[0].pitch.midi)

##
# it is not slower to transpose more, always 2-5s
score = scores[0]
for i in range(5):
    t0=time.time()
    score.transpose(i)
    print(time.time()-t0)

##
# transpose to key C major or A minor

k = score.analyze('key')
p = "C" if k.mode == "major" else "A"
i = m21.interval.Interval(k.tonic, m21.pitch.Pitch(p))
score = score.transpose(i)
print("Transformed from key:", k, "to:", score.analyze('key'))

##
# Count durations of each note type 
# Most are either multiple of 1/4 or 1/3

durations = defaultdict(list)
for s in scores:
    for el in s.recurse().notesAndRests:
        if isinstance(el, m21.note.Note):
            durations["note"].append(float(el.duration.quarterLength))
        elif isinstance(el, m21.chord.Chord):
            durations["chord"].append(float(el.duration.quarterLength))
        elif isinstance(el, m21.note.Rest):
            durations["rest"].append(float(el.duration.quarterLength))

print("Note durations")
print(sorted(Counter(durations["note"]).items(), key=lambda x: -x[1]))
print("Chord durations")
print(sorted(Counter(durations["chord"]).items(), key=lambda x: -x[1]))
print("Rest durations")
print(sorted(Counter(durations["rest"]).items(), key=lambda x: -x[1]))

##
import signal

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

##
# One song object has many part objects
# For now we only care about notes (not rests and chords)

class Part():
    def __init__(self, part):
        self.part = part
        #TODO: how to represent chords and rests??
        self.notes = m21.stream.Stream([note for note in part.recurse().notesAndRests if isinstance(note, m21.note.Note)])

    def get_instrument_type(self):
        return type(self.part.getInstrument())

    '''
    not usefull when we only have notes
    def count_note_types(self):
        n, r, c = 0, 0, 0
        for note in self.notes:
            if isinstance(note, m21.note.Note): n += 1
            elif isinstance(note, m21.note.Rest): r += 1
            elif isinstance(note, m21.chord.Chord): c += 1
        return (n, r, c)
    '''

    def count_durations(self):
        return Counter([note.duration.quarterLength for note in self.notes])

    # Represent the part as a list of tuples (pitch, duration)
    def as_tuples(self):
        vec = np.zeros((len(self.notes), 2), int)
        for i, note in enumerate(self.notes):
            dur = int(note.duration.quarterLength*4)
            vec[i][0] = note.pitch.midi
            vec[i][1] = dur
        return vec

'''
Other ways to encode the part, not in use

    def one_minus_hot_encode(self):
        song = np.zeros((128,), int)
        for note in self.notes:
            dur = int(note.duration.quarterLength*4)
            if dur > 16:
                continue
            for i in range(dur):
                vec = np.zeros((128,), int)
                if isinstance(note, m21.note.Note):
                    vec[note.pitch.midi] = 1 if i==0 else -1
                    song = np.vstack([song, vec])
                elif isinstance(note, m21.note.Rest):
                    song = np.vstack([song, vec])
                #TODO: support chords
        song = np.delete(song, 0, 0)
        return song

    def dur_hot_encode(self):
        song = np.zeros((len(self.notes), 128), int)
        for i, note in enumerate(self.notes):
            dur = int(note.duration.quarterLength*4)
            if isinstance(note, m21.note.Note):
                song[i][note.pitch.midi] = dur
            elif isinstance(note, m21.note.Rest):
                pass
            elif isinstance(note, m21.chord.Chord):
                for p in note.pitches:
                    song[i][p.midi] = dur
        return song
'''

class Song():
    def __init__(self, path, transpose=True):
        try:
            #with timeout(seconds=59):
             #   self.score = m21.converter.parse(path)
            self.score = m21.converter.parse(path)
            if transpose:
                #TODO: different parts have different keys, but they shouldnt
                #TODO: too slow to transpose all songs
                k = self.score.analyze("key")
                p = "C" if k.mode == "major" else "A"
                i = m21.interval.Interval(k.tonic, m21.pitch.Pitch(p))
                self.score = self.score.transpose(i)

            self.parts = [Part(part) for part in self.score.parts]
        except:
            self.parts = []

        self.excluded_parts = set()
    
    def _include(self, i):
        self.excluded_parts.remove(i)

    def _exclude(self, i):
        self.excluded_parts.add(i)

    # Exclude part based on properties of the part
    # return number of exclusions
    def part_exclusion(self, func):
        count = 0
        for i, part in enumerate(self.parts):
            if func(part):
                self._exclude(i)
                count += 1
        return count

    # Exclude part based on properties of the part's notes
    def note_exclusion(self, func):
        count = 0
        for i, part in enumerate(self.parts):
            if any(map(func, part.notes)):
                self._exclude(i)
                count += 1
        return count

    # Remove notes from a part
    def filter_notes(self, func):
        count = 0
        for i, part in enumerate(self.parts):
            for note in part.notes:
                if func(note):
                    idx = self.parts[i].notes.index(note)
                    self.parts[i].notes.pop(idx)
                    count += 1
        return count
    
    def to_stream(self):
        s = m21.stream.Stream()
        for i, part in enumerate(self.parts):
            if i not in self.excluded_parts: 
                s.append(part.notes)
        return s

    def get_part(self, i):
        return self.parts[i]

    def show(self, fmt="musicxml", i=-1):
        if i > -1:
            self.parts[i].notes.show(fmt)
        else:
            self.to_stream().show(fmt)

    def as_vector(self):
        return [part.as_tuples() for i, part in enumerate(self.parts) if i not in self.excluded_parts]

##

class MaestroSong():
    def __init__(self, path, transpose=True):
        try:
            with timeout(seconds=59):
                self.part = m21.converter.parse(path).parts[0]
                self.notes = m21.stream.Stream([note for note in self.part.recurse().notesAndRests if isinstance(note, m21.note.Note)])
        except Exception as e:
            print("Error loading song", e)
            self.notes = m21.stream.Stream([])
    
    def as_tuples(self):
        vec = np.zeros((len(self.notes), 2), int)
        for i, note in enumerate(self.notes):
            dur = int(note.duration.quarterLength*12)
            vec[i][0] = note.pitch.midi
            vec[i][1] = dur
        return [vec]

    def show(self, fmt="musicxml", i=-1):
        self.part.show(fmt)

##
song = MaestroSong("/home/lukas/ye_tools/datasets/maestro-v3.0.0/2008/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1.midi", transpose=False)

##

# filter notes with duration >4 and <0.25
print(song.filter_notes(lambda note: not (0.25 <= float(note.duration.quarterLength) <= 4)))

# filter out non 12th durations
print(song.filter_notes(lambda note:  float(note.duration.quarterLength) not in [i/12 for i in range(1, 12*4+1)]))

# filter out instruments
#print(song.part_exclusion(lambda part: part.get_instrument_type() == m21.instrument.Sampler))

# filter out parts with too much repetition
print(song.part_exclusion(lambda part: np.std([int(note.pitch.midi) for note in part.notes]) < 5))

print(song.excluded_parts, "len: "+str(len(song.parts)))

##
# Detect too much repetition
# Print typical values for the standard deaviation of the part's pitches
# 2 seem to be a good value to filter on

for i, file in enumerate(glob.glob("data/lmd_full/0/**/*.mid", recursive=True)):
    if i > 5: break
    song = Song(file, transpose=False)
    for part in song.parts:
        print(np.std([int(note.pitch.midi) for note in part.notes]))
        print(Counter([int(note.pitch.midi) for note in part.notes])) 

##
song = Song("data/lmd_full/3/3003bbf06bec7c8ff1add82b50a84ae4.mid", transpose=False)
song.as_vector()

##
# Transpose is really slow 13s
t0 = time.time()
song = Song("data/lmd_full/3/3003bbf06bec7c8ff1add82b50a84ae4.mid", transpose=False)
print(time.time()-t0)

t0 = time.time()
song = Song("data/lmd_full/3/3003bbf06bec7c8ff1add82b50a84ae4.mid")
print(time.time()-t0)

##
# Build dataset as list of parts
# TODO: loses info that parts can share properties if they are from the same song
vecs = []
for i, file in enumerate(glob.glob("data/lmd_full/**/*.mid", recursive=True)):
    if i%100 == 0 and i > 0:
        np.save("data/data_big_"+str(i)+".npy", np.asarray(vecs), allow_pickle=True)
        vecs = []
    song = Song(file, transpose=False)
    # same filters as in test above
    song.filter_notes(lambda note: not (0.25 <= float(note.duration.quarterLength) <= 4))
    song.filter_notes(lambda note:  float(note.duration.quarterLength) not in [i/4 for i in range(1, 17)])
    song.part_exclusion(lambda part: part.get_instrument_type() == m21.instrument.Sampler)
    song.part_exclusion(lambda part: np.std([int(note.pitch.midi) for note in part.notes]) < 2)

    vecs += song.as_vector()

##
# Test loading dataset
data = np.array(np.concatenate([np.load('data/data_big_'+str(i*100)+'.npy', allow_pickle=True) for i in range(1,3)]))
data = np.transpose(np.concatenate(data))
print(len(data[0]))

##
plt.hist2d(data[1], data[0], bins=20)
cb = plt.colorbar()
cb.set_label('Number of notes')
plt.title('2D histogram of notes')
plt.xlabel('Duration')
plt.ylabel('Pitch')

##
spearmanr(data[1], data[0])[0]


##
v = np.load('/home/lukas/ye_tools/data_gen/pitch_dur_tuple/output/tuple_based_big/data_big_100.npy', allow_pickle=True)

##
v

##
twinkle = Song('../../Downloads/twinkle-twinkle-little-star.mid', transpose=False)
np.save("data/twinkle_note_based.npy", np.asarray(twinkle.as_vector()), allow_pickle=True)

##
# 2h
t0=time.time()
files = glob.glob("/home/lukas/ye_tools/datasets/maestro-v3.0.0/**/*.midi", recursive=True)
vecs = []
for i, file in enumerate(files):
    print(i)
    if i%50 == 0:
        print(i, time.time()-t0)
    song = MaestroSong(file)
    vecs+=song.as_tuples()
print("Time:", time.time()-t0)

##
np.save("maestro_data.npy", np.array(vecs, dtype=object), allow_pickle=True)
print("saved")

## 
v = np.load("/home/lukas/maestro_data.npy", allow_pickle=True)
print("loaded")
