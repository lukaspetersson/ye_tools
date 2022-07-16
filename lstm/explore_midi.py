##
import music21 as m21
import numpy as np
from collections import Counter, defaultdict
import glob
import time
import os

##
# Test to filter out notes and display them

score = m21.converter.parse("data/lmd_full/a/a00b0f5acc0fd4e4fc9f32830d61978d.mid")
def include_note(note):
        return float(note.duration.quarterLength) <= 4 and not (float(note.duration.quarterLength) != 4 and isinstance(note, m21.note.Rest))

stream = m21.stream.Stream()
for part in score.parts:
    s = m21.stream.Stream()
    for note in part.notesAndRests:
        if include_note(note):
            s.append(note)
    stream.append(s)

stream.show()
#stream.write()

##
# Dataset has 180k songs

len(glob.glob("data/lmd_full/**/*.mid", recursive=True))

##
# Display what python objects are in the score

types = set()
for el in score.recurse():
    types.add(type(el))
types

## 
# analyze key signature

# use analyze() even if it is slow, other strategy does not allways work
t0 = time.time()
#method 1
k = score.analyze('key')
print(time.time()-t0)
t0 = time.time()
#method 2
for el in score.recurse():
    if isinstance(el, m21.key.Key):
        break
print(time.time()-t0)
print(k, el)

##
# The 2 methods to get key signature dont produce same results

for i, file in enumerate(glob.glob("data/adl-piano-midi/**/*.mid", recursive=True)):
    if i > 5: break
    s = m21.converter.parse(file)
    #method 1
    k = s.analyze('key')
    #method 2
    for el in s.recurse():
        if isinstance(el, m21.key.Key):
            print(el, k, file)
    
##
# investigate if most songs has acceptable durations
# 1/2 has only acceptable, 1/4 has many 1/3 notes, 1/4 has a few >4 notes 
# 75% should be useable

count = 0
acceptable_durations = set([i/4 for i in range(1, 17)])
for i, file in enumerate(glob.glob("data/lmd_full/3/**/*.mid", recursive=True)):
    if i > 10: break
    s = m21.converter.parse(file)
    c = []
    for note in s.parts[0].notesAndRests:
        if float(note.duration.quarterLength) not in acceptable_durations:
            c.append(float(note.duration.quarterLength))
    if len(c) > 0:
        count+=1
        print(c)
        print(file)
print(count)

##
# transpose to key C major or A minor
# SUPER SLOW

k = score.analyze('key')
p = "C" if k.mode == "major" else "A"
i = m21.interval.Interval(k.tonic, m21.pitch.Pitch(p))
score = score.transpose(i)
print("Transformed from key:", k, "to:", score.analyze('key'))

##
# Count durations of each note type 
# Most are either multiple of 1/4 or 1/3

durations = defaultdict(list)
for i, file in enumerate(glob.glob("data/lmd_full/0/**/*.mid", recursive=True)):
    if i > 5: break
    s = m21.converter.parse(file)
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
            self.score = m21.converter.parse(path)
            if transpose:
                #TODO: different parts have different keys, but they shouldnt
                #TODO: too slow to transpose all songs
                k = self.score.analyze("key")
                p = "C" if k.mode == "major" else "A"
                i = m21.interval.Interval(k.tonic, m21.pitch.Pitch(p))
                self.score = self.score.transpose(i)

            self.parts = [Part(part) for part in self.score.parts]
        except m21.midi.MidiException:
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
song = Song("data/lmd_full/3/3003bbf06bec7c8ff1add82b50a84ae4.mid", transpose=False)

# filter notes with duration >4 and <0.25
print(song.filter_notes(lambda note: not (0.25 <= float(note.duration.quarterLength) <= 4)))

# filter out non 4ths
print(song.filter_notes(lambda note:  float(note.duration.quarterLength) not in [i/4 for i in range(1, 17)]))

# filter out instruments
print(song.part_exclusion(lambda part: part.get_instrument_type() == m21.instrument.Sampler))

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
for i, file in enumerate(glob.glob("data/lmd_full/0/**/*.mid", recursive=True)):
    if i > 10: break
    song = Song(file, transpose=False)
    # same filters as in test above
    song.filter_notes(lambda note: not (0.25 <= float(note.duration.quarterLength) <= 4))
    song.filter_notes(lambda note:  float(note.duration.quarterLength) not in [i/4 for i in range(1, 17)])
    song.part_exclusion(lambda part: part.get_instrument_type() == m21.instrument.Sampler)
    song.part_exclusion(lambda part: np.std([int(note.pitch.midi) for note in part.notes]) < 2)

    vecs += song.as_vector()
np.save("data/data_small.npy", np.asarray(vecs), allow_pickle=True)
print(len(vecs))

##
# Test loading dataset
# TODO: many parts are boaring, just repeating same note
vec = np.load("data/data_small.npy", allow_pickle=True)
print(vec)


