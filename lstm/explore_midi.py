##
import m21 as m21
import numpy as np
from collections import Counter, defaultdict
import glob
import time
import os

##
score = m21.converter.parse("data/lmd_full/a/a00b0f5acc0fd4e4fc9f32830d61978d.mid")

##
# 
def include_note(note):
        return float(note.duration.quarterLength) <= 4 and not (float(note.duration.quarterLength) != 4 and isinstance(note, m21.note.Rest)):

stream = m21.stream.Stream()
for part in score.parts:
    s = m21.stream.Stream()
    for note in part.notesAndRests:
        if include_note(note):
            s.append(note)
    stream.append(s)

##
stream.show()

##
stream.write()

##
len(glob.glob("data/lmd_full/**/*.mid", recursive=True))

##
# score, part, meassure is subset of stream
dir(score)

##
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
# show that method only sometimes work
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
    if i > 30: break
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
# transpose key C major or A minor

k = score.analyze('key')
p = "C" if k.mode == "major" else "A"
i = m21.interval.Interval(k.tonic, m21.pitch.Pitch(p))
score = score.transpose(i)
print("Transformed from key:", k, "to:", score.analyze('key'))

##
# count durations of each note type

durations = defaultdict(list)
for i, file in enumerate(glob.glob("data/lmd_full/0/**/*.mid", recursive=True)):
    if i > 0: break
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
a = np.ones((8,), int)
a = np.vstack([a, np.zeros((8,),int)])
a = np.vstack([a, np.ones((5,8),int)])
a = np.delete(a,0,0)
print(len(a))
for i in a:
    print(i)
print(len(a))

## 
#TODO: only accept ceirtain quarterlengths
song = np.zeros((128,), int)
for note in score.parts[0].recurse().notesAndRests:
    dur = int(note.duration.quarterLength*4)
    for i in range(dur):
        vec = np.zeros((128,), int)
        if isinstance(note, m21.note.Note):
            vec[note.pitch.midi] = 1 if i==0 else -1
            song = np.vstack([song, vec])
       #elif isinstance(note, m21.chord.Chord):
        elif isinstance(note, m21.note.Rest):
            song = np.vstack([song, vec])
song.shape

##
for note in song[100:120]:
    if 1 in note:
        print("play", np.argwhere(note==1)[0])
    elif -1 in note:
        print("continue", np.argwhere(note==-1)[0])
    else:
        print("rest")
##
print(stream.timeSignature)

##
idx = 3 
print(len(song.get_part(idx).part), len(song.get_part(idx).part.recurse()))
song.show("text", idx)

##
class Part():
    def __init__(self, part):
        self.part = part
        self.notes = m21.stream.Stream(part.recurse().notesAndRests)

    def get_instrument_type(self):
        return type(self.part.getInstrument())

    def count_note_types(self):
        n, r, c = 0, 0, 0
        for note in self.notes:
            if isinstance(note, m21.note.Note): n += 1
            elif isinstance(note, m21.note.Rest): r += 1
            elif isinstance(note, m21.chord.Chord): c += 1
        return (n, r, c)

    def as_tuples(self):
        vec = np.zeros((len(self.notes), 2), int)
        for i, note in enumerate(self.notes):
            dur = int(note.duration.quarterLength*4)
            if isinstance(note, m21.note.Note):
                vec[i][0] = note.pitch.midi
                vec[i][1] = dur
            elif isinstance(note, m21.note.Rest):
                pass
            elif isinstance(note, m21.chord.Chord):
                #TODO
                pass
        return vec

'''
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
        self.score = m21.converter.parse(path)

        if transpose:
            #TODO: different parts have different keys, but they shouldnt
            k = self.score.analyze("key")
            p = "C" if k.mode == "major" else "A"
            i = m21.interval.Interval(k.tonic, m21.pitch.Pitch(p))
            self.score = self.score.transpose(i)

        self.parts = [Part(part) for part in self.score.parts]
        self.excluded = set()
        self._key = None
    
    def _include(self, i):
        self.excluded.remove(i)

    def _exclude(self, i):
        self.excluded.add(i)

    def part_exclusion(self, func):
        for i, part in enumerate(self.parts):
            if func(part):
                self._exclude(i)
    
    def note_exclusion(self, func):
        for i, part in enumerate(self.parts):
            if any(map(func, part.notes)):
                self._exclude(i)

    def filter_notes(self, func):
        for i, part in enumerate(self.parts):
            for note in part.notes:
                if func(note):
                    idx = self.parts[i].notes.index(note)
                    self.parts[i].notes.pop(idx)
    
    def key(self):
        if not self._key:
            #TODO: does this work?
            self._key = self.score.analyze("key")
        return self._key

    def to_stream(self):
        s = m21.stream.Stream()
        for i, part in enumerate(self.parts):
            if i not in self.excluded: 
                s.append(part.notes)
        return s

    def get_part(self, i):
        return self.parts[i]

    def show(self, fmt="musicxml" ,i=-1):
        if i > -1:
            self.parts[i].notes.show(fmt)
        else:
            self.to_stream().show(fmt)

##
song = Song("data/lmd_full/3/3003bbf06bec7c8ff1add82b50a84ae4.mid", transpose=False)
# filter notes with duration >4 and <0.25
song.filter_notes(lambda note: not (0.25 <= float(note.duration.quarterLength) <= 4))
# filter out non 4ths
song.note_exclusion(lambda note:  float(note.duration.quarterLength) not in [i/4 for i in range(1, 17)])
# filter out instruments
song.part_exclusion(lambda part: part.get_instrument_type() == m21.instrument.Sampler)
song.excluded

##
song = Song("data/lmd_full/3/3003bbf06bec7c8ff1add82b50a84ae4.mid", transpose=False)
song.show()

##
for part in song.parts:
    print(type(part.get_instrument()))

##
song = Song("data/lmd_full/a/a00b0f5acc0fd4e4fc9f32830d61978d.mid", transpose=False)
#print(song.get_part(3).get_instrument_type())
#print(len(song.get_part(3).part.recurse().notesAndRests))

##
t0 = time.time()
song = SongV2("data/lmd_full/3/3000f058dee377b7586acda383472d5a.mid")
print(time.time()-t0)
song.transpose()
print(time.time()-t0)
vec = song.get_vector_representation()
print(time.time()-t0)
# 1s per song = 3h
print(song.has_unacceptable_duration())

##
vecs = []
for i, file in enumerate(glob.glob("data/adl-piano-midi/**/*.mid", recursive=True)):
    if i > 100: break
    song = SongV2(file)
    if song.has_unacceptable_duration():
        continue
    song.transpose()
    vec = song.get_vector_representation()
    vecs.append(vec)
np.save("data/datav2.npy", np.asarray(vecs), allow_pickle=True)
print(len(vecs))

##
np.set_printoptions(threshold=sys.maxsize)
print(len(np.load("data/data.npy", allow_pickle=True)))

##
vec = np.load("data/data.npy", allow_pickle=True)
for i, sixteenth in enumerate(vec[1]):
    print(np.nonzero((sixteenth != 0))[0])
    if i > 100:break

