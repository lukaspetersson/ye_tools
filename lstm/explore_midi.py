##
import music21
import numpy as np
from collections import Counter, defaultdict
import glob
import time
import sys

##
len(glob.glob("data/adl-piano-midi/**/*.mid", recursive=True))

##
score = music21.converter.parse("data/adl-piano-midi/Rock/Psychedelic Rock/Beatles/Hey Jude.mid")

##
dir(score)

##
#TODO: not working, for now use first part
print(len(score.parts))
score.chordify()
print(len(score.parts))

##
score.parts[0].recursionType

##
help(music21.duration.Duration)

##
types = set()
for el in score.recurse():
    types.add(type(el))
    if isinstance(el, music21.key.Key):
        print(el)
types

## 
t0 = time.time()
k = score.analyze('key')
print(time.time()-t0)
t0 = time.time()
for el in score.recurse():
    if isinstance(el, music21.key.Key):
        break
print(time.time()-t0)
print(k==el)

##
#Method 2 does not work!
for i, file in enumerate(glob.glob("data/adl-piano-midi/**/*.mid", recursive=True)):
    if i > 5: break
    s = music21.converter.parse(file)
    #method 1
    k = s.analyze('key')
    #method 2
    for el in s.recurse():
        if isinstance(el, music21.key.Key):
            print(el, k, file)
    
##
count = 0
acceptable_durations = set([i/4 for i in range(1, 17)])
for i, file in enumerate(glob.glob("data/adl-piano-midi/**/*.mid", recursive=True)):
    if i > 30: break
    s = music21.converter.parse(file)
    c = []
    for note in s.parts[0].notesAndRests:
        if float(note.duration.quarterLength) not in acceptable_durations:
            c.append(float(note.duration.quarterLength))
    if len(c) > 0:
        count+=1
        print(c)
        print(file)
print(count, i)

##
k = score.analyze('key')
p = "C" if k.mode == "major" else "A"
i = music21.interval.Interval(k.tonic, music21.pitch.Pitch(p))
score = score.transpose(i)
print("Transformed from key:", k, "to:", score.analyze('key'))

##
durations = defaultdict(list)
for i, file in enumerate(glob.glob("data/adl-piano-midi/**/*.mid", recursive=True)):
    if i > 10: break
    s = music21.converter.parse(file)
    for el in score.recurse().notesAndRests:
        if isinstance(el, music21.note.Note):
            durations["note"].append(float(el.duration.quarterLength))
        elif isinstance(el, music21.chord.Chord):
            durations["chord"].append(float(el.duration.quarterLength))
        elif isinstance(el, music21.note.Rest):
            durations["rest"].append(float(el.duration.quarterLength))

print("Note durations")
print(sorted(Counter(durations["note"]).items(), key=lambda x: -x[1]))
print("Chord durations")
print(sorted(Counter(durations["chord"]).items(), key=lambda x: -x[1]))
print("Rest durations")
print(sorted(Counter(durations["rest"]).items(), key=lambda x: -x[1]))

##
acceptable_durations = set([i/4 for i in range(1, 17)])

##
# whole note with duration 1 and pitch 3
a = np.zeros((4, 8),int)
a[0][3] = 1
for i in range(1, 4):
    a[i][3] = -1


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
#TODO: can order be gueranteed?
for note in score.recurse().notesAndRests:
    print(note.offset)


## 
#TODO: only accept ceirtain quarterlengths
song = np.zeros((128,), int)
for note in score.parts[0].recurse().notesAndRests:
    dur = int(note.duration.quarterLength*4)
    for i in range(dur):
        vec = np.zeros((128,), int)
        if isinstance(note, music21.note.Note):
            vec[note.pitch.midi] = 1 if i==0 else -1
            song = np.vstack([song, vec])
       #elif isinstance(note, music21.chord.Chord):
        elif isinstance(note, music21.note.Rest):
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
class Song():
    def __init__(self, path):
        self.score = music21.converter.parse(path)
        self.acceptable_durations = set([i/4 for i in range(1, 17)])
        self.notes = self.score.parts[0].recurse().notesAndRests

    def has_unacceptable_duration(self):
        for note in self.notes:
            if float(note.duration.quarterLength) not in self.acceptable_durations and float(note.duration.quarterLength) < 4:
                return True
        return False

    def transpose(self):
        k = self.score.analyze("key")
        p = "C" if k.mode == "major" else "A"
        i = music21.interval.Interval(k.tonic, music21.pitch.Pitch(p))
        self.score = self.score.transpose(i)

    def get_vector_representation(self):
        song = np.zeros((128,), int)
        for note in self.notes:
            dur = int(note.duration.quarterLength*4)
            if dur > 16:
                continue
            for i in range(dur):
                vec = np.zeros((128,), int)
                if isinstance(note, music21.note.Note):
                    vec[note.pitch.midi] = 1 if i==0 else -1
                    song = np.vstack([song, vec])
                elif isinstance(note, music21.note.Rest):
                    song = np.vstack([song, vec])
                #TODO: support chords
        song = np.delete(song, 0, 0)
        return song

##
t0 = time.time()
song = Song("data/adl-piano-midi/Rock/Art Rock/Talking Heads/Burning Down The House.mid")
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
    song = Song(file)
    if song.has_unacceptable_duration():
        continue
    song.transpose()
    vec = song.get_vector_representation()
    vecs.append(vec)
np.save("data/data.npy", np.asarray(vecs), allow_pickle=True)
print(len(vecs))

##
np.set_printoptions(threshold=sys.maxsize)
print(len(np.load("data/data.npy", allow_pickle=True)))

##
vec = np.load("data/data.npy", allow_pickle=True)
for i, sixteenth in enumerate(vec[1]):
    print(np.nonzero((sixteenth != 0))[0])
    if i > 100:break

