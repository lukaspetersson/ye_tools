##
import music21 as m21
import numpy as np
from collections import Counter, defaultdict
import glob
import time
import os
import subprocess
import sys
from contextlib import contextmanager

##
s = m21.stream.Stream()
s.warnings = False


##
score = m21.converter.parse("data/lmd_full/a/a00b0f5acc0fd4e4fc9f32830d61978d.mid")

##
print(inspect.signature(m21.stream.Stream().show()))

##
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

##
stream = m21.stream.Stream()
for part in score.parts:
    s = m21.stream.Stream()
    for note in part.notesAndRests:
        if float(note.duration.quarterLength) <= 4 and not (float(note.duration.quarterLength) != 4 and isinstance(note, m21.note.Rest)):
            s.append(note)
    stream.append(s)

print(len(stream.scores))
##
stream.write("midi", fp="/home/lukas/ye_tools/lstm/test.midi")
#os.system("mscore3 test.midi")
subprocess.run("mscore3 test.midi", shell=True, check=False)

##
with contextlib.redirect_stdout(None):
    stream.show()

##
s = m21.stream.Stream()
s.append(score.parts[7])
#s.append(score.parts[1].notesAndRests)
s.write("lily.png",fp="~/ye_tools/lstm/test.png")

##
print("Hello"

##
m21.environment.set("musescoreDirectPNGPath", "/usr/bin/mscore3")

##
len(glob.glob("data/adl-piano-midi/**/*.mid", recursive=True))


##
score.parts[0].timeSignature
##
s = m21.stream.Stream()
s.append(score)
s.show()
#score.parts[0].iter().filter.ClassFilter("Rest").show()


##
for part in score.parts:
    for note in part.notesAndRests:

##
dir(score)

##
#TODO: not working, for now use first part, also very slow
print(len(score.parts))
score.chordify()
print(len(score.parts))

##
dir(score.parts[0])

##
types = set()
for el in score.recurse():
    types.add(type(el))
    if isinstance(el, m21.instrument.Instrument):
        print("HHHHHHHHHHHHHHHHHHhh")
types

## 
t0 = time.time()
k = score.analyze('key')
print(time.time()-t0)
t0 = time.time()
for el in score.recurse():
    if isinstance(el, m21.key.Key):
        break
print(time.time()-t0)
print(k==el)

##
#Method 2 does not work!
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
count = 0
acceptable_durations = set([i/4 for i in range(1, 17)])
for i, file in enumerate(glob.glob("data/adl-piano-midi/**/*.mid", recursive=True)):
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
print(count, i)

##
k = score.analyze('key')
p = "C" if k.mode == "major" else "A"
i = m21.interval.Interval(k.tonic, music21.pitch.Pitch(p))
score = score.transpose(i)
print("Transformed from key:", k, "to:", score.analyze('key'))

##
durations = defaultdict(list)
for i, file in enumerate(glob.glob("data/adl-piano-midi/**/*.mid", recursive=True)):
    if i > 10: break
    s = m21.converter.parse(file)
    for el in score.recurse().notesAndRests:
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
class Song():
    def __init__(self, path):
        self.score = m21.converter.parse(path)
        self.acceptable_durations = set([i/4 for i in range(1, 17)])
        #TODO: use more parts
        #TODO: filter out parts which are majority rests
        #TODO: filter out bass and drums? which play melody?
        self.notes = self.score.parts[-1].recurse().notesAndRests

    def get_timesignature(self):
        #return self.score.parts[0].getElementsByClass(m21.meter.TimeSignature)[0]
        return self.score.parts[0].timeSignature

    def get_instrument(self):
        #return self.score.parts[0].recurse().getElementsByClass(m21.instrument.Instrument)[0]
        return self.score.parts[0].getInstrument()

    def has_unacceptable_duration(self):
        for note in self.notes:
            if float(note.duration.quarterLength) not in self.acceptable_durations and float(note.duration.quarterLength) < 4:
                return True
        return False

    def transpose(self):
        k = self.score.analyze("key")
        p = "C" if k.mode == "major" else "A"
        i = m21.interval.Interval(k.tonic, music21.pitch.Pitch(p))
        self.score = self.score.transpose(i)

    def count_note_types(self):
        n, r, c = 0, 0, 0
        for note in self.notes:
            if isinstance(note, m21.note.Note): n += 1
            elif isinstance(note, m21.note.Rest): r += 1
            elif isinstance(note, m21.chord.Chord): c += 1
        return (n, r, c)

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

    def as_tuples(self):
        song = np.zeros((len(self.notes), 2), int)
        for i, note in enumerate(self.notes):
            dur = int(note.duration.quarterLength*4)
            if isinstance(note, m21.note.Note):
                song[i][0] = note.pitch.midi
                song[i][1] = dur
            elif isinstance(note, m21.note.Rest):
                pass
            elif isinstance(note, m21.chord.Chord):
                #TODO
                pass
        return song
##
song = Song("data/adl-piano-midi/Rock/Art Rock/Talking Heads/Burning Down The House.mid")

##
song.as_tuples()
##
song.count_note_types()

##

with open("data/test.txt", "w") as f:
    with np.printoptions(threshold=np.inf):
       f.write(np.array2string(song.as_tuples()))
##
len(song.score.parts.chordify())

##
t0 = time.time()
song = SongV2("data/adl-piano-midi/Rock/Art Rock/Talking Heads/Burning Down The House.mid")
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

