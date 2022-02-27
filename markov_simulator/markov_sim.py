import numpy as np

# ^ for sharp
notes = ['C',"^C" ,'D',"^D", 'E', 'F',"^F", 'G',"^G", 'A',"^A", 'B', 'c',"^c", 'd',"^d", 'e', 'f',"^f", 'g',"^g", 'a',"^a", 'b']
chords = ['C', 'G', 'Am', 'F'] 

# Matrix for rational-number aproximation of the relative frequencies
# load matrix from approximation.py
N = np.ones([24,24])
f = open("matrix.txt")
for i, line in enumerate(f):
    for j, num in enumerate(line[1:-2].split(', ')):
        N[i][j] = float(num)

# Matrix to restrict notes to be in the chord's scale
p = 1/14
C = np.array([
                [p,0,p,0,p,p,0,p,0,p,0,p,p,0,p,0,p,p,0,p,0,p,0,p],
                [p,0,p,0,p,0,p,p,0,p,0,p,p,0,p,0,p,0,p,p,0,p,0,p],
                [p,0,p,0,p,p,0,p,0,p,0,p,p,0,p,0,p,p,0,p,0,p,0,p],
                [p,0,p,0,p,p,0,p,0,p,p,0,p,0,p,0,p,p,0,p,0,p,p,0]
            ])

#Matrix to limit the length of the frequency inteval between notes
D = np.zeros([24,24])
for i in range(24):
    for j in range(24):
        D[i][j] = 1/(abs(i-j)+1.4)
    s = np.sum(D[i])
    D[i] /= s

#prepare file in abc format
f = open("out.abc", "w")
f.write('%%abc-charset utf-8\n')
f.write('%%titlefont NewCenturySchlbk-Roman 22\n')
f.write('%%subtitlefont NewCenturySchlbk 16\n')
f.write('%%composerfont NewCenturySchlbk-Italic 14\n')
f.write('%%footerfont NewCenturySchlbk 16\n')
f.write('%%headerfont NewCenturySchlbk 16\n')
f.write('%%tempofont NewCenturySchlbk-Bold 15\n')
f.write('X:1\n')
f.write('T:Markovtest\n')
f.write('C:strauss (localhost)\n')
f.write('L:1/16\n')
f.write('M:4/4\n')
f.write('R:walz\n')
# 180=presto, 160=vivace, 132=allegro, 96=moderato, 72=andante
f.write('Q: "Vivace" 1/4 = 20\n')
f.write('V:1\n')
f.write('K:C\n')
f.write('%%MIDI channel 1\n')
#f.write('%%MIDI program 1 48\n') # strängar
#f.write('%%MIDI program 1 0\n') # piano
f.write('%%MIDI program 1 22\n')
f.write('%%MIDI chordprog 0\n')
f.write('%%MIDI chordvol 40\n')
f.write('%%MIDI drum ddd 60 61 61 50 40 40\n')
f.write('%%MIDI drumon\n')

def next(note, chord):
    #element wise multiply and normalize
    #TODO: weight differently
    M = np.multiply(np.multiply(N[note],D[note]),C[chord])
    s = np.sum(M)
    M /= s

    r = np.random.rand()
    j = -1
    s = 0
    while s < r:
        j = j + 1
        s = s + M[j]
    return j

t = open("../rnn_beat/generated_beats.txt")
beats = t.readline()
tune = ""

#choose a random init chord and a random init note in the chord's scale
chord = np.random.randint(0, len(chords))
note = next(np.random.randint(0, len(notes)), chord)

for i, c in enumerate(beats):
    if c == '*':
        tune += notes[note]
        note = next(note, chord)
    elif c == '|':
        tune += c+'"'+chords[chord]+'"'
    else:
        tune += c
    chord = (chord+1)%4

f.write(tune)
f.close()

# chovert abc file to midi
from music21 import converter
s = converter.parse('out.abc')
s.write('midi', fp = 'out.mid')
