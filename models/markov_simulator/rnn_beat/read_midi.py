import mido
import os
import string

#Some midi files are easy, one note after the other.
#For such files we can just keep track of the "of_note" message type, the length of this message is the time the previous note was played
def beats_per_note_easy(file):
    # a beat in midi is a 1/4 note
    tpb = file.ticks_per_beat
    # store the beats of each note
    beat = []
    for msg in file.tracks[0]:
        if msg.type == "note_off":
            # msg.time gives number of ticks since last msg. Devide by ticks/beat to get the number of beats for that note
            # Add 1 because the note_on is allways 1
            # note_beats = 1 if the note is a 1/4 note
            note_beats = (msg.time+1)/tpb
            # There is often noise in the file so that we dont get perfect integers
            # Remove noise by accepting a range as an integer
            for i in range(4):
                for j in range(1, 5):
                    # cand becomes 0.25, 0.5, 0.75, ... 4, 
                    # i.e. from 1/16th-note to whole-note 
                    cand = i + j * 0.25
                    if cand-0.03 < note_beats < cand +0.03:
                        beat.append(cand)
    return beat

def write_beats_as_chars(songs):
    f = open("note_beats.txt", "w")
    msg = ""
    alphabet = list(string.ascii_lowercase)
    for song in songs:
        for note in song:
            # encode 0/16th note as 'a', 1/16th as 'b', ...
            msg += alphabet[int(note*4)]
        if len(msg) > 0:
            msg += '\n'
    f.write(msg)
    f.close()

songs = []
for song_file_name in os.listdir("songs"):
    song = mido.MidiFile(os.path.join("songs",song_file_name))
    songs.append(beats_per_note_easy(song))

write_beats_as_chars(songs)
