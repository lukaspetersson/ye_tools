import mido

def print_meta(file):
    print(file.ticks_per_beat)
    print(len(file.tracks))
    for track in file.tracks:
        for i, msg in enumerate(track):
            print(msg)

def get_takt_easy(file):
    tpb = file.ticks_per_beat
    takt = []
    for msg in file.tracks[0]:
        if msg.type == "note_off":
            num = (msg.time+1)/tpb
            # if there is noise
            for i in range(4):
                for j in range(1, 5):
                    cand = i + j * 0.25
                    if cand-0.03 < num < cand +0.03:
                        #midi has 1/4 note as default, we want 1/8 as default
                        takt.append(cand*2)
    return takt

def write_takt(songs):
    f = open("takt.txt", "w")
    msg = ""
    for song in songs:
        for num in song:
            msg += str(num)
            msg += " "
        msg += "\n"
        msg += "\n"
    f.write(msg)
    f.close()

def write_takt_string(songs):
    f = open("takt_string.txt", "w")
    msg = ""
    for song in songs:
        for note in song:
            if note == 0.5:
                msg += 'a'
            elif note == 1:
                msg += 'b'
            elif note == 1.5:
                msg += 'c'
            elif note == 2:
                msg += 'd'
            elif note == 3:
                msg += 'e'
            elif note == 4:
                msg += 'f'
            elif note == 8:
                msg += 'g'
        msg += '\n'
        msg += '\n'
    f.write(msg)
    f.close()

import os
songs = []
for song_file_name in os.listdir("songs"):
    song = mido.MidiFile(os.path.join("songs",song_file_name))
    songs.append(get_takt_easy(song))

write_takt_string(songs)
