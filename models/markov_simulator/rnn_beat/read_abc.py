#Not in use, ABC format was more complicated than I thought

import os

for song_file_name in os.listdir("/songs"):
    song_file = open(os.path.join("/songs", song_file_name))
    songs = []
    for line in song_file:
        if line[0] == '|':
            song = []
            meassure = []
            for c in line:
                if c.isnumeric():
                    meassure.append(c)
                elif c == '|':
                    if len(meassure) > 0:
                        song.append(meassure)
                        meassure = []
            songs.append(song)

f = open("songs.txt", "w")
for song in songs:
    for meassure in song:
        for c in meassure:
            f.write(c)
        f.write('|')
    f.write("\n")

f.close()



