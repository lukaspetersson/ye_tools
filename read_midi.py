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

def write_takt(arr):
    f = open("takt.txt", "w")
    msg = ""
    for num in arr:
        msg += str(num)
        msg += " "
    f.write(msg)
    f.close()


mid = mido.MidiFile('songs/Pärongrisen_fa1d70.mid')
write_takt(get_takt_easy(mid))
