from pathlib import Path
import json
import re
import warnings
import charset_normalizer

FILES_PATH = Path("files")
NOT_FOUND = "###NOTFOUND###"


def load_items():
    ret = []
    with open("mixotic-crawl.jsonl") as f:
        for line in f.readlines():
            ret.append(json.loads(line))
    return ret


def get_file(item, ext):
    for i in item["files"]:
        if i["path"].endswith(ext):
            return i


def parse_info(path):
    with open(path, "rb") as f:
        contents = f.read()
        match = charset_normalizer.from_bytes(
            contents,
            cp_isolation=["utf-8", "iso8859-15", "windows-1252", "utf-16"],
            explain=True,
        ).best()
        if match is None:
            raise RuntimeError(f"Could not detect encoding of {path}")
        print(path, match.encoding)
        txt = contents.decode(match.encoding).splitlines()

    # clean useless lines
    txt[0] = txt[0].replace("\ufeff", "")  # remove unicode BOM
    txt = [i.strip() for i in txt]
    txt = [i for i in txt if i != ""]
    txt = [i for i in txt if not "downloaded from the netlabel" in i.lower()]
    txt = [i for i in txt if not "playlist" in i.lower()]
    txt = [i for i in txt if not "reissue, originally released" in i.lower()]
    txt = [i for i in txt if not "free to download and use" in i.lower()]

    print("\n".join(txt))

    # parse header
    if m := re.match(r"[\w\s]+ (\d+) - (.+) - (.+)", txt[0]):
        id = m.group(1)
        artist = m.group(2)
        title = m.group(3)
    else:
        raise RuntimeError(f"Could not parse header in {path}", txt[0])

    playlist = []
    for i in range(1, len(txt) - 1, 2):
        if m := re.match(r"\w+\s*[-–]?\s*(.+)\s+[-–]\s+(.+)", txt[i]):
            track = m.group(2)
            trackartist = m.group(1)
        elif m := re.match(r"\w+\s*[-–]?\s*(.+)\s+\((.+)\)", txt[i]):
            track = m.group(2)
            trackartist = m.group(1)
        else:
            track = NOT_FOUND
            trackartist = NOT_FOUND
            warnings.warn(f"Could not parse track info in {path}: {txt[i]}")

        # parse second line
        if txt[i + 1].lower() == "unreleased":
            release = None
            extra = None
        elif m := re.match(r"(.+)\s+(\(?.+\)?)?", txt[i + 1]):
            release = m.group(1)
            extra = m.group(2)
        else:
            release = NOT_FOUND
            extra = NOT_FOUND
            warnings.warn(f"Could not parse label info in {path}: {txt[i + 1]}")

        playlist.append(
            {
                "title": track,
                "artist": trackartist,
                "release": release,
                "extra": extra,
            }
        )

    return {"title": title, "artist": artist, "playlist": playlist}


items = load_items()
for i in items:
    txt = get_file(i, "txt")
    mp3 = get_file(i, "mp3")
    cover = get_file(i, "jpg")
    if cover is None:
        cover = get_file(i, "gif")
    if cover is None:
        cover = get_file(i, "png")

    assert txt is not None
    assert mp3 is not None
    assert cover is not None

    txt_path = FILES_PATH / txt["path"]
    meta = parse_info(txt_path)
    del i["file_urls"]
    del i["files"]

    i["audio"] = mp3["path"]
    i["cover"] = cover["path"]
    i.update(meta)

# sort by id
items = sorted(items, key=lambda i: i["id"])

with open("mixotic.json", "w") as f:
    json.dump(items, f, indent=4)
