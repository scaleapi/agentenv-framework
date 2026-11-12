"""unpack-msix.py MSIX DEST: the ChatGPT app's resources out of its MSIX into DEST. An MSIX is a zip whose entry names
are percent-encoded, '@' as %40 say, as Windows decodes them when it installs one, so they're decoded here; the app's
own Electron code is left out, since the driver never runs the app."""
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

RESOURCES = "app/resources/"
LEFT_OUT = ("app/resources/app.asar",)


def main(msix: Path, dest: Path) -> None:
    with zipfile.ZipFile(msix) as package:
        for entry in package.infolist():
            name = unquote(entry.filename)
            if entry.is_dir() or not name.startswith(RESOURCES) or name.startswith(LEFT_OUT):
                continue
            target = dest / name[len(RESOURCES):]
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(entry) as source, open(target, "wb") as out:
                shutil.copyfileobj(source, out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
