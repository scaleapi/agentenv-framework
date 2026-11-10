# Run as root once the app's files are in: who ran it, in the folder it was given.
echo "$(id -un) in $PREFIX" > run.txt
