MAESTRO_DIRECTORY='maestro-v2.0.0'

test -f $MAESTRO_DIRECTORY"-midi.zip" || \
curl "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/"$MAESTRO_DIRECTORY"-midi.zip" --output $MAESTRO_DIRECTORY"-midi.zip"
unzip -n $MAESTRO_DIRECTORY"-midi.zip" > /dev/null

test -f $MAESTRO_DIRECTORY'-tensors.zip' || \
curl 'https://dl.dropboxusercontent.com/s/ebh7g5e623rfl7y/'$MAESTRO_DIRECTORY'-tensors.zip' --output $MAESTRO_DIRECTORY'-tensors.zip'
unzip -n $MAESTRO_DIRECTORY'-tensors.zip' > /dev/null
