#!/usr/bin/env bash

fileid=$1
filename=$2

if [[ "$fileid" == "" || "$filename" == "" ]]; then
    echo "Usage: bash dl_gdrive.sh FILE_ID FILE_NAME"
    exit 1
fi

wget --save-cookies cookies.txt \
    'https://docs.google.com/uc?export=download&id='${fileid} -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O ${filename} \
     'https://docs.google.com/uc?export=download&id='${fileid}'&confirm='$(<confirm.txt)

rm -f confirm.txt cookies.txt