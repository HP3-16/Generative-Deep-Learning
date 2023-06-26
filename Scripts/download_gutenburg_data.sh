
FILE=$1
NAME=$2

URL=http://www.gutenberg.org/cache/epub/$FILE/pg$FILE.txt

TARGET_DIR=./data/$NAME/
mkdir $TARGET_DIR
TXT_FILE=./data/$NAME/data.txt
wget --no-check-certificate -N $URL -O $TXT_FILE

