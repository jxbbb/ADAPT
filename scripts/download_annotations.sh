# --------------------------------
# Setup
# --------------------------------
export REPO_DIR=$PWD
if [ ! -d $REPO_DIR/datasets ] ; then
    mkdir -p $REPO_DIR/datasets
fi
BLOB='https://datarelease.blob.core.windows.net/swinbert'

# --------------------------------
# Download caption annotations pre-parsed in TSV format
# --------------------------------

for DATASET in 'VATEX' 'MSRVTT-v2' 'TVC' 'YouCook2' 'MSVD'
do
    wget -nc $BLOB/datasets/${DATASET}.zip -O $REPO_DIR/datasets/${DATASET}.zip
    unzip $REPO_DIR/datasets/${DATASET}.zip -d $REPO_DIR/datasets/
    rm $REPO_DIR/datasets/${DATASET}.zip
done

# --------------------------------
# Note: Due to copyright issue, we are not able to release raw video files
# Please visit each dataset website to download the videos
# --------------------------------