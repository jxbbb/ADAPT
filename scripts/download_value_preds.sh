# --------------------------------
# Setup
# --------------------------------
export REPO_DIR=$PWD
if [ ! -d $REPO_DIR/value-submit ] ; then
    mkdir -p $REPO_DIR/value-submit
fi

BLOB='https://datarelease.blob.core.windows.net/swinbert'


# --------------------------------
# Download our prediction files that were evaluated on VALUE Leaderboard Evaluation Server
# --------------------------------

wget -nc $BLOB/swinbert-value-submit.zip -O $REPO_DIR/value-submit/swinbert-value-submit.zip