# --------------------------------
# Setup
# --------------------------------
export REPO_DIR=$PWD
if [ ! -d $REPO_DIR/models ] ; then
    mkdir -p $REPO_DIR/models
fi
if [ ! -d $REPO_DIR/models/table1 ] ; then
    mkdir -p $REPO_DIR/models/table1
fi
if [ ! -d $REPO_DIR/models/32frm ] ; then
    mkdir -p $REPO_DIR/models/32frm
fi
BLOB='https://datarelease.blob.core.windows.net/swinbert'


# --------------------------------
# Download our best performing checkpoints for each dataset (corresponding to Table 1 in paper)
# --------------------------------

for DATASET in 'youcook2'
do
    wget -nc $BLOB/models/${DATASET}-table1.zip -O $REPO_DIR/models/table1/${DATASET}-table1.zip
    unzip $REPO_DIR/models/table1/${DATASET}-table1.zip -d $REPO_DIR/models/table1/${DATASET}/
    rm $REPO_DIR/models/table1/${DATASET}-table1.zip
done


# --------------------------------
# Download our 32-frame-based model 
# --------------------------------

for DATASET in 'youcook2'
do
    wget -nc $BLOB/models/${DATASET}-32frm.zip -O $REPO_DIR/models/32frm/${DATASET}-32frm.zip
    unzip $REPO_DIR/models/32frm/${DATASET}-32frm.zip -d $REPO_DIR/models/32frm/${DATASET}/
    rm $REPO_DIR/models/32frm/${DATASET}-32frm.zip
done
