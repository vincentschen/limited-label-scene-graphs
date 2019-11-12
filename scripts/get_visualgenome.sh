#!/bin/bash
mkdir -p data
cd data
mkdir -p VisualGenome
cd VisualGenome

# Get VG metadata files 
wget http://visualgenome.org/static/data/dataset/image_data.json.zip
wget http://visualgenome.org/static/data/dataset/relationships.json.zip
wget http://visualgenome.org/static/data/dataset/objects.json.zip
unzip image_data.json.zip
unzip relationships.json.zip
unzip objects.json.zip
rm image_data.json.zip
rm relationships.json.zip
rm objects.json.zip

wget http://visualgenome.org/static/data/dataset/relationship_alias.txt
wget http://visualgenome.org/static/data/dataset/object_alias.txt
wget https://raw.githubusercontent.com/danfeiX/scene-graph-TF-release/master/data_tools/VG/predicate_list.txt
wget https://raw.githubusercontent.com/danfeiX/scene-graph-TF-release/master/data_tools/VG/object_list.txt

wget https://www.dropbox.com/s/1bzhco3fmjvrg9k/vg_splits.zip
unzip vg_splits.zip
rm vg_splits.zip

# Download and unzip images.
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip images.zip
unzip images2.zip
rm images.zip
rm images2.zip

# Move images from VG_100K_2 to VG_100K.
find VG_100K_2/ -name "*.jpg" -exec mv {} VG_100K \;
rm -r VG_100K_2/

cd ../..
