# SatLakeIce
Make Ice Great Again

# Instructions
1. Obtain the geotiff data and unzip into the ./data folder (one folder per
image class)
2. From the project root, run ```python geotiff_convert.py``` to turn the 
GeoTiff data into numpy arrays (saved by joblib into the ./data folder).
3. Run ```python train.py``` to train the network. The best result will be saved
into the ./trained folder.
4. Run ```python inference.py``` to obtain a result on our test split data, or
give inference.py a path to a folder of geotiffs (or a single geotiff) to classify
those images instead.
