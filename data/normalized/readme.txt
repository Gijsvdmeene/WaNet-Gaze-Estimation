######################################################################################################################################
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
######################################################################################################################################

Here is the normalized data for MPIIFaceGaze dataset.
There are 15 data files for the 15 participants. In each file, there are is a "Data" structure includes "Data.data" and "Data.label".

-Data.data
This is the normalized face images in 448*448 pixels size, in the shape of [image height, image width, color channel, number of samples] which is [448, 448, 3, 3000] for each file.

-Data.label
- This is the labels for these data. It has the shape of [channel, number of samples] which is [16, 3000].
The first two dimensions are the normalized gaze direction, the 3~4 dimensions are the normalized head pose, and the 5~16 dimensions are facial landmarks as (x, y) for 6 landmarks.

Please refer to the code "save_to_hdf5.m" for how to convert these .mat file to .h5 files

Enjoy!