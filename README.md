# FITS-image-viewer
sample script to read and display FITS images as well as PIL supported images using PySimpleGUI
<img src=https://github.com/meteorspectroscopy/FITS-image-viewer/blob/master/doc/fits_image_viewer.PNG>
```Python
"""
    Demo Image Album.... displays images on Graph Element and transitions
    by sliding them across.  Click on right side of image to navigate down through filenames, left side for up.

    Contains a couple of handy PIL-based image functions that resize an image while maintaining correct proportion.
    One you pass a filename, the other a BASE64 string.

    Copyright 2020 PySimpleGUI.org
    modified by M. Dubs:
    - added reading astronomical FITS image files ('BITPIX' = 16 and -32 bit)
      (https://docs.astropy.org/en/stable/io/fits/)
    - ".png", ".jpg", "jpeg", ".tiff", ".bmp", ".ico", ".fit" files are read
    - comparison of new image display with previously used method
    - slide show with timing
    - zoom functions: scaling by factor
                      select rectangle for zoom
"""
```
two versions are available:

image_viewer_test.py: compares speed using Byte array versus storing images on disk for display

fits_image_viewer.py: old method of storing images on disk eliminated, but same functionality
archived
