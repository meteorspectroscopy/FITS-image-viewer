#!/usr/bin/env python
import PySimpleGUI as sg
from PIL import Image
import PIL
import io
import base64
import os
import time
from astropy.io import fits
import numpy as np
import math

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
# fixed parameters: image area
G_SIZE = [800, 500]
# scale factor for increase/ decrease size
sqrt2 = math.sqrt(2.0)


def get_img_filename(f, resize=None, crop=(None,), scale=False):
    """
    Generate image data using PIL
    works for image files .jpg, .png, .tif, .ico etc.
    extended for 32 and 64 bit fits-images
    f: image file PIL-readable (.png, .jpg etc) or fits-file (32 or 64 bit, b/w, color images)
    resize: (width, height) fit to size, keep aspect ratio
    cro: (left, bottom, right , top) crop original image before display
    scale: if true return imscale, cur_width, cur_height else
    return: byte-array from buffer
    """

    def scale_factor(cur_width, cur_height, new_width, new_height):
        """
        calculates scale factor to fit current image into new size
        return: image scale factor
        """
        scale_h = new_height / cur_height
        scale_w = new_width / cur_width
        imscale = min(scale_h, scale_w)  # fit image into new size
        return imscale

    im_scale = 1.0
    if f.lower().endswith('.fit'):
        ima, header = get_fits_image(f)  # get numpy array and fits-header
        cur_width, cur_height = ima.shape[1], ima.shape[0]  # size of image
        if crop[0]:
            # ima = ima0
            (left, bottom, right, top) = crop
            # clip if partial overlap
            bottom = max(bottom, 0)
            top = min(top, cur_height - 1)
            left = max(left, 0)
            right = min(right, cur_width - 1)
            if len(ima.shape) == 3:  # color image, crop color planes separately
                ima3 = ima[bottom:top, left:right, :]
                ima = ima3.copy()
            else:
                ima2 = ima[bottom:top, left:right]
                ima = ima2.copy()
            if ima.shape[0] * ima.shape[1] <= 0:
                print('no crop, replace original', ima.shape)
                sg.PopupError('crop area not within image area,\nreplace original', title='no crop possible')
                # ima = np.array([[0., 1.],[1., 0.]])  # any valid image array
                ima = np.array([[0., 0.25, 0.5, 0.75], [0.75, 0.5, 0.25, 0.]])  # any valid image array
        cur_width = ima.shape[1]
        cur_height = ima.shape[0]  # size of cropped image
        ima = np.clip(ima, 0.0, 1.0)  # Image does not like negative values
        ima = np.uint8(255 * ima)  # converts floating point to int8-array
        # https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
        # needed for imag.resize, converts numpy array to PIL format
        imag = Image.fromarray(np.array(ima))
        if resize:
            new_width, new_height = resize
            im_scale = scale_factor(cur_width, cur_height, new_width, new_height)
            ima = imag.resize((int(cur_width * im_scale), int(cur_height * im_scale)), PIL.Image.ANTIALIAS)
        if scale:
            del imag
            return im_scale, cur_width, cur_height  # used for zoom into image
        imag = ima
    else:
        imag = PIL.Image.open(f)
        if crop[0]:
            imag = imag.crop(crop)
        cur_width, cur_height = imag.size  # size of image
        if resize:
            new_width, new_height = resize
            im_scale = scale_factor(cur_width, cur_height, new_width, new_height)
            imag = imag.resize((int(cur_width * im_scale), int(cur_height * im_scale)), PIL.Image.ANTIALIAS)
        if scale:
            del imag
            return im_scale, cur_width, cur_height  # used for zoom into image
    bio = io.BytesIO()
    imag.save(bio, format="PNG")
    del imag
    return bio.getvalue()


def get_img_data(data, resize=None):
    """Generate PIL.Image data using PIL
    """
    imag = PIL.Image.open(io.BytesIO(base64.b64decode(data)))
    cur_width, cur_height = imag.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height / cur_height, new_width / cur_width)
        imag = imag.resize((cur_width * scale, cur_height * scale), PIL.Image.ANTIALIAS)
    bio = io.BytesIO()
    imag.save(bio, format="PNG")
    del imag
    return bio.getvalue()


def get_fits_image(fimage):
    """
    reads fits image data and header
    fimage: filename with or without extension
    converts 32-bit floating values and 16-bit data to Python compatible values
    reads also color images and transposes matrix to correct order
    (normalizes images to +/- 1 range)
    returns: image as np array, header
    """
    # fimage = change_extension(fimage, '.fit') # function only called if extension '.fit'
    ima, header = fits.getdata(fimage, header=True)
    ima = np.flipud(ima)
    # scale intensity to +/-1 range
    if int(header['BITPIX']) == -32:
        ima = np.array(ima) / 32767
    elif int(header['BITPIX']) == 16:
        ima = np.array(ima)
    else:
        print(f'unsupported data format BITPIX: {header["BITPIX"]}')
        exit()
    ima = ima/np.max(ima)
    if len(ima.shape) == 3:
        ima = np.transpose(ima, (1, 2, 0))
        ima = np.flipud(ima)  # otherwise, color images are upside down!?
    return ima, header


# ===========================================================================================================
# get size of window, default 800x500
layout_size = [[sg.Text('for PIL and FITS-images')], [sg.Text('Window width', size=(20, 1)),
                sg.InputText('800', size=(10, 1), key='win_x')],
               [sg.Text('Window height', size=(20, 1)),
                sg.InputText('500', size=(10, 1), key='win_y')], [sg.Button('Ok')]]
window_size = sg.Window('Image viewer', layout_size, finalize=True)
event, values = window_size.read(close=True)
try:
    G_SIZE = (int(values['win_x']), int(values['win_y']))
except:
    sg.PopupError('wrong input format, positive integers required')
folder = sg.popup_get_folder('Where are your images?', default_path='./')
if not folder:
    exit('no valid folder')

fit_zoom_area = True  # adjust zoom window aspect ratio to canvas size
file_list = os.listdir(folder)
fnames = [f for f in file_list if os.path.isfile(
    os.path.join(folder, f)) and f.lower().endswith((".png", ".jpg", "jpeg", ".tiff", ".bmp", ".ico", ".fit"))]
num_files = len(fnames)
if num_files == 0:
    sg.PopupError('no image files in this directory')
    exit('folder contains no images to display')
text_string = f'Size {G_SIZE[0]}x{G_SIZE[1]}, {folder}, N = {num_files} '
graph = sg.Graph(canvas_size=G_SIZE, graph_bottom_left=(0, 0), graph_top_right=G_SIZE, enable_events=True,
                 key='-GRAPH-', drag_submits=True)
layout = [[sg.Button('+'), sg.Button('-'), sg.Button('<'), sg.Button('>'), sg.Button('Slide_show'),
           sg.Button('Zoom'), sg.Checkbox('Fill zoom window', default=fit_zoom_area, key='fit_zoom'),
           sg.InputText(text_string + ' File: ' + fnames[0], size=(50, 1), key='text')], [graph]]

window = sg.Window('FITS Image Viewer', layout, margins=(0, 0), element_padding=(0, 0), use_default_focus=False,
                   finalize=True)

g00 = G_SIZE  # original area for images, can be zoomed in or out t0 G_SIZE
(g0, g1) = G_SIZE
offset, direction = 0, 'left'
rect_select_active = True
rectangle_selected = False
dragging = False
reset_zoom = False
start_point = end_point = prior_rect = event = None
x0 = y0 = dx = dy = 0

while True:
    file_to_display = os.path.join(folder, fnames[offset])
    img_data = get_img_filename(file_to_display, resize=G_SIZE)
    if event != 'Slide_show' and event != 'Zoom':
        idg = graph.draw_image(data=img_data, location=((g00[0] - G_SIZE[0]) // 2, (g00[1] + G_SIZE[1]) // 2))
        window['text'].Update(' File: ' + file_to_display)

    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

    if event == 'Slide_show':
        t0 = time.time()
        ti = time.time()
        direction = None  # avoids showing other images with button clicks
        graph.delete_figure(idg)
        for f in fnames:
            file_to_display = os.path.join(folder, f)
            img_data = get_img_filename(file_to_display, resize=G_SIZE)  # one function for different image formats
            idg = graph.draw_image(data=img_data, location=((g00[0] - G_SIZE[0]) // 2, (g00[1] + G_SIZE[1]) // 2))
            window['text'].Update(text_string + ' File: ' + f)
            window.refresh()
            graph.delete_figure(idg)
        t = time.time() - t0
        window['text'].Update(text_string + f' Time = {t:8.2f} sec')

    elif event == '-GRAPH-':
        # draw rectangle to set zoom window (does not work for fit-images)
        rect_select_active = True
        rectangle_selected = False
        print('graph')
        while rect_select_active:
            event, values = window.read()
            # graph.draw_rectangle((0, 0), (G_SIZE), line_color='blue')  # works with scale = 1
            # if event == "-GRAPH-":  # if there's a "Graph" event, then it's a mouse
            x, y = (values["-GRAPH-"])
            if not dragging:
                start_point = (x, y)
                dragging = True
            else:
                end_point = (x, y)
            if prior_rect:
                graph.delete_figure(prior_rect)
            if None not in (start_point, end_point):
                prior_rect = graph.draw_rectangle(start_point,
                                                  end_point, line_color='red')
            if event is not None and event.endswith('+UP'):
                # The drawing has ended because mouse up
                xy0 = tuple(0.5*(np.array(start_point) + np.array(end_point)))
                size = tuple(np.array(end_point) - np.array(start_point))
                print(start_point, end_point)
                reset_zoom = True if size[0] < 0 else False
                start_point, end_point = None, None  # enable grabbing a new rect
                dragging = False
                if prior_rect:
                    graph.delete_figure(prior_rect)
                if min(abs(size[0]), abs(size[1])) > 1:  # rectangle
                    window["text"].Update(value=f"rectangle at {xy0} with size {size}")
                    window.refresh()
                    xy0 = (x0, y0) = (int(xy0[0]), int(xy0[1]))
                    size = (dx, dy) = (int(abs((size[0] + 1) / 2)), int(abs((size[1] + 1) / 2)))
                    window["text"].Update(value=f"grabbed rectangle at {xy0} half size {size}")
                    window.refresh()
                    time.sleep(1)
                    rectangle_selected = True
                if idg:
                    graph.delete_figure(idg)
                G_SIZE = g00  # zoom rectangle works only with original size
                rect_select_active = False  # finished rectangle selection
                direction = None  # do not load other images

    elif event == 'Zoom':  # zoom to selected rectangle
        if rectangle_selected and not reset_zoom:
            # rectangle selected
            direction = None
            if not reset_zoom:
                # graph.change_coordinates(g0z, g1z) # does not work for images, only shifts image
                # calculate crop area in PIL coordinates (top left = (0,0)
                graph.delete_figure(idg)
                scale, imw, imh = get_img_filename(file_to_display, resize=G_SIZE, scale=True)
                del_y = y0 - g00[1] + imh * scale // 2
                if values['fit_zoom']:
                    if dx/G_SIZE[0] < dy/G_SIZE[1]:
                        dx = dy*G_SIZE[0]/G_SIZE[1]
                    else:
                        dy = dx*G_SIZE[1]/G_SIZE[0]
                print('zoom get_img_filename', file_to_display, G_SIZE)
                print('x, y, dx ,dy', x0, y0, dx, dy)
                left = int((x0 - dx) / scale)
                bottom = int(imh / 2 - (del_y + dy) / scale)
                right = int((x0 + dx) / scale)
                top = int(imh / 2 - (del_y - dy) / scale)
                crop_window = (left, bottom, right, top)
                print(crop_window)
                ima = get_img_filename(file_to_display, resize=G_SIZE, crop=crop_window)
                print('scale, imw, imh', scale, imw, imh)
                print('crop', crop_window)
                idg = graph.draw_image(data=ima, location=((g00[0] - G_SIZE[0]) // 2, (g00[1] + g00[1]) // 2))
                G_SIZE = g00  # reset zoom factor to original
            window.refresh()

    elif event in ('+', '-'):
        reset_zoom = False
        (g0, g1) = tuple(np.array((g0, g1)) * sqrt2) if event == '+' else tuple(np.array((g0, g1)) / sqrt2)
        G_SIZE = (int(g0), int(g1))
        print(G_SIZE)
        text_string = f'Size {G_SIZE[0]}x{G_SIZE[1]}, {folder}, N = {num_files} '
        direction = None
        graph.delete_figure(idg)
    elif event in ('<', '>'):
        direction = 'left' if event == '<' else 'right'
        graph.delete_figure(idg)
    else:
        print(event, values)

    if direction == 'left':
        offset = (offset + (num_files - 1)) % num_files  # Decrement - roll over to MAX from 0
        graph.delete_figure(idg)
    elif direction == 'right':
        offset = (offset + 1) % num_files  # Increment to MAX then roll over to 0
        graph.delete_figure(idg)

window.close()
