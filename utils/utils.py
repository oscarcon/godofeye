def crop_image(img):
    height, width, color_channel = img.shape
    width_range = slice(width//2-width//8, width//2+width//8)
    height_range = slice(int(0.2*height),int(0.8*height))
    img = img[height_range, width_range, :]
    return img