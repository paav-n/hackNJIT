from PIL import Image

def flavorify(color_obj):
    r = color_obj[0]
    b = color_obj[1]
    g = color_obj[2]

    """
    if r > b and r > g:
        if .54 < (g/r) < 1.0:
            return 'yellow'
        if .1 < (g/r) < .54:
            return 'orange'
        return 'red'
    if b > r and b > g:
        return 'blue'
    if g > r and g > b:
        return 'green'
    return 'brown'
    """

    if b > r:
        return 'purple'
    if g > r:
        return 'green'
    if b > g:
        return 'red'
    if 2 * g > r:
        return 'yellow'
    return 'orange'

# Reads image
img = Image.open('image.jpg')
colorsets = img.getcolors()
for item in colorsets.colors():
    colorset.append(item)
avgcolor = sum(colorset)/len(colorset)
print(flavorify(avgcolor))
