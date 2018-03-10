import matplotlib
from matplotlib import pyplot
from skimage import io, filters, color, measure, feature

print('All packages loaded successfully >>')

#filepath = input('Enter file name >>')

#load image and convert to gray
image1 = color.rgb2gray(io.imread('ISIC_0000013_30_f_m.jpg'))
image2 = color.rgb2gray(io.imread('ISIC_0000029_45_f_m.jpeg'))
image3 = color.rgb2gray(io.imread('ISIC_0000034_30_f_b.jpg'))
image4 = color.rgb2gray(io.imread('ISIC_0011415_55_m_b.jpg'))
print('Successfully loaded images and converted to grayscale >>')

# otsu thresholding function
def OtsuThreshold(graycolouredimage):
    otsuVal = filters.threshold_otsu(graycolouredimage)
    binary = graycolouredimage > otsuVal
    print('Successfully applied otsu method to threshold images >>')
    print('Dimensional Information: ', otsuVal.shape, 'for Otsu Value', binary.shape, 'for Binarization Array')
    print('Otsu Value Type:', type(otsuVal), 'Binarization Array Type:', type(binary))
    pyplot.imshow(binary, cmap=pyplot.cm.gray)
    pyplot.show()
    ots = [binary, otsuVal]
    return ots;

# compute perimeter
def ComputePerimeter(binarizedImage):
    perim = measure.perimeter(binarizedImage)
    print('Perimeter calculation complete >>')
    print('Perimeter is:', perim, 'pixels.')
    print('Dimensions of preimeter variable is:', perim.shape)
    print('Perimeter Type:', type(perim))
    return perim;

# use canny algorithm to edge filter
def CannyFilter(grayscaledimage):
    canny = feature.canny(grayscaledimage, sigma=0.75)
    print('Edge detected using canny algorithm >>')
    print('Diemnsional Information for Canny Array:', canny.shape)
    print('Canny Array Type:', type(canny))
    pyplot.imshow(canny, cmap=pyplot.cm.gray)
    pyplot.show()
    return canny;

# find contours
def FindContour(grayImage):
    cont = measure.find_contours(grayImage, level = 0.5)
    print('Contours of the image were evaluated >>')
    fig, ax = pyplot.subplots()
    ax.imshow(grayImage, interpolation='nearest', cmap=pyplot.cm.gray)
    for n, contr in enumerate(cont):
        ax.plot(contr[:,1], contr[:,0], linewidth=2)
    ax.axis('Image')
    ax.set_xticks([])
    ax.set_yticks([])
    #print('Dimensional Information for Contour Array:', cont.shape)
    print('Contour Array Type:', type(cont))
    pyplot.show()
    return cont;


#main program

while True:
    print('To exit, enter "done".')
    filepath = input('Enter file path >> ');

    if filepath == 'done':
        print ('Exiting >>');
        break
    try:
        image = color.rgb2gray(io.imread(filepath))
    except:
        print('Invalid input or filepath not found.')
        continue

    otsu = OtsuThreshold(image)
    print(otsu)

    perimeter = ComputePerimeter(otsu[0])

    cannyEdge = CannyFilter(image)
    print(cannyEdge)

    contours = FindContour(image)
    print(contours)

    #create Feature Vector
    featureVector = [perimeter, otsu[0], otsu[1], cannyEdge, contours]
    print('Feature vector created >>')
    print('Length of Feature Vector is:',len(featureVector))

    #save featureVector to a .txt file
    print('Saving Feature Vector as .txt file >>')
    with open('featVect.txt', 'w') as f:
        for s in featureVector:
            f.write(str(s)+"\n")
    print('.txt file saved >>')

#print(type(image2))



# use canny algorithm for edge detection


# Basic Algorithm
# take input image
# resie it to a fixed height x width
# take single set of pixel intensities
# 32 (height) x 32 (width) x 3 (RGB)
