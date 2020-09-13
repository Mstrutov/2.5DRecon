import os
from ij import IJ
from ij import plugin
from ij import process
source_dir = "D:/_MEL_Science/datasets/20"

#print type(os.listdir(source_dir)[0])
print source_dir
#imp = IJ.openImage(source_dir + "/frame120.jpg")
#print imp

imps = plugin.FolderOpener().open(source_dir)
#imps.setDisplayMode(IJ.GRAYSCALE)
#print(imps.getType())
#print(imps.getPixel(200, 200))
print(imps)
print(imps.isDisplayedHyperStack())
imps.show()
converter = process.ImageConverter(imps)
converter.convertToGray8()
#imps.show()