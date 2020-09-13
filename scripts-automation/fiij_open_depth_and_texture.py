import os
from ij import IJ
from ij import plugin
from ij import process
source_dir = "D:/_MEL_Science/results/HD. Human Hair by-hand"
#print type(os.listdir(source_dir)[0])
#print source_dir
im = IJ.openImage(source_dir + "/Texture-ZS.jpg")
im.show()

dm = IJ.openImage(source_dir + "/Depthmap-OTSU_2.png")
dm.show()




IJ.run("Interactive 3D Surface Plot")

#print imp

#imps = plugin.FolderOpener().open(source_dir)
#imps.setDisplayMode(IJ.GRAYSCALE)
#print(imps.getType())
#print(imps.getPixel(200, 200))
#print(imps)
#print(imps.isDisplayedHyperStack())
#imps.show()
#converter = process.ImageConverter(imps)
#converter.convertToGray8()
#imps.show()