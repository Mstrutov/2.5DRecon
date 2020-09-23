import os
from ij import IJ
from ij import plugin
from ij import process
stacks_dir = "D:/_MEL_Science/datasets/thick_specimen/20_frames/"
results_dir = "D:/_MEL_Science/results/thick-specimen/"
stack_name = os.listdir(stacks_dir)[19]
stack_dir = stacks_dir + stack_name
#print type(os.listdir(source_dir)[0])
#print source_dir
#imp = IJ.openImage(source_dir + "/frame120.jpg")
#print imp

imps = plugin.FolderOpener().open(stack_dir)
#imps.setDisplayMode(IJ.GRAYSCALE)
#print(imps.getType())
#print(imps.getPixel(200, 200))
print(imps)
print(imps.isDisplayedHyperStack())
imps.show()
converter = process.ImageConverter(imps)
converter.convertToGray8()

os.mkdir(results_dir + stack_name)
#imps.show()