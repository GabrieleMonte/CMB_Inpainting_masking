from CMB_Inpainting_PConv_project.Mask_generator import MaskGenerator
import matplotlib.pyplot as plt
from os import makedirs

# make folder for results
makedirs('mask_templates_single', exist_ok=True)
makedirs('mask_templates_unified', exist_ok=True)

# Instantiate mask generator
mask_generator = MaskGenerator(512, 512, 3, rand_seed=42, filepath=None)


###############################################
#METHOD 1: PLOT EACH MASK IN A DIFFERENT FIGURE
###############################################
axes=25
for i in range(axes):
    # Generate image
    img = mask_generator.sample()
    #plot each image on a different png file
    plt.imshow(img*255)
    plt.savefig('mask_templates_single/mask_templates_single_'+ str(i+1)+'.png')

################################################################
#METHOD 2: PLOT A SET OF n MASKS IN A SINGLE FIGURE (HERE n=25)
################################################################

# Plot the results
#_, axes = plt.subplots(5, 5, figsize=(20, 20))
#axes = list(itertools.chain.from_iterable(axes))

#for i in range(len(axes)):
    # Generate image
    #img = mask_generator.sample()
    #Plot image on axis
    #axes[i].imshow(img * 255)

#plt.savefig('mask_templates_unified/mask_templates_unified_1.png')


################################################################################
#VARIANT 1= LOAD MASKS FROM DIRECTORY AND AUGMENT THEM (works only with method 2)
#################################################################################

#mask_generator = MaskGenerator(512, 512, 3, rand_seed=42, filepath='./mask_templates_unified/')


# Plot the results
#_, axes = plt.subplots(5, 5, figsize=(20, 20))
#axes = list(itertools.chain.from_iterable(axes))

#for i in range(len(axes)):
    # Generate image
    #img = mask_generator.sample()
    #Plot image on axis
    #axes[i].imshow(img * 255)

#plt.savefig('mask_templates_unified/mask_templates_augmented_1.png')