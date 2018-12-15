#-----------------------------------------------------------------------------
# Generator

# for displaying the target image, intermittently (default: 400)
SHOW_EVERY = 2 

# decide how many iterations to update your image (default: 5000)
STEPS = 4

# weights
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 1e6

#-----------------------------------------------------------------------------
# VGG19

VGG19_FEATURE_LAYERS = {
'0': 'conv1_1',
'5': 'conv2_1',
'10': 'conv3_1',
'19': 'conv4_1',
'21': 'conv4_2',
'28': 'conv5_1'
}

# weights for each style layer
# weighting earlier layers more will result in larger style artifacts
# notice we are excluding conv4_2 content representation
VGG19_STYLE_WEIGHTS = {
'conv1_1': 1.,
'conv2_1': 0.75,
'conv3_1': 0.2,
'conv4_1': 0.2,
'conv5_1': 0.2
}

