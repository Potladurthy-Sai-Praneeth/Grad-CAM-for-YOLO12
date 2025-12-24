# # DATASET_PATH = "./dataset"
# # OUTPUT_DIR = "./output"
# NUM_CLASSES = 10
# IMGSZ = 256
# BATCH_SIZE = 512
# EPOCHS = 30
# DATASET_YAML = "dataset.yaml"
# MODEL_SIZE = 's'


# label_class_mapping = {
#     'TMHE': 0,
#     'TMBS': 1,
#     'TMEB': 2,
#     'TMLB': 3,
#     'TMLM': 4,
#     'TMSL': 5,
#     'TMSM': 6,
#     'TMTS': 7,
#     'TMYC': 8,
#     'TMMV': 9
# }

# class_name_mapping = {
#   0: 'Healthy',
#   1: 'Bacterial_spot',
#   2: 'Early_blight',
#   3: 'Late_blight',
#   4: 'Leaf_Mold',
#   5: 'Septoria_leaf_spot',
#   6: 'Two-spotted_spider_mite',
#   7: 'Target_Spot',
#   8: 'Yellow_Leaf_Curl_Virus',
#   9: 'Mosaic_virus'
# }

# label_name_mapping = {
#   'TMHE': 'Healthy',
#   'TMBS': 'Bacterial_spot',
#   'TMEB': 'Early_blight',
#   'TMLB': 'Late_blight',
#   'TMLM': 'Leaf_Mold',
#   'TMSL': 'Septoria_leaf_spot',
#   'TMSM': 'Two-spotted_spider_mite',
#   'TMTS': 'Target_Spot',
#   'TMYC': 'Yellow_Leaf_Curl_Virus',
#   'TMMV': 'Mosaic_virus'
# }


# DATASET_PATH = "./dataset"
# OUTPUT_DIR = "./output"
NUM_CLASSES = 3
IMGSZ = 500
BATCH_SIZE = 256
EPOCHS = 40
DATASET_YAML = "dataset.yaml"
MODEL_SIZE = 'n'


label_class_mapping = {
    'Caterpillar': 0,
    'Diabrotica speciosa': 1,
    'Healthy': 2,
}

class_name_mapping = {
  0: 'Caterpillar',
  1: 'Diabrotica speciosa',
  2: 'Healthy',

}

label_name_mapping = {
  'caterpillar': 'Caterpillar',
  'diabroticaspeciosa': 'Diabrotica speciosa',
  'healthy': 'Healthy',
}
