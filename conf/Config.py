class Config:
    polish_coin_classifier_dataset = {
        'training': 'dataset/single-coins/train',
        'validation': 'dataset/single-coins/validation',
        'test': 'dataset/single-coins/test',
        'logs': 'logs',
        'output': 'output/vgg19',
    }

    polish_coin_classifier = {
        'polish_coin_classes_num': 9,
        'classifier_model_path': 'output/vgg19/final-model-weights_3.013_150x150.hdf5',
        'image_datatype': 'float32',
        'epochs_training': 40,
        'non_polish_coin_threshold': 0.8,
        'best_scores_multiplier': 2,
    }

    main = {
        'img_dim': (600, 400),
        'classifier_image_shape': (150, 150, 3),
        'detected_image': 'dataset/multiple-coins/265814725_854205531939912_7533564023313811302_n.jpg',
        'task': 'full_detection'
    }

    circle_drawer = {
        'bbox_color': (0, 255, 0),
        'draw_non_polish': False
    }

    circle_detection = {
        'min_dist': 50, 'min_radius': 10, 'max_radius': 60
    }
