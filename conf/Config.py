class Config:
    polish_coin_classifier_dataset = {
        'training': 'dataset/single-coins/train',
        'validation': 'dataset/single-coins/validation',
        'test': 'dataset/single-coins/test',
        'logs': 'logs/100x100-one-layer-1024',
        'output': 'output/vgg19',
    }

    polish_coin_classifier = {
        'full_layer_size': 1024,
        'full_con_layers': 1,
        'polish_coin_classes_num': 9,
        # 'classifier_model_path': 'output/vgg19/final-model-weights_3.013_150x150.hdf5',
        'image_datatype': 'float32',
        'epochs_training': 60,
        'non_polish_coin_threshold': 0.8,
        'best_scores_multiplier': 2,
    }

    main = {
        'img_dim': (900, 600), # zwiekszyc wielkosc obrazka
        'classifier_image_shape': (80, 80, 3), # TODO zmniejszyc wielkosc okna
        'detected_image': 'dataset/multiple-coins/pink-field/averse-with-non-polish/20220117_100734.jpg',
        'task': 'classifier_training'
    }

    #dla 50x50 val_accuracy ok. 0.18 (jak robie briteness i contrast augmentation to jeszcze gorzej)

    circle_drawer = {
        'bbox_color': (0, 255, 0),
        'draw_non_polish': False
    }

    circle_detection = {
        'min_dist': 50, 'min_radius': 10, 'max_radius': 60
    }
