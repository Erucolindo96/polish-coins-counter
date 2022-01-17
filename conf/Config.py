class Config:
    polish_coin_classifier_dataset = {
        'training': 'dataset/single-coins/train',
        'validation': 'dataset/single-coins/validation',
        'test': 'dataset/single-coins/test',
        'logs': 'logs/80x80-two-layers-1024-full-train-ext-aug',
        'output': 'output/vgg19',
    }

    polish_coin_classifier = {
        'full_layer_size': 1024,
        'full_con_layers': 2,
        'polish_coin_classes_num': 9,
        'classifier_model_path': 'logs/80x80-two-layers-1024-full-train-ext-aug/weights-epoch_30-val_loss_1.12.hdf5',
        'image_datatype': 'float32',
        'epochs_training': 60,
        'non_polish_coin_threshold': 0.8,
        'best_scores_multiplier': 1.5,
    }

    main = {
        'img_dim': (900, 600), # zwiekszyc wielkosc obrazka
        'classifier_image_shape': (80, 80, 3), # TODO zmniejszyc wielkosc okna
        'detected_image': 'dataset/multiple-coins/pink-field/reverse-only/20220117_100547.jpg',
        'task': 'classifier_test'
    }

    circle_drawer = {
        'bbox_color': (0, 255, 0),
        'draw_non_polish': False,
        'subimage_margin': 1
    }

    circle_detection = {
        'min_dist': 50, 'min_radius': 10, 'max_radius': 60
    }
