class Config:
    polish_coin_classifier_dataset = {
        'training': 'dataset/single-coins/train',
        'validation': 'dataset/single-coins/validation',
        'test': 'dataset/single-coins/test',
        'logs': 'logs/80x80-two-layers-1024-hist-stretch',
        'output': 'output/vgg19',
    }

    polish_coin_classifier = {
        'full_layer_size': 1024,
        'full_con_layers': 2,
        'polish_coin_classes_num': 9,
        'classifier_model_path': 'changeMe',
        # najlepsze wychodzą 80x80-two-layers-1024, moze bez pełnej augmentacji
        'image_datatype': 'float32',
        'epochs_training': 60,
        'non_polish_coin_threshold': 0.8,
        'best_scores_multiplier': 1.5,
    }

    main = {
        'img_dim': (900, 600),
        'classifier_image_shape': (80, 80, 3),
        'detected_image': 'changeMe',
        'task': 'full_detection'
    }

    font = {
        'path': '/usr/share/fonts/truetype/freefont/FreeMono.ttf'
    }

    circle_drawer = {
        'bbox_color': (0, 255, 0),
        'draw_non_polish': True,
        'subimage_margin': 0
    }

    circle_detection = {
        'min_dist': 50, 'min_radius': 10, 'max_radius': 60
    }

    image_scaler = {
        'stretching': None  # 'stretching': (41, 254)
    }
