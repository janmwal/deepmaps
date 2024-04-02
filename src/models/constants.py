
models_meta = { 
    'vta_net' : {
        'class_name' :'VTANet',
        'print_name' :'VTANET',
        'dataset_module_name' : 'vta_dataset',
        'dataset_class_name' : 'VTADataset',
        'criterion' :'BCEWithLogitsLoss',
        'versions' : {
            'V1' : {
                'optimizer' : 'Adam',
                'learning_rate' : 1e-3,
                'batch_size' : 32,
                'conv_channels' : (32, 16, 8),
                'fc_channels' : (128, 64),
                'dropout' : (0.25, 0.),
                'label_threshold' : 0.99
            },
            'V2' : {
                'optimizer' : 'Adam',
                'learning_rate' : 1e-3,
                'batch_size' : 32,
                'conv_channels' : (32, 16, 8, 8),
                'fc_channels' : (64, 32),
                'dropout' : (0., 0.),
                'label_threshold' : 0.99
            }, 
            'ResAgnostic' : {
                'optimizer' : 'Adam',
                'learning_rate' : 1e-5,
                'batch_size' : 16,
                'conv_channels' : (8, 4, 4, 2),
                'fc_channels' : (64, 32),
                'dropout' : (0., 0.),
                'label_threshold' : 0.99
            },
            'OptiV1' : {
                'optimizer' : 'Adam',
                'learning_rate' : 1.2e-5,
                'batch_size' : 4,
                'conv_channels' : (10,),
                'fc_channels' : (42, 19, 30, 23),
                'dropout' : (0.25, 0.25, 0.25, 0.25),
                'label_threshold' : 0.61,
                'resolution' : 500,
                'augmentation_type' : 'augmented_full',
                'interpolation' : 'linear_interp'
            },
            'OptiV1Batch32' : {
                'optimizer' : 'Adam',
                'learning_rate' : 1.2e-5,
                'batch_size' : 32,
                'conv_channels' : (10,),
                'fc_channels' : (42, 19, 30, 23),
                'dropout' : (0.25, 0.25, 0.25, 0.25),
                'label_threshold' : 0.61,
                'resolution' : 500,
                'augmentation_type' : 'augmented_full',
                'interpolation' : 'linear_interp'
            },
            'OptiV2' : {
                'optimizer' : 'Adam',
                'learning_rate' : 7.66e-5,
                'batch_size' : 16,
                'conv_channels' : (7, 10 , 13, 16),
                'fc_channels' : (16, 56),
                'dropout' : (0.10, 0.27),
                'label_threshold' : 0.74,
                'resolution' : 500,
                'augmentation_type' : 'augmented_full',
                'interpolation' : 'linear_interp',
                'weight_decay' : 0.076,
                'alpha_gl' : 1.44,
                'lambda_coeff' : 0.52
            }
        }
    },
    'props_net' : { 
        'class_name' : 'PropsNet',
        'print_name' : 'PROPSNET',
        'dataset_module_name' : 'props_dataset',
        'dataset_class_name' : 'PropsDataset',
        'criterion' :'BCEWithLogitsLoss',
        'versions' : {
            'V1' : {
                'optimizer' : 'Adam',
                'learning_rate' : 1e-2,
                'batch_size' : 32,
                'fc_channels' : (64, 32),
                'dropout' : (0.25, 0.25),
                'label_threshold' : 0.99,
                'resolution' : 250,
                'augmentation_type' : 'augmented_part',
                'interpolation' : 'step_interp'
            },
            'V2' : {
                'optimizer' : 'Adam',
                'learning_rate' : 1e-2,
                'batch_size' : 32,
                'fc_channels' : (24,),
                'dropout' : (0.25,),
                'label_threshold' : 0.99,
                'resolution' : 250,
                'augmentation_type' : 'augmented_part',
                'interpolation' : 'step_interp'
            },
            'Opti1' : {
                'optimizer' : 'Adam',
                'learning_rate' : 2.78e-5,
                'batch_size' : 32,
                'fc_channels' : (40,),
                'dropout' : (0.20,),
                'label_threshold' : 0.63,
                'resolution' : 250,
                'augmentation_type' : 'augmented_full',
                'interpolation' : 'step_interp',
                'weight_decay' : 0.023,
                'alpha_gl' : 9.0,
                'lambda_coeff' : 0.96
            }
        }
    },
    'proj_net' : { 
        'class_name' : 'ProjNet',
        'print_name' : 'PROJNET',
        'dataset_module_name' : 'proj_dataset',
        'dataset_class_name' : 'ProjectionDataset',
        'criterion' :'BCEWithLogitsLoss',
        'versions' : { 
            'V1' : {
                'optimizer' : 'Adam',
                'learning_rate' : 1e-5,
                'batch_size' : 64,
                'conv_channels' : (32, 32, 16),
                'fc_channels' : (128, 64),
                'dropout' : (0.25, 0.25),
                'label_threshold' : 0.99
            }, 
            'Opti1' : {
                'optimizer' : 'Adamax',
                'learning_rate' : 3.5e-5,
                'batch_size' : 64,
                'conv_channels' : (16,),
                'fc_channels' : (256, 8, 16),
                'dropout' : (0.25, 0.2, 0.2),
                'label_threshold' : 0.99
            }, 
            'Opti2' : {
                'optimizer' : 'Adam',
                'learning_rate' : 7.2e-5,
                'batch_size' : 64,
                'conv_channels' : (4, 16, 128, 32),
                'fc_channels' : (8, 64),
                'dropout' : (0.35, 0.2),
                'label_threshold' : 0.99
            }, 
            'Opti3' : {
                'optimizer' : 'Adam',
                'learning_rate' : 2.2e-4,
                'batch_size' : 16,
                'conv_channels' : (128, 16, 64, 64, 4),
                'fc_channels' : (8,),
                'dropout' : (0.25,),
                'label_threshold' : 0.99
            },
            'Opti4' : {
                'optimizer' : 'Adam',
                'learning_rate' : 4.1e-5,
                'batch_size' : 8,
                'conv_channels' : (100, 70, 120, 35, 16),
                'fc_channels' : (50,),
                'dropout' : (0.25,),
                'label_threshold' : 0.6,
                'resolution' : 250,
                'augmentation_type' : 'augmented_part',
                'interpolation' : 'step_interp'
            },
            'Opti5' : {
                'optimizer' : 'Adam',
                'learning_rate' : 3.2e-5,
                'batch_size' : 4,
                'conv_channels' : (128, 90, 128, 50, 5),
                'fc_channels' : (120,),
                'dropout' : (0.25,),
                'label_threshold' : 0.59,
                'resolution' : 250,
                'augmentation_type' : 'augmented_part',
                'interpolation' : 'step_interp',
                'epochs' : 5
            },
            'Opti6' : {
                'optimizer' : 'Adam',
                'learning_rate' : 6e-5,
                'batch_size' : 4,
                'conv_channels' : (90, 90, 90, 70, 30),
                'fc_channels' : (90,),
                'dropout' : (0.25,),
                'label_threshold' : 0.59,
                'resolution' : 250,
                'augmentation_type' : 'augmented_part',
                'interpolation' : 'step_interp',
                'weight_decay' : 0.15,
                'alpha_gl' : 19.5,
                'lambda_coeff' : 0.75
            },
            'Merged2' : {
                'optimizer' : 'Adam',
                'label_threshold' : 0.9244687670750145, 
                'interpolation' : 'step_interp', 
                'noise_factor' : 8.88096856012708, 
                'learning_rate' : 0.000270880756917335, 
                'batch_size' : 128, 
                'resolution' : 250,
                'augmentation_type' : 'augmented_full',
                'weight_decay' : 0.016069808295861163, 
                'lambda_coeff' : 0.6548214598709491, 
                'alpha_gl' : 0.8343982195623596, 
                'conv_channels' : (140, 55),
                'fc_channels' : (90, 70),
                'dropout' : (0.15, 0.15),
                'patience' : 3,
                'epochs' : 12
            },
            'Merged3' : {
                'optimizer' : 'Adam',
                'label_threshold' : 0.9929490446255225, 
                'interpolation' : 'step_interp', 
                'noise_factor' : 4.771723703613423, 
                'learning_rate' : 8.308742771905538e-05, 
                'batch_size' : 256, 
                'resolution' : 500,
                'augmentation_type' : 'augmented_full',
                'weight_decay' : 0.03607266420460243, 
                'lambda_coeff' :  0.8832262925514519, 
                'alpha_gl' : 3.2444406526148613, 
                'conv_channels' : (75, 36, 105, 149),
                'fc_channels' : (141,),
                'dropout' : (0.21613114278688128,),
                'patience' : 3,
                'epochs' : 12
            },
            'Merged4' : {
                'optimizer' : 'Adam',
                'label_threshold' : 0.8330957480952411, 
                'interpolation' : 'step_interp', 
                'noise_factor' : 23.41944493202871, 
                'learning_rate' : 0.0003146887491673534, 
                'batch_size' : 128, 
                'resolution' : 1000,
                'augmentation_type' : 'augmented_full',
                'weight_decay' : 0.009239266314472397, 
                'lambda_coeff' :  0.5259890112459549, 
                'alpha_gl' : 83.53151930143831, 
                'conv_channels' : (20, 144, 104, 18, 89),
                'fc_channels' : (126, 93),
                'dropout' : (0.18305708098284967, 0.18305708098284967),
                'patience' : 3,
                'epochs' : 19
            },
            'Bern1' : {
                'optimizer' : 'Adam',
                'label_threshold' : 0.9609441191501632, 
                'interpolation' : 'step_interp', 
                'noise_factor' : 10.739046350531009, 
                'learning_rate' : 0.0004707558090574162, 
                'batch_size' : 256, 
                'resolution' : 1000,
                'augmentation_type' : 'augmented_full',
                'weight_decay' : 0.06541961183714957, 
                'lambda_coeff' :  0.6125613507779666, 
                'alpha_gl' : 90.87621980724451, 
                'conv_channels' : (104, 89, 92, 128, 79),
                'fc_channels' : (143,),
                'dropout' : (0.26352575984096915,),
                'patience' : 1,
                'epochs' : 4
            }
        }
    },
    'resnet18' : {
        'class_name' : 'ResNet18',
        'print_name' : 'RESNET18',
        'dataset_module_name' : 'proj_dataset',
        'dataset_class_name' : 'ProjectionDataset',
        'criterion' :'BCEWithLogitsLoss',
        'versions' : ('V1',)
    }
}