label_dict = {
    "vessel":1,
    "not_vessel":2,
    "vessel_fishing":1,
    "vessel_not_fishing":2,
    "vessel_not":3,
}


def load_model_config(backbone):
    
    return dict(
        dr_encoder = dict(
            backbone = backbone,
            layer_no = 4,
            layer_names = ("layer1", "layer2", "layer3", "layer4")
        ),
        db_decoder_FA = dict(
            fpn_features_out = 256, 
            dbd_features_out = 128, 
            features_in = (256, 512, 1024, 2048),
            strides = (4, 8, 16, 32)
        ),
        db_decoder_SR = dict(
            no_of_cls = 3, 
            fpn_features_out = 256, 
            dbd_features_out = 128, 
            features_in = (256, 512, 1024, 2048),
            strides = (4, 8, 16, 32)
        )
    )