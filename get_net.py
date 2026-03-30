from TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg      

def get_transunet():
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 2
    config_vit.n_skip = 3
    config_vit.patches.grid = (14,14)
    # config_vit.transformer.attention_dropout_rate = 0.3
    # config_vit.transformer.dropout_rate = 0.3
    net = ViT_seg(config_vit, img_size=224, num_classes=1)
    return net


