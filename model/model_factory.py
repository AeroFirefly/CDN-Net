def get_model(modelName):
    if modelName.lower() == 'cdnnet':
        from model.model_CDNNet import Res_CBAM_block, CDNNet
        return Res_CBAM_block, CDNNet
    elif modelName.lower() == 'dnanet':
        from model.model_DNANet import Res_CBAM_block, DNANet
        return Res_CBAM_block, DNANet
    else:
        from model.model_CDNNet import Res_CBAM_block, DNANet
        return Res_CBAM_block, DNANet
