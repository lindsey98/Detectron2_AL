from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import ProposalNetwork, GeneralizedRCNN
import torch

__all__ = ['ActiveLearningRCNN']


@META_ARCH_REGISTRY.register()
class ActiveLearningRCNN(GeneralizedRCNN):
    
    def __init__(self, cfg):
        
        super(ActiveLearningRCNN, self).__init__(cfg)
        self.cfg = cfg
    
#     def _perturb_image(self, images):
        
#         if self.cfg.AL.OBJECT_SCORING == 'location': # N=6 noise levels
            
#         elif self.cfg.AL.OBJECT_SCORING == 'consistency': # horizontally flip the images  
             
#         else:
#             raise NotImplementedError
            
        
    def _estimate_feature_proposal(self, batched_inputs):
        
        with torch.no_grad():
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            if self.proposal_generator:
                rpn_proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                rpn_proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        return features, rpn_proposals

    def generate_object_al_scores(self, batched_inputs, do_postprocess=False):
        
        with torch.no_grad():
            features, rpn_proposals = self._estimate_feature_proposal(batched_inputs)

        return  self.roi_heads.generate_object_scores(features, rpn_proposals)
    
    def generate_image_al_scores(self, batched_inputs):
        """
        Returns: List[Tuple]
            A list of (image_score, image_id) tuple
        """
        with torch.no_grad():
            features, rpn_proposals = self._estimate_feature_proposal(batched_inputs)

        image_scores = self.roi_heads.generate_image_scores(features, rpn_proposals)

        return [(score, gt['image_id']) for score, gt in zip(image_scores, batched_inputs)]

    def forward_al(self, batched_inputs, do_postprocess=True):

        """
        Run inference on the given inputs with active learning.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        
        with torch.no_grad():
            features, rpn_proposals = self._estimate_feature_proposal(batched_inputs)
            
#         if self.cfg.AL.OBJECT_SCORING == 'location' or self.cfg.AL.OBJECT_SCORING == 'consistency':
#             # generate perturbed images/horizontally flipped images etc.
#             augment_images = self._perturb_image(images)

        detection_results = self.roi_heads.generate_object_scores(features, rpn_proposals)
    
        if do_postprocess:
            detection_results = GeneralizedRCNN._postprocess(detection_results, batched_inputs, images.image_sizes) 
        
        return detection_results