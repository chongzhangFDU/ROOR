from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class GeoLayoutLMConfig(PretrainedConfig):

    def __init__(
        self,
        backbone_config,
        backbone="alibaba-damo/geolayoutlm-large-uncased",
        head="vie",
        n_classes=7,
        use_inner_id=True,
        max_prob_as_father=True,
        max_prob_as_father_upperbound=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.backbone_config = backbone_config
        self.backbone = backbone
        self.head = head
        self.n_classes = n_classes
        self.use_inner_id = use_inner_id
        self.max_prob_as_father = max_prob_as_father
        self.max_prob_as_father_upperbound = max_prob_as_father_upperbound
