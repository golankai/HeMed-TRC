"""
This file contains the class of the configuration of the model for the TRC task.
"""

from transformers import BertConfig, RobertaConfig


def _init_fn(config, **kwargs):
    config.architecture = kwargs["architecture"]
    config.EMS1 = kwargs["EMS1"]
    config.EMS2 = kwargs["EMS2"]
    config.EME1 = kwargs["EME1"]
    config.EME2 = kwargs["EME2"]
    config.base_lm = config.name_or_path
    config.class_weights = kwargs["class_weights"]


class TRCBertConfig(BertConfig):
    model_type = "BertTemporalRelationClassification"

    def __init__(
        self,
        EMS1=0,
        EMS2=0,
        EME1=0,
        EME2=0,
        architecture=0,
        class_weights=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        _init_fn(
            self,
            EMS1=EMS1,
            EMS2=EMS2,
            EME1=EME1,
            EME2=EME2,
            architecture=architecture,
            class_weights=class_weights,
        )


class TRCRobertaConfig(RobertaConfig):
    model_type = "RobertaTemporalRelationClassification"

    def __init__(
        self,
        EMS1=0,
        EMS2=0,
        EME1=0,
        EME2=0,
        architecture=0,
        class_weights=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        _init_fn(
            self,
            EMS1=EMS1,
            EMS2=EMS2,
            EME1=EME1,
            EME2=EME2,
            architecture=architecture,
            class_weights=class_weights,
        )
