from stable_baselines3.common.policies import ActorCriticPolicy

from src.models.satellite_image_encoder import SatelliteFeatureExtractor


class SatelliteRLPolicy(ActorCriticPolicy):
    """
    Extends the ActorCriticPolicy from SB3, the policy uses the extracted features to make decisions
    about agent movement within the satellite image environment, aiming to locate specific targets.
    """

    def __init__(self, *args, **kwargs):
        super(SatelliteRLPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=SatelliteFeatureExtractor,
            features_extractor_kwargs={"features_dim": 1280},
        )
