from pydantic import BaseSettings
import logging
import sys
import mlflow


class NLPSettings(BaseSettings):
    log_level: str = 'INFO'

    # mlflow config
    mlflow_pg_url: str = "postgresql://postgres:AKIA2QRVYAA7CGPVTUQX@dev-tip.cupzbhodsxus.rds.cn-northwest-1.amazonaws.com.cn:5432/mlflow"
    mlflow_artifact_url: str = "s3://mesoor-models/mesoorflow-dev"
    local_artifact_path: str = "./experiment_log"


mf_settings = NLPSettings()

log_fmt = '%(asctime)s %(levelname)s %(message)s'
log_level = logging.getLevelName(mf_settings.log_level)
logging.basicConfig(level=log_level, format=log_fmt)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(consoleHandler)
logger.propagate = False

logger.info(f"{mf_settings}")

mlflow.set_tracking_uri(mf_settings.mlflow_pg_url)
logger.info(f"set mlflow url to {mf_settings.mlflow_pg_url}")
