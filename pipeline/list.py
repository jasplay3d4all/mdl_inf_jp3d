# from pipeline.product_paradise import product_paradise_api
# from pipeline.ad import ad_api
from pipeline import single_state_api


def get_pipeline_list(config):
    return {
        # 'product_paradise' : product_paradise_api.product_paradise(config),
        # 'ad' : ad_api.ad(config),
        'single_state' : single_state_api.single_state(config),
    }