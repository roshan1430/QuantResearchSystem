from app.services.ml_service import ml_service
from app.services.timeseries_service import timeseries_service


def get_ml_service():
    return ml_service


def get_timeseries_service():
    return timeseries_service
