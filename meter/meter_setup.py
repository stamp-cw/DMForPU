from selector.meter_selector import _METERS

class MeterSetup:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self._create_meter()
        self._log_meter_info()

    def _create_meter(self):
        self.meter = _METERS[self.config.metric.name](self.config)

    def _log_meter_info(self):
        self.logger.debug(self.meter)