from tensorflow.python.estimator.exporter import Exporter


class ReportExporter(Exporter):

    def __init__(self, reporter, report_metrics: list):
        self._reporter = reporter
        self._report_metrics = report_metrics

    @property
    def name(self):
        return 'report'

    def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):
        global_step = int(checkpoint_path.split('-')[-1])
        metrics = {metric_name.replace('/', '_'): eval_result[metric_name]
                   for metric_name in self._report_metrics}

        self._reporter(**metrics,
                       timesteps_total=global_step,
                       checkpoint=checkpoint_path)
