"""
Show retrieval results.
"""

import re

from coot.configs_retrieval import CootMetersConst, ExperimentTypesConst
from nntrainer import arguments, utils
from nntrainer.view_results import (
    PrintGroupConst, PrintMetric, collect_results_data, output_results, update_performance_profile)


EXP_TYPE = ExperimentTypesConst.RETRIEVAL


class CootPrintGroupConst(PrintGroupConst):
    RETRIEVAL = "retrieval"
    VID = "vid"
    CLIP = "clip"


def main():
    # setup arguments
    parser = utils.ArgParser(description=__doc__)
    arguments.add_multi_experiment_args(parser)  # support multi experiment groups and search
    arguments.add_show_args(parser)  # options for the output table
    arguments.add_path_args(parser, dataset_path=False)  # source path for experiments
    arguments.add_default_args(parser)
    args = parser.parse_args()
    utils.create_logger_without_file(utils.LOGGER_NAME, log_level=args.log_level, no_print=True)

    # find experiments to show depending on arguments
    exp_groups_names = utils.match_folder(args.log_dir, EXP_TYPE, args.exp_group, args.exp_list, args.search)
    collector = collect_results_data(
        EXP_TYPE, exp_groups_names, log_dir=args.log_dir, read_last_epoch=args.last, add_group=args.add_group)
    collector = update_performance_profile(collector)

    # ---------- Define the custom retrieval metrics to show for these experiments ----------

    # define the retrieval validation metrics to show
    retrieval_metrics = {}
    # retrieval validation metrics must be constructed as product of two lists
    re_retrieval_at = re.compile(r"r[0-9]+")
    for modality, shortcut in zip(CootMetersConst.RET_MODALITIES, CootMetersConst.RET_MODALITIES_SHORT):
        # modality: retrieval from where to where
        for metric in CootMetersConst.RET_METRICS:
            # metric: retrieval@1, mean, ...
            if metric == "r1":
                # log r1 metric to the overview class
                metric_class = "val_base"
            else:
                # log all other metrics to the detail class
                metric_class = "val_ret"
            decimals = 2
            formatting = "%" if re_retrieval_at.match(metric) else "f"
            key = f"{metric_class}/{modality}-{metric}"
            print_group = CootPrintGroupConst.VID if "vid" in modality else CootPrintGroupConst.CLIP
            retrieval_metrics[f"{shortcut}-{metric}"] = PrintMetric(key, formatting, decimals, print_group)

    # define average of R@1 text->video, video->text to get a single metric. same for clip->sentence, sentence->clip
    retrieval_metrics["vp-r1"] = PrintMetric("vp-r1", "%", 2, CootPrintGroupConst.RETRIEVAL)
    retrieval_metrics["cs-r1"] = PrintMetric("cs-r1", "%", 2, CootPrintGroupConst.RETRIEVAL)

    # calculate those R@1 averages for each model
    for model_name, metrics in collector.items():
        try:
            metrics["vp-r1"] = (metrics[f"val_base/vid2par-r1"] + metrics[f"val_base/par2vid-r1"]) / 2
            # only calculate average clip-sentence r1 if clips where evaluated
            if f"val_base/cli2sen-r1" in metrics:
                metrics["cs-r1"] = (metrics[f"val_base/cli2sen-r1"] + metrics[f"val_base/sen2cli-r1"]) / 2
        except KeyError as e:
            print(f"WARNING: {e} for {model_name}")

    # ---------- Define which metrics to print ----------
    default_metrics = []
    default_fields = ["v2p-r1", "p2v-r1", "c2s-r1", "s2c-r1", "Time"]
    output_results(collector, custom_metrics=retrieval_metrics, metrics=args.metrics, default_metrics=default_metrics,
                   fields=args.fields, default_fields=default_fields, mean=args.mean, mean_all=args.mean_all,
                   sort=args.sort, sort_asc=args.sort_asc,
                   compact=args.compact)


if __name__ == "__main__":
    main()
