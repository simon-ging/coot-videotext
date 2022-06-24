"""
Show retrieval results.
"""

from coot.configs_retrieval import ExperimentTypesConst
from nntrainer import arguments, utils
from nntrainer.metric import TEXT_METRICS
from nntrainer.view_results import collect_results_data, output_results, update_performance_profile


EXP_TYPE = ExperimentTypesConst.CAPTION


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

    # ---------- Define which metrics to print ----------
    default_metrics = []
    default_fields = ["bleu4", "meteo", "rougl", "cider", "re4"]
    output_results(collector, custom_metrics=TEXT_METRICS, metrics=args.metrics, default_metrics=default_metrics,
                   fields=args.fields, default_fields=default_fields, mean=args.mean, mean_all=args.mean_all,
                   sort=args.sort, sort_asc=args.sort_asc,
                   compact=args.compact)


if __name__ == "__main__":
    main()
