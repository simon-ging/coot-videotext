"""
Train captioning with MART.

Originally published by https://github.com/jayleicn/recurrent-transformer under MIT license
Reworked by https://github.com/gingsi/coot-videotext under Apache 2 license
"""

from coot.configs_retrieval import ExperimentTypesConst
from mart import arguments_mart
from mart.configs_mart import MartConfig as Config
from mart.model import create_mart_model
from mart.recursive_caption_dataset import create_mart_datasets_and_loaders
from mart.trainer_mart import MartTrainer
from nntrainer import arguments, utils
from nntrainer.utils_torch import set_seed
from nntrainer.utils_yaml import load_yaml_config_file


EXP_TYPE = ExperimentTypesConst.MART


def main():
    # ---------- Setup script arguments. ----------
    parser = utils.ArgParser(description=__doc__)
    arguments.add_default_args(parser)  # logging level etc.
    arguments.add_exp_identifier_args(parser)  # arguments to identify the experiment to run
    arguments.add_trainer_args(parser, dataset_path=False)  # general trainer arguments
    parser.add_argument("--preload", action="store_true", help="Preload everything.")  # feature preloading
    arguments_mart.add_mart_args(parser)  # some more paths for mart
    parser.add_argument("--load_model", type=str, default=None, help="Load model from file.")
    parser.add_argument("--print_model", action="store_true", help=f"Print model")
    args = parser.parse_args()

    # load repository config yaml file to dict
    exp_group, exp_name, config_file = arguments.setup_experiment_identifier_from_args(args, EXP_TYPE)
    config = load_yaml_config_file(config_file)

    # update experiment config given the script arguments
    config = arguments.update_config_from_args(config, args)
    config = arguments_mart.update_mart_config_from_args(config, args)

    # read experiment config dict
    cfg = Config(config)
    if args.print_config:
        print(cfg)

    # set seed
    if cfg.random_seed is not None:
        print(f"Set seed to {cfg.random_seed}")
        set_seed(cfg.random_seed, set_deterministic=False)  # set deterministic via config if needed

    # create dataset
    train_set, val_set, train_loader, val_loader = create_mart_datasets_and_loaders(
        cfg, args.coot_feat_dir, args.annotations_dir, args.video_feature_dir)

    for i, run_number in enumerate(range(args.start_run, args.start_run + args.num_runs)):
        run_name = f"{args.run_name}{run_number}"

        # create model from config
        model = create_mart_model(cfg, len(train_set.word2idx), cache_dir=args.cache_dir)

        # print model for debug if requested
        if args.print_model and i == 0:
            print(model)

        # always load best epoch during validation
        load_best = args.load_best or args.validate

        # create trainer
        trainer = MartTrainer(
            cfg, model, exp_group, exp_name, run_name, len(train_loader), log_dir=args.log_dir,
            log_level=args.log_level, logger=None, print_graph=args.print_graph, reset=args.reset, load_best=load_best,
            load_epoch=args.load_epoch, load_model=args.load_model, inference_only=args.validate,
            annotations_dir=args.annotations_dir)

        if args.validate:
            # run validation
            if not trainer.load and not args.ignore_untrained:
                raise ValueError("Validating an untrained model! No checkpoints were loaded. Add --ignore_untrained "
                                 "to ignore this error.")
            trainer.validate_epoch(val_loader)
        else:
            # run training
            trainer.train_model(train_loader, val_loader)

        # done with this round
        trainer.close()
        del model
        del trainer


if __name__ == "__main__":
    main()
