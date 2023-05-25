
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64,
                        help="The number of places to use per iteration (one place is N images)")
    parser.add_argument("--img_per_place", type=int, default=4,
                        help="The effective batch size is (batch_size * img_per_place)")
    parser.add_argument("--min_img_per_place", type=int, default=4,
                        help="places with less than min_img_per_place are removed")
    parser.add_argument("--max_epochs", type=int, default=20,
                        help="stop when training reaches max_epochs")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of processes to use for data loading / preprocessing")
    parser.add_argument("--load_checkpoint", default=False, #action=argparse.BooleanOptionalAction,
                        help="whether to load pytorch lightning checkpoints")

    # Architecture parameters
    parser.add_argument("--descriptors_dim", type=int, default=512,
                        help="dimensionality of the output descriptors")
    # parser.add_argument("--feature_mixing", default=False, action=argparse.BooleanOptionalAction,
    #                     help="whether to adopt Feature Mixing module")
    parser.add_argument('--feature_mixing', action='store_true')
    parser.add_argument('--no-feature_mixing', dest='feature_mixing', action='store_false')

    # parser.add_argument("--gpm", default=False, action=argparse.BooleanOptionalAction,
    #                     help="whether to adopt Global Proxy Mining module")
    # Solo per Kaggle. Se usi Colab usa l'altra versione
    parser.add_argument('--gpm', action='store_true')
    parser.add_argument('--no-gpm', dest='gpm', action='store_false')

    # parser.add_argument("--template_injection", default=False, action=argparse.BooleanOptionalAction,
    #                     help="whether to adopt Template Injector module")
    parser.add_argument('--template_injection', action='store_true')
    parser.add_argument('--no-template_injection', dest='template_injection', action='store_false')
    
    # Visualizations parameters
    parser.add_argument("--num_preds_to_save", type=int, default=3,
                        help="At the end of training, save N preds for each query. ")
    parser.add_argument("--num_queries_to_save", type=int, default=10,
                        help="At the end of training, save N queries. ")
    parser.add_argument("--save_only_wrong_preds", action="store_true",
                        help="When saving preds (if num_preds_to_save != 0) save only "
                        "preds for difficult queries, i.e. with uncorrect first prediction")

    # Paths parameters
    parser.add_argument("--train_path", type=str, default="data/gsv_xs/train",
                        help="path to train set")
    parser.add_argument("--val_path", type=str, default="data/sf_xs/val",
                        help="path to val set (must contain database and queries)")
    parser.add_argument("--test_path", type=str, default="data/sf_xs/test",
                        help="path to test set (must contain database and queries)")
    parser.add_argument("--log_path", type=str, default="./LOGS",
                        help="path to store log files of pytorch lightning library and checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default="./LOGS/lightning_logs/version_0/checkpoints/*.ckpt",
                        help="path for loading pytorch lightning checkpoints")

    parser.add_argument("--neptune_api_key", type=str, default="", help="api key for neptune")
    
    args = parser.parse_args()
    return args

