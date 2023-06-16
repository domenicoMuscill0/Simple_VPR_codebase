
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

    # parser.add_argument("--reweighting", default=False, action=argparse.BooleanOptionalAction,
    #                     help="whether to adopt Contextual Feature Reweighting module")
    parser.add_argument('--reweighting', action='store_true')
    parser.add_argument('--no-reweighting', dest='reweighting', action='store_false')

    # parser.add_argument("--template_injection", default=False, action=argparse.BooleanOptionalAction,
    #                     help="whether to adopt Template Injector module")
    parser.add_argument('--template_injection', action='store_true')
    parser.add_argument('--no-template_injection', dest='template_injection', action='store_false')

    # parser.add_argument("--manifold_loss", default=False, action=argparse.BooleanOptionalAction,
    #                     help="whether to adopt Manifold Loss module")
    parser.add_argument('--manifold_loss', action='store_true')
    parser.add_argument('--no-manifold_loss', dest='manifold_loss', action='store_false')

    # parser.add_argument("--p2s_grad_loss", default=False, action=argparse.BooleanOptionalAction,
    #                     help="whether to adopt P2SGrad Loss module")
    parser.add_argument('--p2s_grad_loss', action='store_true')
    parser.add_argument('--no-p2s_grad_loss', dest='p2s_grad_loss', action='store_false')
	
	  # parser.add_argument("--arcface_loss", default=False, action=argparse.BooleanOptionalAction,
    #                     help="whether to adopt Arcface Loss")
    parser.add_argument('--arcface_loss', action='store_true')
    parser.add_argument('--no-arcface_loss', dest='arcface_loss', action='store_false')
    
    # parser.add_argument("--multisim_loss", default=False, action=argparse.BooleanOptionalAction,
    #                     help="whether to adopt Multisimilarity Loss")
    parser.add_argument('--multisim_loss', action='store_true')
    parser.add_argument('--no-multisim_loss', dest='multisim_loss', action='store_false')
    
    # parser.add_argument("--triplet_loss", default=False, action=argparse.BooleanOptionalAction,
    #                     help="whether to adopt Triplet Loss")
    parser.add_argument('--triplet_loss', action='store_true')
    parser.add_argument('--no-triplet_loss', dest='triplet_loss', action='store_false')
    
    # parser.add_argument("--miner", default=False, action=argparse.BooleanOptionalAction,
    #                     help="whether to adopt Miner")
    parser.add_argument('--miner', action='store_true')
    parser.add_argument('--no-miner', dest='miner', action='store_false')
    
    # Losses params
    parser.add_argument("--arcface_margin", type=float, default=28.6,
                        help="Arcface loss margin")

    parser.add_argument("--arcface_scale", type=int, default=64,
                        help="Arcface loss scale")
	
	  parser.add_argument("--arcface_subcenters", type=int, default=1,
                        help="SubCenterArcFace subcenters")

    parser.add_argument("--contrastive_pos_margin", type=float, default=1,
                        help="Contrastive loss positive margin")
    parser.add_argument("--contrastive_neg_margin", type=float, default=0,
                        help="Contrastive loss negative margin")
    
    parser.add_argument("--multisim_alpha", type=int, default=2,
                        help="MultiSimilarity loss alpha")
    parser.add_argument("--multisim_beta", type=int, default=50,
                        help="MultiSimilarity loss beta")
    parser.add_argument("--multisim_base", type=float, default=0.5,
                        help="MultiSimilarity loss base")
    parser.add_argument("--multisim_miner_epsilon", type=float, default=0.1,
                        help="MultiSimilarity miner epsilon")
    
    parser.add_argument("--triplet_margin", type=float, default=0.05,
                        help="Triplet margin")
    parser.add_argument("--triplet_miner_margin", type=float, default=0.2,
                        help="Triplet miner margin")
    
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

