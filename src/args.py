from argparse import ArgumentParser

class args_define():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='circo', choices=['cirr', 'circo', 'fashioniq'], help="Dataset to use")
    parser.add_argument("--dataset-path", type=str, help="Path to the dataset", default='CIRCO')
    parser.add_argument("--eval-type", type=str, choices=['LDRE-B', 'LDRE-L', 'LDRE-G'], default='LDRE-L',
                        help="if 'LDRE-B' uses the pre-trained CLIP ViT-B/32, "
                             "if 'LDRE-L' uses the pre-trained CLIP ViT-L/14, "
                             "if 'LDRE-G' uses the pre-trained CLIP ViT-G/14")
    parser.add_argument("--type", type=str, default='L')

    parser.add_argument("--preprocess-type", default="targetpad", type=str, choices=['clip', 'targetpad'],
                        help="Preprocess pipeline to use")
    ### 
    parser.add_argument("--submission-name", type=str, default='test_LDRE', help="Filename of the generated submission file")
    parser.add_argument("--caption_type", type=str, default='opt', choices=['none', 't5', 'opt'], 
                        help="language model of blip-2, 't5' or 'opt'")
    parser.add_argument("--is_gpt_caption", type=bool, default=True)
    parser.add_argument("--is_rel_caption", type=bool, default=True)
    parser.add_argument("--multi_caption", type=bool, default=True)
    parser.add_argument("--nums_caption", type=int, default=15)
    
    parser.add_argument("--use_momentum_strategy", type=bool, default=True)
    parser.add_argument("--momentum_factor", type=float, default=0.3)
    ###
    parser.add_argument("--use_debiased_sample", type=bool, default=True)
    parser.add_argument("--debiased_temperature", type=float, default=0.01)
    
    parser.add_argument("--is_gpt_predicted_features", type=bool, default=False)
    parser.add_argument("--is_blip_predicted_features", type=bool, default=False)
    parser.add_argument("--features_save_path", type=str, default=None)
    

    args = parser.parse_args()

    