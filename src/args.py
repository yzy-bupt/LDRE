from argparse import ArgumentParser

class args_define():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='circo', choices=['cirr', 'circo', 'fashioniq'], help="Dataset to use")
    parser.add_argument("--dataset-path", type=str, help="Path to the dataset", default='CIRCO')
    parser.add_argument("--eval-type", type=str, choices=['LDRE-B', 'LDRE-L', 'LDRE-G'], default='LDRE-G',
                        help="if 'LDRE-B' uses the pre-trained CLIP ViT-B/32, "
                             "if 'LDRE-L' uses the pre-trained CLIP ViT-L/14, "
                             "if 'LDRE-G' uses the pre-trained CLIP ViT-G/14")
    parser.add_argument("--preprocess-type", default="targetpad", type=str, choices=['clip', 'targetpad'],
                        help="Preprocess pipeline to use")
    ### 
    parser.add_argument("--submission-name", type=str, default='test', help="Filename of the generated submission file")
    parser.add_argument("--caption_type", type=str, default='opt', choices=['none', 't5', 'opt'], 
                        help="language model of blip-2, 't5' or 'opt'")
    parser.add_argument("--use_gpt_caption", type=bool, default=True)
    parser.add_argument("--use_rel_caption", type=bool, default=True)
    parser.add_argument("--multi_caption", type=bool, default=True)
    parser.add_argument("--nums_caption", type=int, default=15)
    
    ### 
    parser.add_argument("--use_semantic_relevance_score", type=bool, default=True,
                        help="True: calculate features for both reference captions (blip_predicted_features)"
                             "and edited captions (gpt_predicted_features)."
                             "False: calculate features for the edited caption only."
                             "Use True for LDRE.")
    parser.add_argument("--momentum_factor", type=float, default=0.0, help="TODO for SEIZE, can be ignored for now")
    
    ### 
    parser.add_argument("--use_adaptive_ensemble", type=bool, default=True,
                        help="Whether to calculate weights for each pair of captions. True for LDRE")
    parser.add_argument("--adaptive_temperature", type=float, default=0.04)
    parser.add_argument("--adaptive_topk", type=int, default=5, 
                        help="Select the top k largest differences as weights, with k=5 to 50 yielding the best results")
    
    parser.add_argument("--use_gpt_predicted_features", type=bool, default=False,
                        help="Whether to use predicted features for edited captions by GPT. False for the first time")
    parser.add_argument("--use_blip_predicted_features", type=bool, default=False,
                        help="Whether to use predicted features for reference captions by BLIP2. False for the first time")
    parser.add_argument("--save_features", type=bool, default=True)
    

    args = parser.parse_args()

    