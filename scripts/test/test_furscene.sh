python ../../src/test_cluster.py --dataDir ../../data/furscene/Test/Rot_Light_PL --logDir ../../test_results/furscene/Rot_light_PL --checkpoint ../../models/furscene --max_steps 360  --data_max_val 1.000000 --rescale_input 1.000000 --rescale_output 1.000000 --indexDir ../../data/furscene/Test/Rot_Light_PL/views_fix_repeat_360_nearby_indices.txt   

python ../../src/test_cluster.py --dataDir ../../data/furscene/Test/Rot_View_PL --logDir ../../test_results/furscene/Rot_View_PL --checkpoint ../../models/furscene/ --max_steps 360  --data_max_val 1.000000 --rescale_input 1.000000 --rescale_output 1.000000 --indexDir ../../data/furscene/Test/Rot_View_PL/views_RotY_360_nearby_indices.txt    

python ../../src/test_cluster.py --dataDir ../../data/furscene/Test/Rot_Light_Env5 --logDir ../../test_results/furscene/Rot_Light_Env5/raw_output --checkpoint ../../models/furscene --max_steps 360  --data_max_val 1.000000 --rescale_input 0.320000 --rescale_output 1.000000 --indexDir ../../data/furscene/Test/Rot_Light_Env5/views_fix_repeat_360_nearby_indices.txt   

python ../../src/test_mask.py --uvDir ../../data/furscene/Test/Rot_Light_Env5/UV --logDir  ../../test_results/furscene/Rot_Light_Env5 --checkpoint ../../models_mask/furscene --begin 0 --end 360 --imageDir ../../test_results/furscene/Rot_Light_Env5/raw_output

python ../../src/test_cluster.py --dataDir ../../data/furscene/Test/Rot_View_Env5 --logDir ../../test_results/furscene/Rot_View_Env5/raw_output --checkpoint ../../models/furscene --max_steps 360  --data_max_val 1.000000 --rescale_input 0.320000 --rescale_output 1.000000 --indexDir ../../data/furscene/Test/Rot_View_Env5/views_RotY_360_nearby_indices.txt    

python ../../src/test_mask.py --uvDir ../../data/furscene/Test/Rot_View_Env5/UV --logDir  ../../test_results/furscene/Rot_View_Env5 --checkpoint ../../models_mask/furscene --begin 0 --end 360 --imageDir ../../test_results/furscene/Rot_View_Env5/raw_output