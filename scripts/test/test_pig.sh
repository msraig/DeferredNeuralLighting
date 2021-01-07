python ../../src/test_cluster.py --dataDir ../../data/pig/Test/pl_rotate --logDir ../../test_results/pig/pl_rotate --checkpoint ../../models/pig  --max_steps 360 --data_max_val 3.834000 --rescale_input 1.000000 --rescale_output 1.000000 --indexDir ../../data/pig/Test/pl_rotate/test_indices_60.txt 


python ../../src/test_cluster.py --dataDir ../../data/pig/Test/pl_rotate_both --logDir ../../test_results/pig/pl_rotate_both --checkpoint ../../models/pig  --max_steps 1024 --data_max_val 3.834000 --rescale_input 1.000000 --rescale_output 1.000000 --indexDir ../../data/pig/Test/pl_rotate_both/test_indices_60.txt 


python ../../src/test_cluster.py --dataDir  ../../data/pig/Test/rotate_light_env6 --logDir ../../test_results/pig/rotate_light_env6/raw_output --checkpoint ../../models/pig --max_steps 360  --data_max_val 3.834000 --rescale_input 0.320000 --rescale_output 2.750000 --indexDir ../../data/pig/Test/rotate_light_env6/test_indices_60.txt 

python ../../src/test_mask.py --uvDir ../../data/pig/Test/rotate_light_env6/UV --logDir  ../../test_results/pig/rotate_light_env6/ --checkpoint ../../models_mask/pig --begin 0 --end 360 --imageDir ../../test_results/pig/rotate_light_env6/raw_output


python ../../src/test_cluster.py --dataDir ../../data/pig/Test/rotate_view_env6_dist2 --logDir ../../test_results/pig/rotate_view_env6_dist2/raw_output --checkpoint ../../models/pig --max_steps 360  --data_max_val 3.834000 --rescale_input 0.320000 --rescale_output 2.750000 --indexDir ../../data/pig/Test/rotate_view_env6_dist2/test_indices_60.txt  

python ../../src/test_mask.py --uvDir ../../data/pig/Test/rotate_view_env6_dist2/UV --logDir  ../../test_results/pig/rotate_view_env6_dist2 --checkpoint ../../models_mask/pig --begin 0 --end 360 --imageDir ../../test_results/pig/rotate_view_env6_dist2/raw_output




