python ../../src/test_cluster.py --dataDir ../../data/tree/Test/pl_rotate --logDir ../../test_results/tree/pl_rotate --checkpoint ../../models/tree  --max_steps 360  --data_max_val 2.000000 --rescale_input 1.000000 --rescale_output 2.750000 --indexDir ../../data/tree/Test/pl_rotate/test_indices_60.txt 

python ../../src/test_cluster.py --dataDir ../../data/tree/Test/pl_rotate_both --logDir ../../test_results/tree/pl_rotate_both --checkpoint ../../models/tree  --max_steps 1024  --data_max_val 2.000000 --rescale_input 1.000000 --rescale_output 2.750000 --indexDir ../../data/tree/Test/pl_rotate_both/test_indices_60.txt 

python ../../src/test_cluster.py --dataDir ../../data/tree/Test/rotate_light_env5/ --logDir ../../test_results/tree/rotate_light_env5/raw_output --checkpoint ../../models/tree --max_steps 360  --data_max_val 2.000000 --rescale_input 0.320000 --rescale_output 2.750000 --indexDir ../../data/tree/Test/rotate_light_env5/test_indices_60.txt  

python ../../src/test_mask.py --uvDir ../../data/tree/Test/rotate_light_env5/UV --logDir  ../../test_results/tree/rotate_light_env5 --checkpoint ../../models_mask/tree --begin 0 --end 360 --imageDir ../../test_results/tree/rotate_light_env5/raw_output

python ../../src/test_cluster.py --dataDir ../../data/tree/Test/rotate_view_env5_dist2 --logDir ../../test_results/tree/rotate_view_env5_dist2/raw_output --checkpoint ../../models/tree --max_steps 360  --data_max_val 2.000000 --rescale_input 0.320000 --rescale_output 2.750000 --indexDir ../../data/tree/Test/rotate_view_env5_dist2/test_indices_60.txt 

python../../src/test_mask.py --uvDir ../../data/tree/Test/rotate_view_env5_dist2/UV --logDir  ../../test_results/tree/rotate_view_env5_dist2 --checkpoint ../../models_mask/tree --begin 0 --end 360 --imageDir ../../test_results/tree/rotate_view_env5_dist2/raw_output
