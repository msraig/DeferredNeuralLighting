python ../../src/test_cluster.py --dataDir ../../data/sphere/Test/pl_rotate --logDir ../../test_results/sphere/v2/pl_rotate --checkpoint ../../models/spheres/final2 --max_steps 360  --data_max_val 10.000000 --rescale_input 1.000000 --rescale_output 1.000000 --indexDir ../../data/sphere/Test/pl_rotate/test_indices_60.txt 

python ../../src/test_cluster.py --dataDir ../../data/sphere/Test/pl_rotate_both --logDir ../../test_results/sphere/v2/pl_rotate_both --checkpoint ../../models/spheres/final2  --max_steps 1024  --data_max_val 10.000000 --rescale_input 1.000000 --rescale_output 1.000000 --indexDir ../../data/sphere/Test/pl_rotate_both/test_indices_60.txt

python ../../src/test_cluster.py --dataDir ../../data/sphere/Test/rotate_light_env7 --logDir ../../test_results/sphere/v2/rotate_light_env7/raw_output --checkpoint ../../models/spheres/final2 --max_steps 360  --data_max_val 10.000000 --rescale_input 0.320000 --rescale_output 2.750000 --indexDir ../../data/sphere/Test/rotate_light_env7/test_indices_nearby_60_fixed.txt 

python ../../src/test_mask.py --uvDir ../../data/sphere/Test/rotate_light_env7/UV --logDir  ../../test_results/sphere/v2/rotate_light_env7 --checkpoint ../../models_mask/sphere --begin 0 --end 360  --imageDir ../../test_results/sphere/v2/rotate_light_env7/raw_output

python ../../src/test_cluster.py --dataDir ../../data/sphere/Test/rotate_view_env7_dist2/ --logDir ../../test_results/sphere/v2/rotate_view_env7_dist2/raw_output --checkpoint  ../../models/spheres/final2 --max_steps 360  --data_max_val 10.000000 --rescale_input 0.320000 --rescale_output 2.750000 --indexDir ../../data/sphere/Test/rotate_view_env7_dist2/test_indices_nearby_60.txt 

python ../../src/test_mask.py --uvDir ../../data/sphere/Test/rotate_view_env7_dist2/UV --logDir  ../../test_results/sphere/v2/rotate_view_env7_dist2 --checkpoint ../../models_mask/sphere --begin 0 --end 360 --imageDir ../../test_results/sphere/v2/rotate_view_env7_dist2/raw_output