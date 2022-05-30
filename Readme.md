# POVGAN
## nuscenes pre-processing
```
python ./script/preproccess_nuscenes.py --size {width/height of images} --mask {enable generating mask}
optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET, -d DATASET
                        Dataset folder of NuScenes
  --size SIZE, -s SIZE  Size of the image
  --output OUTPUT, -o OUTPUT
                        Output folder
  --mask, -m            Generate Masks
```

## Installation
```
pip install -r requirements.txt
```

## Training
```
python ./train.py --help
  -h, --help            show this help message and exit
  --name NAME           name of the experiment. It decides where to store samples and models
  --gpu_ids GPU_IDS     gpu ids: e.g. 0 0,1,2, 0,2. use -1 for CPU
  --checkpoints_dir CHECKPOINTS_DIR
                        models are saved here
  --model MODEL         which model to use
  --norm NORM           instance normalization or batch normalization
  --use_dropout         use dropout for the generator
  --data_type {8,16,32}
                        Supported data type i.e. 8, 16, 32 bit
  --verbose             toggles verbose
  --fp16                train with AMP
  --local_rank LOCAL_RANK
                        local rank for distributed training
  --batchSize BATCHSIZE
                        input batch size
  --loadSize LOADSIZE   scale images to this size
  --fineSize FINESIZE   then crop to this size
  --label_nc LABEL_NC   # of input label channels
  --input_nc INPUT_NC   # of input image channels
  --output_nc OUTPUT_NC
                        # of output image channels
  --dataroot DATAROOT
  --resize_or_crop RESIZE_OR_CROP
                        scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]
  --serial_batches      if true, takes images in order to make batches, otherwise takes them randomly
  --no_flip             if specified, do not flip the images for data argumentation
  --nThreads NTHREADS   # threads for loading data
  --max_dataset_size MAX_DATASET_SIZE
                        Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.
  --display_winsize DISPLAY_WINSIZE
                        display window size
  --tf_log              if specified, use tensorboard logging. Requires tensorflow installed
  --netG NETG           selects model to use for netG
  --ngf NGF             # of gen filters in first conv layer
  --n_downsample_global N_DOWNSAMPLE_GLOBAL
                        number of downsampling layers in netG
  --n_blocks_global N_BLOCKS_GLOBAL
                        number of residual blocks in the global generator network
  --n_blocks_local N_BLOCKS_LOCAL
                        number of residual blocks in the local enhancer network
  --n_local_enhancers N_LOCAL_ENHANCERS
                        number of local enhancers to use
  --niter_fix_global NITER_FIX_GLOBAL
                        number of epochs that we only train the outmost local enhancer
  --no_instance         if specified, do *not* add instance map as input
  --instance_feat       if specified, add encoded instance features as input
  --label_feat          if specified, add encoded label features as input
  --feat_num FEAT_NUM   vector length for encoded features
  --load_features       if specified, load precomputed feature maps
  --n_downsample_E N_DOWNSAMPLE_E
                        # of downsampling layers in encoder
  --nef NEF             # of encoder filters in the first conv layer
  --n_clusters N_CLUSTERS
                        number of clusters for features
  --mask
  --display_freq DISPLAY_FREQ
                        frequency of showing training results on screen
  --print_freq PRINT_FREQ
                        frequency of showing training results on console
  --save_latest_freq SAVE_LATEST_FREQ
                        frequency of saving the latest results
  --save_epoch_freq SAVE_EPOCH_FREQ
                        frequency of saving checkpoints at the end of epochs
  --no_html             do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/
  --debug               only do one epoch and displays at each iteration
  --continue_train      continue training: load the latest model
  --load_pretrain LOAD_PRETRAIN
                        load the pretrained model from the specified location
  --which_epoch WHICH_EPOCH
                        which epoch to load? set to latest to use latest cached model
  --phase PHASE         train, val, test, etc
  --niter NITER         # of iter at starting learning rate
  --niter_decay NITER_DECAY
                        # of iter to linearly decay learning rate to zero
  --beta1 BETA1         momentum term of adam
  --lr LR               initial learning rate for adam
  --num_D NUM_D         number of discriminators to use
  --n_layers_D N_LAYERS_D
                        only used if which_model_netD==n_layers
  --ndf NDF             # of discrim filters in first conv layer
  --lambda_feat LAMBDA_FEAT
                        weight for feature matching loss
  --no_ganFeat_loss     if specified, do *not* use discriminator feature matching loss
  --no_vgg_loss         if specified, do *not* use VGG feature matching loss
  --no_lsgan            do *not* use least square GAN, if false, use vanilla GAN
  --pool_size POOL_SIZE
                        the size of image buffer that stores previously generated images
```
## Benchmarking
```
python benchmark.py
optional arguments:
  -h, --help            show this help message and exit
  --name NAME           name of the experiment. It decides where to store samples and models
  --gpu_ids GPU_IDS     gpu ids: e.g. 0 0,1,2, 0,2. use -1 for CPU
  --checkpoints_dir CHECKPOINTS_DIR
                        models are saved here
  --model MODEL         which model to use
  --norm NORM           instance normalization or batch normalization
  --use_dropout         use dropout for the generator
  --data_type {8,16,32}
                        Supported data type i.e. 8, 16, 32 bit
  --verbose             toggles verbose
  --fp16                train with AMP
  --local_rank LOCAL_RANK
                        local rank for distributed training
  --batchSize BATCHSIZE
                        input batch size
  --loadSize LOADSIZE   scale images to this size
  --fineSize FINESIZE   then crop to this size
  --label_nc LABEL_NC   # of input label channels
  --input_nc INPUT_NC   # of input image channels
  --output_nc OUTPUT_NC
                        # of output image channels
  --dataroot DATAROOT
  --resize_or_crop RESIZE_OR_CROP
                        scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]
  --serial_batches      if true, takes images in order to make batches, otherwise takes them randomly
  --no_flip             if specified, do not flip the images for data argumentation
  --nThreads NTHREADS   # threads for loading data
  --max_dataset_size MAX_DATASET_SIZE
                        Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.
  --display_winsize DISPLAY_WINSIZE
                        display window size
  --tf_log              if specified, use tensorboard logging. Requires tensorflow installed
  --netG NETG           selects model to use for netG
  --ngf NGF             # of gen filters in first conv layer
  --n_downsample_global N_DOWNSAMPLE_GLOBAL
                        number of downsampling layers in netG
  --n_blocks_global N_BLOCKS_GLOBAL
                        number of residual blocks in the global generator network
  --n_blocks_local N_BLOCKS_LOCAL
                        number of residual blocks in the local enhancer network
  --n_local_enhancers N_LOCAL_ENHANCERS
                        number of local enhancers to use
  --niter_fix_global NITER_FIX_GLOBAL
                        number of epochs that we only train the outmost local enhancer
  --no_instance         if specified, do *not* add instance map as input
  --instance_feat       if specified, add encoded instance features as input
  --label_feat          if specified, add encoded label features as input
  --feat_num FEAT_NUM   vector length for encoded features
  --load_features       if specified, load precomputed feature maps
  --n_downsample_E N_DOWNSAMPLE_E
                        # of downsampling layers in encoder
  --nef NEF             # of encoder filters in the first conv layer
  --n_clusters N_CLUSTERS
                        number of clusters for features
  --mask
  --ntest NTEST         # of test examples.
  --results_dir RESULTS_DIR
                        saves results here.
  --aspect_ratio ASPECT_RATIO
                        aspect ratio of result images
  --phase PHASE         train, val, test, etc
  --which_epoch WHICH_EPOCH
                        which epoch to load? set to latest to use latest cached model
  --how_many HOW_MANY   how many test images to run
  --cluster_path CLUSTER_PATH
                        the path for clustered results of encoded features
  --use_encoded_image   if specified, encode the real image to get the feature map
  --export_onnx EXPORT_ONNX
                        export ONNX model to a given file
  --engine ENGINE       run serialized TRT engine
  --onnx ONNX           run ONNX model via TRT
```