import os
from collections import OrderedDict
from torch.autograd import Variable
from options.benchmark_options import BenchmarkOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import torchvision
from util.mask_rcnn import instance_segmentation
# from PIL import Image
# import torchvision.transforms as transforms

opt = BenchmarkOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
counter_classes = ['person', 'car', 'truck', 'bus']
counter_synthetic = {t:0 for t in counter_classes}
counter_real = {t:0 for t in counter_classes}
counter_result = {t:0 for t in counter_classes}
object_detector = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda().eval()
# transform = transforms.Compose([
#         transforms.ToTensor()
#         ])

visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx
    
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst']  = data['inst'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst']  = data['inst'].uint8()
    if opt.export_onnx:
        print ("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        torch.onnx.export(model, [data['label'], data['inst']],
                          opt.export_onnx, verbose=True)
        exit(0)
    minibatch = 1 
    if opt.engine:
        generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
    elif opt.onnx:
        generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
    else:        
        generated = model.inference(data['label'], data['inst'], data['image'])
    image_path = data['image_path'][0]        
    # img = Image.open(image_path).convert('RGB')
    # img_tensor = transform(img)
    # img_tensor = img_tensor[None, ...].cuda()
    synthetic_image = util.tensor2im(generated.data[0])
    detected_image, counter_r = instance_segmentation(image_path, object_detector)
    detected_synthetic_image, counter_s = instance_segmentation(synthetic_image, object_detector)
    for key in counter_classes:
        counter_real[key] += counter_r[key]
        counter_synthetic[key] += counter_s[key]
        counter_result[key] += (counter_s[key] / counter_r[key]) / opt.how_many if counter_r[key] > 0 else 1 / opt.how_many
    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0][0][None, ...], opt.label_nc)),
                           ('synthesized_image', synthetic_image),
                           ('real_image', util.tensor2im(data['image'][0])),
                           ('mask_rcnn_real', detected_image),
                           ('mask_rcnn_synthetic', detected_synthetic_image)
                           ])
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
print(counter_real)
print(counter_synthetic)
print(counter_result)
