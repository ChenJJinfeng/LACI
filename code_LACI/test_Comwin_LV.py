import os
import argparse
import torch
from utils.test_3d_patch import test_all_case
from networks.Comwin_net import TriDSBAVNet_after8
# from test_util import test_all_case_dsba as test_all_case
def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--dataset', type=str,  default="LV", help='dataset')
parser.add_argument('--exp', type=str,  default="Comwin", help='model_name')
parser.add_argument('--testlist', type=str,  default="test.txt", help='model_name')
parser.add_argument('--labeled_num', type=int,  default=16, help='trained samples')
parser.add_argument('--num_classes', type=int,  default='3', help='num_classes')
parser.add_argument('--total_num', type=int,  default=369, help='maximum samples to train')
parser.add_argument('--patch_size', type=list, default=[112,112,80],help='patch size')
parser.add_argument('--stride_xy', type=int, default=18, help='stride_xy')
parser.add_argument('--stride_z', type=int, default=4, help='stride_z')
# parser.add_argument('--root_path', type=str, default='../data/Pancreas-CT-all/', help='Name of Experiment')
# parser.add_argument('--image_list_path', type=str, default='pancreas_test.list', help='image_list_path')
# parser.add_argument('--dataset_name', type=str, default='pancreas', help='dataset_name')
parser.add_argument('--model', type=str,  default='pancreas_v2_000', help='model_name')
parser.add_argument('--gpu', type=str,  default='3', help='GPU to use')
parser.add_argument('--iter', type=int,  default=3000, help='model iteration')      ## 6k

parser.add_argument('--ds_starting_layer', type=int,  default=8, help='ds_starting_layer')
parser.add_argument('--head_type', type=int,  default=1, help='head_type')
parser.add_argument('--window_size', type=int,  default=2, help='window_size')
parser.add_argument('--self_atten_head_num', type=int,  default=1, help='self_atten_head_num')
parser.add_argument('--sparse_attn', type=str2bool,  default=False, help='sparse_attn')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "/data/chenjinfeng/code/semi_supervised/All_code/code_all/weights/{}/{}/{}".format(FLAGS.dataset, FLAGS.exp, FLAGS.labeled_num)
test_save_path = os.path.join(snapshot_path, 'predict/')
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = FLAGS.num_classes
if FLAGS.dataset == "LA":
    with open('/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods/LA/'+FLAGS.testlist, 'r') as f:  # todo change test flod
        image_list = f.readlines()
    image_list = [item.replace('\n', '') + "/mri_norm_new.h5" for item in image_list]
elif FLAGS.dataset == "LV":
    with open('/data/chenjinfeng/code/semi_supervised/All_code/code_all/Flods/LV_112/test.txt',
              'r') as f:  # todo change test flod
        image_list = f.readlines()
        print(image_list)
    image_list = [item.replace('\n', '') for item in image_list]

patch_size = FLAGS.patch_size

def test_calculate_metric(epoch_num):
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    checkpoint = torch.load(save_mode_path)
    if FLAGS.ds_starting_layer == 8:
        net = TriDSBAVNet_after8(input_channels=1, num_classes=num_classes, head_type = FLAGS.head_type, window_size = FLAGS.window_size, self_atten_head_num = FLAGS.self_atten_head_num, sparse_attn = FLAGS.sparse_attn, has_dropout=True).cuda()
    else:
        raise NotImplementedError
    net.load_state_dict(checkpoint['model'])

    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=FLAGS.patch_size, stride_xy=FLAGS.stride_xy, stride_z=FLAGS.stride_z,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric(FLAGS.iter)
    print(metric)

