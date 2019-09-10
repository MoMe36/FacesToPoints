import time
import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf
import argparse
import torch.backends.cudnn as cudnn
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
from PIL import Image 
from sklearn.preprocessing import MinMaxScaler 
import os 
import shutil
import glob 
from tqdm import tqdm 

def eulerAnglesToRotationMatrix(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
         
         
                     
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R

def plot_face(points, img_fp): 

    x, y, z = np.zeros((3,3))
    u, v, w = np.array([[1,0,0],[0,1,0],[0,0,1]])
    # ur,vr,wr= np.dot(rot, np.array([[1,0,0],[0,1,0],[0,0,1]]))

    elev = 90
    azim = 90
    # f_name = str(int(time.time()))[-5:]

    fig = plt.figure()
    ax = fig.add_subplot(1,2,1, projection='3d')
        
    ax.scatter(points[:,0], points[:,1],points[:,2], s = 1)

    for key in parts.keys():

        face_part = np.array(parts[key])

        if len(face_part.shape)>1:  # handles mouth and lips
            selected_points = np.concatenate([points[fp[0]:fp[1]] for fp in face_part], axis = 0)
            selected_points = np.vstack([selected_points, selected_points[0,:]])
        elif 'eye' in key: 
            selected_points = points[face_part[0]:face_part[1]]
            selected_points = np.vstack([selected_points, selected_points[0,:]]) 
        
        else: 
            selected_points = points[face_part[0]:face_part[1]]

        ax.plot(selected_points[:,0], 
                selected_points[:,1], 
                selected_points[:,2])



    ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, colors = ['r','g','b'])
    ax.view_init(elev = elev, azim = azim) 


    ax = fig.add_subplot(1,2,2)
    ax.imshow(Image.open(img_fp))



STD_SIZE = 120


parts = {'jaw': [0,17], 
         'right_b': [17,22], 
         'left_b': [22,27], 
         'nose_1': [27,31], 
         'nose_2': [31,36], 
         'right_eye': [36,42],  
         'left_eye': [42,48],  
         'lips': [[48,55], [55,60]],  
         'mouth': [[60,65],[65,67]]}




def main(args):
    # 1. load pre-tained model
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    # 2. load dlib model for face detection and landmark used for face cropping
    if args.dlib_landmark:
        dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
        face_regressor = dlib.shape_predictor(dlib_landmark_model)
    if args.dlib_bbox:
        face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    tri = sio.loadmat('visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    files = sorted(glob.glob(os.path.join(args.folder, '*.jpg')) + glob.glob(os.path.join(args.folder, '*.png')))
    p_bar = tqdm(total = len(files))
    for img_fp in files:

        img_ori = cv2.imread(img_fp)
        if args.dlib_bbox:
            rects = face_detector(img_ori, 1)
        else:
            rects = []

        if len(rects) == 0:
            rects = dlib.rectangles()
            rect_fp = img_fp + '.bbox'
            lines = open(rect_fp).read().strip().split('\n')[1:]
            for l in lines:
                l, r, t, b = [int(_) for _ in l.split(' ')[1:]]
                rect = dlib.rectangle(l, r, t, b)
                rects.append(rect)

        pts_res = []
        Ps = []  # Camera matrix collection
        poses = []  # pose collection, [todo: validate it]
        vertices_lst = []  # store multiple face vertices
        ind = 0
        suffix = get_suffix(img_fp)
        for rect in rects:
            # whether use dlib landmark to crop image, if not, use only face bbox to calc roi bbox for cropping
            if args.dlib_landmark:
                # - use landmark for cropping
                pts = face_regressor(img_ori, rect).parts()
                pts = np.array([[pt.x, pt.y] for pt in pts]).T
                roi_box = parse_roi_box_from_landmark(pts)
            else:
                # - use detected face bbox
                bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                roi_box = parse_roi_box_from_bbox(bbox)

            img = crop_img(img_ori, roi_box)

            # forward: one step
            img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input_ = transform(img).unsqueeze(0)
            with torch.no_grad():
                if args.mode == 'gpu':
                    input_ = input_.cuda()
                param = model(input_)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            # 68 pts
            pts68 = predict_68pts(param, roi_box)

            # two-step for more accurate bbox to crop face
            if args.bbox_init == 'two':
                roi_box = parse_roi_box_from_landmark(pts68)
                img_step2 = crop_img(img_ori, roi_box)
                img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input_ = transform(img_step2).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'gpu':
                        input_ = input_.cuda()
                    param = model(input_)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                pts68 = predict_68pts(param, roi_box)

            pts_res.append(pts68)
            P, pose = parse_pose(param)
            Ps.append(P)
            poses.append(pose)

            points= np.array(pts_res)[0].T

            rotated = eulerAnglesToRotationMatrix(np.array([0., np.pi,0.]))
            points = points.dot(rotated)

            scaler = MinMaxScaler(feature_range = (-1.,1))
            scaled_points = scaler.fit_transform(points)
            points = scaled_points
            

            f_name = img_fp.replace(args.folder + '/', '').replace('.png','').replace('.jpg', '')
            np.save('./results/{}.npy'.format(f_name), points.reshape(-1))

            if args.plot: 
                plot_face(points, img_fp)
                if args.show_flg: 
                    plt.show()
                else:
                    plt.savefig('./results/{}.png'.format(f_name))

            p_bar.update(1)



if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-f', '--folder',help='image files paths fed into network, single or multiple images', default = './imgs')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flg', default='false', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--bbox_init', default='one', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_res', default='true', type=str2bool, help='whether write out the visualization image')
    parser.add_argument('--dump_vertex', default='false', type=str2bool,
                        help='whether write out the dense face vertices to mat')
    parser.add_argument('--dump_ply', default='true', type=str2bool)
    parser.add_argument('--dump_pts', default='true', type=str2bool)
    parser.add_argument('--dump_roi_box', default='false', type=str2bool)
    parser.add_argument('--dump_pose', default='true', type=str2bool)
    parser.add_argument('--dump_depth', default='true', type=str2bool)
    parser.add_argument('--dump_pncc', default='true', type=str2bool)
    parser.add_argument('--dump_paf', default='false', type=str2bool)
    parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
    parser.add_argument('--dump_obj', default='true', type=str2bool)
    parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')
    parser.add_argument('--dlib_landmark', default='true', type=str2bool,
                        help='whether use dlib landmark to crop image')

    parser.add_argument('--plot', default = 'false', type = str2bool, help = "Wheter to plot to image or not")
    args = parser.parse_args()
    try: 
        os.makedirs('./results')
    except: 
        pass 

    main(args)
