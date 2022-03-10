import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
import numpy as np

from external.deep_head_pose.code import hopenet


class rotation_dealer:

    def __init__(self, gpu=0):
        cudnn.enabled = True
        self.gpu = gpu
        self.snapshot_path = 'external/deep_head_pose/model/hopenet_robust_alpha1.pkl'

        # ResNet50 structure
        self.model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

        # Load snapshot
        self.saved_state_dict = torch.load(self.snapshot_path)
        self.model.load_state_dict(self.saved_state_dict)

        self.transformations = transforms.Compose([transforms.Resize(224),
                                                   transforms.CenterCrop(224), transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])
        self.model.cuda(gpu)

        # Test the Model
        self.model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

        idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(idx_tensor).cuda(self.gpu)

    def rotation_dealer(self, img, bbox_line_list, landmks):

        height, width, channels = img.shape

        cv2_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for i, (bbox, landmk) in enumerate(zip(bbox_line_list, landmks)):
            x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            x_min -= 50
            x_max += 50
            y_min -= 50
            y_max += 30
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(width, x_max)
            y_max = min(height, y_max)
            # Crop image
            img = cv2_frame[y_min:y_max, x_min:x_max]
            img = Image.fromarray(img)
            # img.show()

            # Transform
            img = self.transformations(img)
            img_shape = img.size()
            img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
            img = Variable(img).cuda(self.gpu)

            yaw, pitch, roll = self.model(img)

            yaw_predicted = F.softmax(yaw)
            pitch_predicted = F.softmax(pitch)
            roll_predicted = F.softmax(roll)
            # Get continuous predictions in degrees.
            yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99

            print(' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
            if abs(yaw_predicted) > 40.0:
                bbox_line_list = np.delete(bbox_line_list, i, 0)
                landmks = np.delete(landmks, i, 0)
        return bbox_line_list, landmks
