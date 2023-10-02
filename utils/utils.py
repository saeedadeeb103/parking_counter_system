import torch
import torchvision.transforms as transforms
from skimage.transform import resize
from train import SimpleCNN
from testing import ResNet18

class Utils:
    def __init__(self, model_path='model.pth', conf_threshold=0.9):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.conf_threshold = conf_threshold  # Confidence threshold for detection

    def load_model(self, model_path):
        model = ResNet18(num_classes=2)  # Instantiate your ResNet-18 model here
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()  # Set the model to evaluation mode
        return model

    def empty_or_not(self, spot_bgr):
        with torch.no_grad():
            img_resize = resize(spot_bgr, (15, 15, 3))
            img_tensor = self.transform(img_resize).float().unsqueeze(0).to(self.device)
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.softmax(outputs, dim=1)[0][predicted[0]].item()
            if confidence > self.conf_threshold:
                if predicted.item() == 0:
                    return "Empty"
                else:
                    return "Not Empty"
            else:
                return "Uncertain"
        
    def get_parking_spots_bboxes(self, connected_components):
        (totalLabels, label_ids, values, centroids) = connected_components

        slots = []
        coef = 1 
        for i in range(1, totalLabels):
            x1, y1, x2, y2 = values[i]
            # Now, you can use x1, y1, x2, and y2 to define the bounding box of the parking spot
            # You may want to store this information in the 'slots' list
            # For example: slots.append((x1, y1, x2, y2))
            slots.append((x1, y1, x2, y2))
        
        return slots


class Utils2:
    def __init__(self, EMPTY=True, NOT_EMPTY=False, model_path='model_simple.pth'):
        self.EMPTY = EMPTY
        self.NOT_EMPTY = NOT_EMPTY
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_model(self, model_path):
        model = SimpleCNN()  # Define your model architecture here
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()  # Set the model to evaluation mode
        return model

    def empty_or_not(self, spot_bgr):
        with torch.no_grad():
            img_resize = resize(spot_bgr, (15, 15, 3))
            img_tensor = self.transform(img_resize).float().unsqueeze(0).to(self.device)
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)

            if predicted.item() == 0:
                return "Empty"
            else:
                return "Not Empty"
        
    def get_parking_spots_bboxes(self, connected_components):
        (totalLabels, label_ids, values, centroids) = connected_components

        slots = []
        coef = 1 
        for i in range(1, totalLabels):
            x1, y1, x2, y2 = values[i]
            # Now, you can use x1, y1, x2, and y2 to define the bounding box of the parking spot
            # You may want to store this information in the 'slots' list
            # For example: slots.append((x1, y1, x2, y2))
        
        return slots
