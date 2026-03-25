import torch
from PIL import Image
import cv2

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    sam_model_registry,
    SamPredictor
)


def keep_or_wall(s):
    parts = s.split(" ", 1)
    if len(parts) == 2 and parts[1] == "wall":
        return "wall"
    return parts[0]


class GroundingDINO:

    def __init__(self, device):
        self.config_file = "./third_party/groundedsam/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        
        self.box_threshold = 0.25
        self.text_threshold = 0.2
        self.device = device


    def load_image(self, image_path):
        self.image_pil = Image.open(image_path).convert("RGB")  

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.image, _ = transform(self.image_pil, None)  # 3, h, w


    def load_model(self, ckpt_grounding):
        # GroundingDINO
        args = SLConfig.fromfile(self.config_file)
        args.device = self.device
        args.bert_base_uncased_path = None
        self.model = build_model(args)
        checkpoint = torch.load(ckpt_grounding, map_location="cpu")
        load_res = self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = self.model.eval()


    def get_detection_output(self, caption, with_logits=True):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = self.model.to(self.device)
        image = self.image.to(self.device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  
        boxes = outputs["pred_boxes"].cpu()[0]  
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  
        boxes_filt = boxes_filt[filt_mask]  
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(keep_or_wall(pred_phrase))

        return boxes_filt, pred_phrases


class SAM:
    def __init__(self, device):
        self.device = device
        self.version = "vit_h"


    def load_model(self, ckpt_sam):
        self.predictor = SamPredictor(sam_model_registry[self.version](checkpoint=ckpt_sam).to(self.device))


    def load_image(self, file_path):
        self.image = cv2.imread(file_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(self.image)


    def get_segmentation_mask(self, boxes_filt):
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, self.image.shape[:2]).to(self.device)

        masks, _, _ = self.predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )

        return masks
