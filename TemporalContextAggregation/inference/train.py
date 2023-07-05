import logging
import os
import zipfile
import sys
import json
import time
from tqdm import tqdm
import torch
import numpy as np

import sklearn
import sklearn.metrics
from sklearn.metrics import average_precision_score
from SoccerNet.Evaluation.ActionSpotting import evaluate
from SoccerNet.Evaluation.utils import AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1


def predict(dataloader, model, model_name, overwrite=True, NMS_window=30, NMS_threshold=0.5):
    
    # split = '_'.join(dataloader.dataset.split)
    # # print(split)
    # output_results = os.path.join("models", model_name, f"results_spotting_{split}.zip")
    output_folder = f"outputs_inference"


    batch_time = AverageMeter()
    data_time = AverageMeter()

    spotting_predictions = list()

    model.eval()

    # count_visible = torch.FloatTensor([0.0]*dataloader.dataset.num_classes)
    # count_unshown = torch.FloatTensor([0.0]*dataloader.dataset.num_classes)
    # count_all = torch.FloatTensor([0.0]*dataloader.dataset.num_classes)

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, (feat_half1) in t:
            data_time.update(time.time() - end)

            # Batch size of 1
            feat_half1 = feat_half1.squeeze(0)

            # Compute the output for batches of frames
            BS = 256
            timestamp_long_half_1 = []
            for b in range(int(np.ceil(len(feat_half1)/BS))):
                start_frame = BS*b
                end_frame = BS*(b+1) if BS * \
                    (b+1) < len(feat_half1) else len(feat_half1)
                feat = feat_half1[start_frame:end_frame].to(torch.device("cpu"))
                output = model(feat).cpu().detach().numpy()
                timestamp_long_half_1.append(output)
            timestamp_long_half_1 = np.concatenate(timestamp_long_half_1)


            timestamp_long_half_1 = timestamp_long_half_1[:, 1:]

            spotting_predictions.append(timestamp_long_half_1)

            batch_time.update(time.time() - end)
            end = time.time()

            desc = f'Test (spot.): '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            t.set_description(desc)



            def get_spot_from_NMS(Input, window=60, thresh=0.0):
                detections_tmp = np.copy(Input)
                indexes = []
                MaxValues = []
                while(np.max(detections_tmp) >= thresh):

                    # Get the max remaining index and value
                    max_value = np.max(detections_tmp)
                    max_index = np.argmax(detections_tmp)
                    MaxValues.append(max_value)
                    indexes.append(max_index)
                    # detections_NMS[max_index,i] = max_value

                    nms_from = int(np.maximum(-(window/2)+max_index,0))
                    nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)))
                    detections_tmp[nms_from:nms_to] = -1

                return np.transpose([indexes, MaxValues])

            framerate = dataloader.dataset.framerate
            get_spot = get_spot_from_NMS

            json_data = dict()
            json_data["predictions"] = list()

            for half, timestamp in enumerate([timestamp_long_half_1]):
                for l in range(dataloader.dataset.num_classes):
                    spots = get_spot(
                        timestamp[:, l], window=NMS_window*framerate, thresh=NMS_threshold)
                    print("\npanjang spots:", len(spots))
                    for spot in spots:
                        # print("spot", int(spot[0]), spot[1], spot)
                        frame_index = int(spot[0])
                        confidence = spot[1]
                        # confidence = predictions_half_1[frame_index, l]

                        seconds = int((frame_index//framerate)%60)
                        minutes = int((frame_index//framerate)//60)

                        prediction_data = dict()
                        prediction_data["gameTime"] = str(half+1) + " - " + str(minutes) + ":" + str(seconds)
                        prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[l]
                        prediction_data["position"] = str(int((frame_index/framerate)*1000))
                        prediction_data["half"] = str(half+1)
                        prediction_data["confidence"] = str(confidence)
                        json_data["predictions"].append(prediction_data)
            # print("\n", json_data)
            os.makedirs(os.path.join("inference/models", model_name, output_folder))
            with open(os.path.join("inference/models", model_name, output_folder, "results_spotting.json"), 'w') as output_file:
                json.dump(json_data, output_file, indent=4)
            return 1
