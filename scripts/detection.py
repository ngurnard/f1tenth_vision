import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
import common
from convert_trt import get_engine
from utils import voting_suppression, bbox_convert_r, label_to_box_xyxy, DisplayLabel


ONNX_FILE_PATH = "../models/hot_wheels_model.onnx"
ENGINE_FILE_PATH = "../models/hot_wheels_model.trt"

def preprocess(image):
 
    image = image / 255.0
    input_img = cv2.resize(image, (320, 180), interpolation=cv2.INTER_NEAREST)
    input_img = np.transpose(input_img, (2, 0, 1))
    input_img = np.expand_dims(input_img, 0)

    return input_img


def postprocess(result):
    voting_iou_threshold = 0.5
    confi_threshold = 0.4
    bboxs, result_prob = label_to_box_xyxy(result, confi_threshold)
    vote_rank = voting_suppression(bboxs, voting_iou_threshold)
    bbox = bboxs[vote_rank[0]]
    [c_x, c_y, w, h] = bbox_convert_r(bbox[0], bbox[1], bbox[2], bbox[3])
    bboxs_2 = np.array([[c_x, c_y, w, h]])


    return bboxs_2


def detect_bbox(image):
    
    with get_engine(ONNX_FILE_PATH , ENGINE_FILE_PATH) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)

        # Do inference
        # print("Running inference on image {}...".format(input_image_path))

        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = np.array(preprocess(image), dtype=np.float32, order='C')   # np.float16 for FP.16
        start = time.time()
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        end = time.time()
        dt = end - start
        print("Time:", dt)

    # post processing
    result = np.reshape(trt_outputs, (5, 5, 10))
    bbox = postprocess(result)
    print("bbox shape:", bbox.shape)

    DisplayLabel(image, bbox)

    return bbox

        


    # engine, context = build_engine(ONNX_FILE_PATH)
    # for binding in engine:
    #     if engine.binding_is_input(binding):  # we expect only one input
    #         input_shape = engine.get_binding_shape(binding)
    #         input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
    #         device_input = cuda.mem_alloc(input_size)
    #     else:  # and one output
    #         output_shape = engine.get_binding_shape(binding)
    #         # create page-locked memory buffers (i.e. won't be swapped to disk)
    #         host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
    #         device_output = cuda.mem_alloc(host_output.nbytes)

    # # Create a stream in which to copy inputs/outputs and run inference.
    # stream = cuda.Stream()

    # # preprocess input data
    # host_input = np.array(preprocess(img_path).numpy(), dtype=np.float32, order='C')
    # cuda.memcpy_htod_async(device_input, host_input, stream)

    # # run inference
    # context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    # cuda.memcpy_dtoh_async(host_output, device_output, stream)
    # stream.synchronize()

    # # postprocess results
    # output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])
    # bbox = postprocess(output_data)  # what to reshape?

    # return bbox


