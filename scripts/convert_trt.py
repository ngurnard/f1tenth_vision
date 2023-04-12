import numpy as np
import tensorrt as trt
import common
import os


# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 28  # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
                
            network.get_input(0).shape = [1, 3, 180, 320]
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


# def build_engine(onnx_file_path):
#     # initialize TensorRT engine and parse ONNX model
#     builder = trt.Builder(TRT_LOGGER)
#     # network = builder.create_network()
#     explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
#     network = builder.create_network(explicit_batch)
#     parser = trt.OnnxParser(network, TRT_LOGGER)

#     # allow TensorRT to use up to 1GB of GPU memory for tactic selection
#     builder.max_workspace_size = 1 << 30
#     # we have only one image in batch
#     builder.max_batch_size = 1
#     # use FP16 mode if possible
#     if builder.platform_has_fast_fp16:
#         builder.fp16_mode = True

#     # parse ONNX
#     with open(onnx_file_path, 'rb') as model:
#         print('Beginning ONNX file parsing')
#         parser.parse(model.read())
#     print('Completed parsing of ONNX file')

#     # generate TensorRT engine optimized for the target platform
#     print('Building an engine...')
#     engine = builder.build_cuda_engine(network)
#     context = engine.create_execution_context()
#     print("Completed creating Engine")

#     return engine, context


# if __name__ == '__main__':
#     # initialize TensorRT engine and parse ONNX model
#     engine, context = build_engine(ONNX_FILE_PATH)
#     print(engine)
#     print(context)
