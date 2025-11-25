import os

import onnx
import torch


def calibrate(output_dir, model_name, onnx_model_path):
    """
    Calibrate onnx model and generate golden
    """
    from hmquant.api import generate_golden
    from hmquant.api import quant_single_onnx_network

    onnx_model = onnx.load(onnx_model_path)
    dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    input_shape = [dim.dim_value for dim in dims]
    torch.manual_seed(100)
    calib_dataset = [
        torch.randint(
            low=-128, high=127, size=input_shape, dtype=torch.float32,
        ),
    ]

    quanttool_config = {
        'inputs_cfg': {
            'ALL': {
                'data_format': 'RGB',
                'first_layer_weight_denorm_mean': [0.485, 0.456, 0.406],
                'first_layer_weight_denorm_std': [0.229, 0.224, 0.225],
                'resizer_crop': {'top': 0, 'left': 0, 'height': input_shape[2], 'width': input_shape[3]},
                'resizer_resize': {
                    'height': input_shape[2],
                    'width': input_shape[3],
                    'align_corners': False,
                    'method': 'bilinear',
                },
                'toYUV_format': 'YUV420',
            },
        },
        'graph_opt_cfg': {},
    }
    sequencer = quant_single_onnx_network(
        quanttool_config,
        calib_dataset,
        onnx_model_path,
        device='cpu',
    )
    # Save golden
    golden_input, golden_inter, golden_onnx = generate_golden(
        sequencer,
        calibset=calib_dataset[0],
        save_path=output_dir,
        model_name=model_name,
    )


def compile_aot(model_name, model_path):
    """
    Compile quantized model and generate hm model
    """
    import tcim

    onnx_model = onnx.load(model_path)
    input_name = onnx_model.graph.input[0].name
    dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    input_shape = [dim.dim_value for dim in dims]
    tcim.build_from_hmonnx(
        model_path,
        output_name=model_name,
        output_dir='./',
        legacy=True,
    )
    return input_name, input_shape, onnx_model.graph.output[0].name


def inference_aot(data_dir: str, model_name: str, input_name: str, input_shape, output_name: str):
    """
    Load hm model and inference with quantool golden, compare the result
    """
    import tcim_lite
    import numpy as np

    model = tcim_lite.runtime.Module.load(model_name + '.hmm')
#    stream = tcim_lite.runtime.Stream()
#    model.set_stream(stream)

    input_data_file_name = os.path.join(
        data_dir, 'hmquant_' + model_name + '_' + input_name + '_input.npy',
    )
    input_data = np.load(
        input_data_file_name,
        allow_pickle=True,
    ).astype('uint8')

    for input_idx in range(model.get_num_inputs()):
        mod_input_name = model.get_input_name(input_idx)
        mod_input_info = model.get_input_info(mod_input_name)
        print(f'{mod_input_name}: shape: {mod_input_info.shape}, dtype: {mod_input_info.dtype}, format: {mod_input_info.format.name}')
    # NCHW -> NHWC
    input_info = model.get_input_info(input_name).ascontiguous()
    print(f'{input_name}: shape: {input_info.shape}, dtype: {input_info.dtype}, format: {input_info.format.name}')
    print(
        f'input name: {input_name} vs {model.get_input_name(0)}, shape: {input_data.shape} vs {input_shape}',
    )
    print(f'      data: {input_data.flatten()[:10]} vs {input_shape}')
    input_data = np.reshape(input_data, input_shape)
    input_tensor = tcim_lite.runtime.Tensor(input_info, input_data)
    model.set_input(input_name, input_tensor)
    model.run()
    model.sync()
    res = True
    for output_idx in range(model.get_num_outputs()):
        output_name = model.get_output_name(output_idx)
        eval_output = model.get_output(output_name).numpy()
        print('shape::::', eval_output.shape)
        golden_output_data_file_name = os.path.join(
            data_dir, 'hmquant_' + model_name + '_with_act', output_name + '.npy',
        )
        golden_output = (
            np.load(golden_output_data_file_name, allow_pickle=True)
            .item()
            .get('output_tensor')
        )

        import sys

        sys.path.append('..')
        from utils import cos_compare
        from utils import bit_compare

        # result1 = cos_compare(eval_output, golden_output)
        # if result1[0] == False:
        #     print(result1[1])
        #     print("cos golden output", golden_output)
        #     print("cos eval output", eval_output)
        #     print("----------------failed0------------------")
        result2 = bit_compare(eval_output, golden_output)
        if result2[0] == False:
            print(result2[1])
            print('bit golden output', golden_output)
            print('bit eval output', eval_output)
            print('----------------failed------------------')
        if result2[0] == True:
            print('----------------Pass------------------')
        else:
            res = False
    return res


if __name__ == '__main__':
    #    fp_model = "/nfsdata/models/model_zoo/yaluete/yolov7s_face_640x640.onnx"
    import sys
    fp_model_path = './avgpool_1.onnx'
    quanted_dir = 'output'
    model_name = 'avgpool_1'
    print(f'Test {model_name} {fp_model_path}')
    if not os.path.exists(os.path.join(quanted_dir, f'hmquant_{model_name}_with_act.onnx')):
        calibrate(quanted_dir, model_name, fp_model_path)
    input_name, input_shape, output_name = compile_aot(
        model_name, os.path.join(
            quanted_dir, f'hmquant_{model_name}_with_act.onnx',
        ),
    )
    assert inference_aot(
        quanted_dir, model_name,
        input_name, input_shape, output_name,
    )
