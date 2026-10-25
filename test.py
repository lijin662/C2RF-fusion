import torch
import os
from model import C2RF
from dataset import TestData, Generate_Reg_Data, imsave
from tqdm import tqdm
from options import TrainOptions
from scipy import io
from utils.utils import checkboard
from utils.device import resolve_device
import torchvision.transforms.functional as TF

if __name__ == '__main__':
    parser = TrainOptions()
    opts = parser.parse()
    device = resolve_device(opts.device, opts.gpu)
    print('Using device: %s' % device)

    # =================================================================================================
    # Generate new test data
    # =================================================================================================
    if opts.generate_test_warp:
        print('============ Generating warped test data ===============')
        gene_dataset = Generate_Reg_Data(opts)
        gene_dataloader = torch.utils.data.DataLoader(
            gene_dataset, batch_size=1, shuffle=False, num_workers=opts.nThreads)
        for _ in tqdm(gene_dataloader, total=len(gene_dataloader)):
            pass

    # 1.modal
    model = C2RF(training_components=False).to(device)
    model_path = os.path.join(opts.checkpoint_dir, opts.dataset)
    required = {
        'Encoder.pth': model.AE_encoder,
        'Decoder.pth': model.AE_decoder,
        'Fusion_layer.pth': model.fusion_layer,
        'RegNet.pth': model.reg_net,
    }
    missing = [os.path.join(model_path, name) for name in required
               if not os.path.isfile(os.path.join(model_path, name))]
    if missing:
        raise FileNotFoundError('Missing checkpoint files: %s' % ', '.join(missing))
    for filename, module in required.items():
        module.load_state_dict(torch.load(os.path.join(model_path, filename), map_location=device))
    model.eval()

    # data
    test_dataloader = TestData(opts)
    p_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

    # test
    print('============ Testing Begins ===============')
    save_dir = os.path.join(opts.output_dir, 'test', opts.dataset)
    for save_name in ['reg_cb', 'reg_vi', 'reg_fu', 'reg_flow']:
        os.makedirs(os.path.join(save_dir, save_name), exist_ok=True)
    for idx, [ir, vi, vi_warp, name] in p_bar:
        vi, ir, vi_warp = vi.to(device), ir.to(device), vi_warp.to(device)
        with torch.no_grad():
            b = vi_warp.shape[0]
            vi_reg, fusion_reg, disp, flow = model.forward(vi_warp, ir, vi)

            # save checkboard
            checkboard_image = checkboard(ir, vi_reg)
            checkboard_save_name = os.path.join(save_dir, 'reg_cb',name)
            TF.to_pil_image(checkboard_image.squeeze(0)).save(checkboard_save_name)

            # save reg_vi
            reg_vi_save_name = os.path.join(save_dir, 'reg_vi',name)
            imsave(vi_reg, reg_vi_save_name)

            # save reg_fu
            reg_fu_save_name = os.path.join(save_dir, 'reg_fu',name)
            imsave(fusion_reg, reg_fu_save_name)

            # save reg_flow
            reg_flow_save_name = os.path.join(save_dir, 'reg_flow', os.path.splitext(name)[0] + '.mat')
            io.savemat(reg_flow_save_name, {'flow': flow.cpu().numpy()})
