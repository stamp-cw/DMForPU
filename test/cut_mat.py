import os

import scipy.io as sio


def save_mat(path, d_name, d_type, mat):
    """Saves a matrix to a text file.

    Args:
        path (str): The file path where the matrix will be saved.
        mat (list of list of float): The matrix to be saved.
    """

    if d_name == 'InSARDLPUMat':
        if d_type == 'wrapped':
            # data = sio.loadmat(path)['input']
            sio.savemat(path, {'input': mat})
        else:
            # data = sio.loadmat(path)['output']
            sio.savemat(path, {'output': mat})

    elif d_name =='SyntheticPUMat':
        if d_type == 'wrapped':
            # data = sio.loadmat(path)['input']
            sio.savemat(path, {'input': mat})
        else:
            # data = sio.loadmat(path)['output']
            sio.savemat(path, {'gt': mat})

def load_mat(path, d_name,d_type):
    """Loads a matrix from a text file.

    Args:
        path (str): The file path from which the matrix will be loaded.

    Returns:
        list of list of float: The loaded matrix.
    """

    if d_name == 'InSARDLPUMat':
        if d_type == 'wrapped':
            data = sio.loadmat(path)['input']
        else:
            data = sio.loadmat(path)['output']

    elif d_name =='SyntheticPUMat':
        if d_type == 'wrapped':
            data = sio.loadmat(path)['input']
        else:
            data = sio.loadmat(path)['gt']
    else:
        assert "Error"
    return data

def process_mat(mat_path, parent_dir, d_name, d_type, raw_zie, cut_size,output_dir):
    mat = load_mat(mat_path,d_name, d_type)
    n = raw_zie // cut_size
    for i in range(0, n):
        for j in range(0, n):
            i_s_idx = i * cut_size
            i_d_idx = (i+1) * cut_size
            j_s_idx = j * cut_size
            j_d_idx = (j+1) * cut_size
            c_mat = mat[i_s_idx:i_d_idx, j_s_idx:j_d_idx]
            save_dir = os.path.join(f"cut_data_{cut_size}",parent_dir)
            save_dir = os.path.join(output_dir, save_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            base_name = os.path.basename(mat_path).split('.')[0]
            save_path = os.path.join(save_dir, f"{base_name}_part_{i}_{j}.mat")
            save_mat(save_path, d_name, d_type, c_mat)
            print("Save mat to ", save_path)

def cut_mat(root_dir, d_name, d_type, raw_zie, cut_size):
    records = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".mat"):
                mat_path = os.path.join(root, file)
                parent_dir = os.path.basename(root)

                records.append({
                    "mat_path": mat_path,
                    "parent_dir": parent_dir
                })
                process_mat(mat_path, parent_dir, d_name, d_type, raw_zie, cut_size, root_dir)


if __name__ == '__main__':
    # cut_mat("/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/tmp/SyntheticPUMat/train_in", d_name='SyntheticPUMat', d_type='wrapped', raw_zie=128, cut_size=64)
    # cut_mat("/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/tmp/SyntheticPUMat/train_gt", d_name='SyntheticPUMat', d_type='unwrapped', raw_zie=128, cut_size=64)

    # cut_mat("/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/InSARDLPUMat/train_wrapped", d_name='InSARDLPUMat', d_type='wrapped', raw_zie=256, cut_size=64)
    # cut_mat("/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/InSARDLPUMat/test_wrapped", d_name='InSARDLPUMat', d_type='wrapped', raw_zie=256, cut_size=64)
    # cut_mat("/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/InSARDLPUMat/train_absolute", d_name='InSARDLPUMat', d_type='unwrapped', raw_zie=256, cut_size=64)
    # cut_mat("/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/InSARDLPUMat/test_absolute", d_name='InSARDLPUMat', d_type='unwrapped', raw_zie=256, cut_size=64)
    # cut_mat("/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat/train_in", d_name='SyntheticPUMat', d_type='wrapped', raw_zie=128, cut_size=64)
    # cut_mat("/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat/test_in", d_name='SyntheticPUMat', d_type='wrapped', raw_zie=128, cut_size=64)
    # cut_mat("/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat/train_gt", d_name='SyntheticPUMat', d_type='gt', raw_zie=128, cut_size=64)
    # cut_mat("/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat/test_gt", d_name='SyntheticPUMat', d_type='gt', raw_zie=128, cut_size=64)
    # cut_mat("/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat/train_in", d_name='SyntheticPUMat', d_type='wrapped', raw_zie=128, cut_size=32)
    # cut_mat("/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat/test_in", d_name='SyntheticPUMat', d_type='wrapped', raw_zie=128, cut_size=32)
    # cut_mat("/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat/train_gt", d_name='SyntheticPUMat', d_type='gt', raw_zie=128, cut_size=32)
    # cut_mat("/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat/test_gt", d_name='SyntheticPUMat', d_type='gt', raw_zie=128, cut_size=32)

    cut_mat("/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128/train_in", d_name='SyntheticPUMat', d_type='wrapped', raw_zie=128, cut_size=64)
    cut_mat("/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128/test_in", d_name='SyntheticPUMat', d_type='wrapped', raw_zie=128, cut_size=64)
    cut_mat("/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128/train_gt", d_name='SyntheticPUMat', d_type='gt', raw_zie=128, cut_size=64)
    cut_mat("/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128/test_gt", d_name='SyntheticPUMat', d_type='gt', raw_zie=128, cut_size=64)