import multiprocessing as mp
import subprocess
import argparse
import numpy as np

BLENDER_PATH_2_79 = '/local/vondrick/skm2167/blender-2.79-linux-glibc219-x86_64/blender' 
BLENDER_PATH_2_81 = '/local/vondrick/skm2167/blender-2.81a-linux-glibc217-x86_64/blender'
BLENDER_PATH = BLENDER_PATH_2_81
OUT_PATH = '/local/vondrick/skm2167/CATER_OUT'
OUTPUT_SCENE_FILE_PATH = '/local/vondrick/skm2167/CATER_OUT/CLEVR_scenes.json'
CAM_MOTION = False
MAX_MOTIONS = 2

def parse_args():
    parser = argparse.ArgumentParser(description='Launch blender')
    parser.add_argument(
        '--gpus', '-g', default=None, type=str,
        help='GPUs to run on')
    parser.add_argument(
        '--num_jobs', '-n', default=1, type=int,
        help='Run n jobs per GPU')
    return parser.parse_args()


def get_gpu_count():
    count = int(subprocess.check_output(
        'ls /proc/driver/nvidia/gpus/ | wc -l', shell=True))
    return count


def run_blender(gpu_id):
    # sleep for a random time, to make sure it does not overlap!
    # Thanks to Ishan Misra for providing the singularity file
    sleep_time = 1 + int(np.random.random() * 5)  # upto 6 seconds
    subprocess.call('sleep {}'.format(sleep_time), shell=True)

    cmd = '''
        CUDA_VISIBLE_DEVICES="{gpu_id}" \
            {blender_path} \
            data/base_scene.blend \
            --background --python render_videos.py -- \
            --num_images 50000 \
            --suppress_blender_logs \
            --save_blendfiles 1 \
            {cam_motion} \
            {max_motions} \
            --output_dir {output_dir} \
            --output_scene_file {output_scene_file} \
            --fps 24 \
            --render_num_samples 32 \
            --start_idx 5500
    '''

    final_cmd = cmd.format(
        gpu_id=gpu_id,
        output_dir=OUT_PATH,
        output_scene_file=OUTPUT_SCENE_FILE_PATH,
        cam_motion='--random_camera' if CAM_MOTION else '',
        max_motions='--max_motions={}'.format(MAX_MOTIONS),
        blender_path=BLENDER_PATH)

    print('Running {}'.format(final_cmd))
    subprocess.call(final_cmd, shell=True)


args = parse_args()
if args.gpus is None:
    ngpus = get_gpu_count()
    gpu_ids = range(ngpus)
else:
    gpu_ids = [int(el) for el in args.gpus.split(',')]
ngpus = len(gpu_ids)
print('Found {} GPUs. Using all of those.'.format(ngpus))
# Repeat jobs per GPU
gpu_ids *= args.num_jobs
pool = mp.Pool(len(gpu_ids))
pool.map(run_blender, gpu_ids)
