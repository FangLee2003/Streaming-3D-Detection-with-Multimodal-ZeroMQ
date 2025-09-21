# custom/sender.py
"""
KITTI 3D Object Detection Real-time Sender
Real-time inference và streaming qua ZeroMQ đồng bộ hóa

Features:
- Real-time MMDetection3D inference
- ZeroMQ streaming với REQ-REP pattern (đồng bộ)
- Memory management và CUDA optimization
- Fallback support cho missing pkl files
- FPS control và statistics
- Linux optimized (Ubuntu/WSL2)
"""

import os
import sys
import argparse
import time
import json
import copy
import signal
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import cv2
import zmq
import numpy as np
import mmengine
from mmdet3d.apis import MultiModalityDet3DInferencer, init_model
from mmdet3d.utils import register_all_modules

# Global variables for graceful shutdown
shutdown_event = threading.Event()
stats = {
    'frames_sent': 0,
    'frames_skipped': 0,
    'start_time': time.time(),
    'last_fps_update': time.time(),
    'fps': 0.0,
    'avg_inference_time': 0.0,
    'avg_encode_time': 0.0,
    'connection_status': 'disconnected'
}


def signal_handler(signum, frame):
    """Handle graceful shutdown"""
    print(f"\nReceived signal {signum}, initiating shutdown...")
    shutdown_event.set()


def parse_args():
    parser = argparse.ArgumentParser(
        description='KITTI 3D Detection Real-time Sender')

    # Data arguments
    parser.add_argument('--data-root', required=True,
                        help='Path to KITTI data root')
    parser.add_argument('--config', required=True,
                        help='Path to MMDet3D config file')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--info-pkl', required=True,
                        help='KITTI infos pkl file (e.g., kitti_infos_train.pkl)')
    parser.add_argument('--split', choices=['training', 'testing'],
                        default='training', help='Dataset split')

    # Frame range
    parser.add_argument('--start', type=int, default=0,
                        help='Start frame index')
    parser.add_argument('--end', type=int, default=100,
                        help='End frame index')
    parser.add_argument('--loop', action='store_true',
                        help='Loop through frames continuously')

    # Inference settings
    parser.add_argument('--device', default='cuda:0',
                        help='Device for inference')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='Score threshold for detection')
    parser.add_argument('--force-files', action='store_true',
                        help='Force processing without pkl info')

    # Streaming settings
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='ZeroMQ host for streaming')
    parser.add_argument('--port', type=int, default=5555,
                        help='ZeroMQ port for streaming')

    parser.add_argument('--fps', type=float, default=10.0,
                        help='Target FPS for streaming')
    parser.add_argument('--timeout', type=int, default=5000,
                        help='ZeroMQ timeout in milliseconds')
    parser.add_argument('--jpeg-quality', type=int, default=85,
                        help='JPEG compression quality (1-100)')

    # Debug settings
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--stats-interval', type=int, default=10,
                        help='Statistics display interval (frames)')

    return parser.parse_args()


def load_kitti_infos_efficiently(info_pkl_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Load KITTI infos và tạo mapping hiệu quả"""
    if not os.path.exists(info_pkl_path):
        raise FileNotFoundError(f'Info pkl not found: {info_pkl_path}')

    infos_data = mmengine.load(info_pkl_path)

    if verbose:
        print(f"Loaded data type: {type(infos_data)}")

    # Extract data_list từ various formats
    if isinstance(infos_data, dict):
        if 'data_list' in infos_data:
            data_list = infos_data['data_list']
        elif 'infos' in infos_data:
            data_list = infos_data['infos']
        else:
            raise ValueError('Unknown format: no data_list or infos key found')
    elif isinstance(infos_data, list):
        data_list = infos_data
    else:
        raise ValueError('Loaded data is neither dict nor list')

    # Tạo mapping sample_idx -> data_info
    infos_map = {}
    for i, info in enumerate(data_list):
        if 'sample_idx' in info:
            if isinstance(info['sample_idx'], str):
                sample_idx = info['sample_idx'].zfill(6)
            else:
                sample_idx = f"{info['sample_idx']:06d}"
        else:
            sample_idx = f"{i:06d}"

        infos_map[sample_idx] = info

    if verbose:
        print(f"Total samples in pkl: {len(infos_map)}")
        available_indices = sorted([int(k) for k in infos_map.keys()])
        if available_indices:
            print(
                f"📋 Range: {available_indices[0]:06d} - {available_indices[-1]:06d}")

    return infos_map


def parse_kitti_calib_file(calib_path: str) -> Optional[Dict[str, np.ndarray]]:
    """Parse KITTI calibration file"""
    if not os.path.exists(calib_path):
        return None

    calib_data = {}
    try:
        with open(calib_path, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, values = line.split(':', 1)
                    values = [float(x) for x in values.strip().split()]
                    calib_data[key] = np.array(values)
    except Exception as e:
        print(f"Error parsing calibration file {calib_path}: {e}")
        return None

    # Extract matrices
    result = {}

    # cam2img (P2)
    if 'P2' in calib_data:
        result['cam2img'] = calib_data['P2'].reshape(3, 4)

    # lidar2cam (Tr_velo_to_cam) - convert 3x4 to 4x4
    if 'Tr_velo_to_cam' in calib_data:
        Tr_values = calib_data['Tr_velo_to_cam']
        if len(Tr_values) == 12:
            Tr_3x4 = Tr_values.reshape(3, 4)
            lidar2cam = np.eye(4)
            lidar2cam[:3, :] = Tr_3x4
            result['lidar2cam'] = lidar2cam

    # R0_rect - convert 3x3 to 4x4
    if 'R0_rect' in calib_data:
        R0_values = calib_data['R0_rect']
        if len(R0_values) == 9:
            R0_3x3 = R0_values.reshape(3, 3)
            R0_rect = np.eye(4)
            R0_rect[:3, :3] = R0_3x3
            result['R0_rect'] = R0_rect

    # Compute lidar2img
    if all(k in result for k in ['cam2img', 'lidar2cam', 'R0_rect']):
        cam2img_4x4 = np.eye(4)
        cam2img_4x4[:3, :] = result['cam2img']
        result['lidar2img'] = cam2img_4x4 @ result['R0_rect'] @ result['lidar2cam']

    return result if result else None


def create_fallback_data_info(data_root: str, split: str, idx: int, verbose: bool = False) -> Optional[Dict[str, Any]]:
    """Tạo data_info từ file system khi pkl không có đủ thông tin"""
    if verbose:
        print(f"Creating fallback data_info for {idx:06d}")

    # Check file existence
    lidar_path = os.path.join(data_root, split, 'velodyne', f'{idx:06d}.bin')
    img_path = os.path.join(data_root, split, 'image_2', f'{idx:06d}.png')
    calib_path = os.path.join(data_root, split, 'calib', f'{idx:06d}.txt')

    for path, name in [(lidar_path, 'LiDAR'), (img_path, 'Image'), (calib_path, 'Calibration')]:
        if not os.path.exists(path):
            if verbose:
                print(f"{name} file not found: {path}")
            return None

    # Parse calibration
    calib_info = parse_kitti_calib_file(calib_path)
    if not calib_info or 'cam2img' not in calib_info or 'lidar2cam' not in calib_info:
        if verbose:
            print(f"Failed to parse calibration: {calib_path}")
        return None

    # Create data_info structure
    data_info = {
        'sample_idx': idx,
        'lidar_points': {
            'lidar_path': lidar_path,
            'num_pts_feats': 4
        },
        'images': {
            'CAM2': {
                'img_path': img_path,
                'height': 375,
                'width': 1242,
                'cam2img': calib_info['cam2img'].tolist(),
                'lidar2cam': calib_info['lidar2cam'].tolist()
            }
        }
    }

    if 'lidar2img' in calib_info:
        data_info['images']['CAM2']['lidar2img'] = calib_info['lidar2img'].tolist()

    if verbose:
        print(f"Created fallback data_info for {idx:06d}")

    return data_info


def prepare_data_info(infos_map: Dict[str, Any], data_root: str, split: str,
                      idx: int, force_files: bool = False, verbose: bool = False) -> Optional[Dict[str, Any]]:
    """Chuẩn bị data_info cho frame cụ thể với fallback support"""
    target = f'{idx:06d}'

    # Try pkl first
    data_info = None
    if target in infos_map:
        data_info = copy.deepcopy(infos_map[target])
        if verbose:
            print(f'📋 Found frame {target} in pkl')
    elif force_files:
        if verbose:
            print(f'Frame {target} not in pkl, trying fallback...')
        data_info = create_fallback_data_info(data_root, split, idx, verbose)
        if data_info is None:
            return None
    else:
        if verbose:
            print(f'Frame {target} not found in pkl')
        return None

    # Ensure sample_idx is set correctly
    data_info['sample_idx'] = idx

    # Prepare LiDAR path
    if 'lidar_points' not in data_info:
        lidar_path = os.path.join(
            data_root, split, 'velodyne', f'{idx:06d}.bin')
        if os.path.exists(lidar_path):
            data_info['lidar_points'] = {
                'lidar_path': lidar_path,
                'num_pts_feats': 4
            }
        else:
            if verbose:
                print(f'LiDAR file not found: {lidar_path}')
            return None

    # Convert relative to absolute paths
    lidar_info = data_info['lidar_points']
    rel_lidar = lidar_info.get('lidar_path', f'{target}.bin')

    if not os.path.isabs(rel_lidar):
        abs_lidar = os.path.join(data_root, split, 'velodyne', rel_lidar)
    else:
        abs_lidar = rel_lidar

    if not os.path.exists(abs_lidar):
        if verbose:
            print(f'LiDAR file not found: {abs_lidar}')
        return None

    data_info['lidar_points']['lidar_path'] = abs_lidar

    # Prepare image info
    if 'images' not in data_info:
        data_info['images'] = {}

    if 'CAM2' not in data_info['images']:
        # Create fallback CAM2 info
        img_path = os.path.join(data_root, split, 'image_2', f'{idx:06d}.png')
        calib_path = os.path.join(data_root, split, 'calib', f'{idx:06d}.txt')

        if os.path.exists(img_path) and os.path.exists(calib_path):
            calib_info = parse_kitti_calib_file(calib_path)
            if calib_info and 'cam2img' in calib_info and 'lidar2cam' in calib_info:
                data_info['images']['CAM2'] = {
                    'img_path': img_path,
                    'height': 375,
                    'width': 1242,
                    'cam2img': calib_info['cam2img'].tolist(),
                    'lidar2cam': calib_info['lidar2cam'].tolist()
                }
                if 'lidar2img' in calib_info:
                    data_info['images']['CAM2']['lidar2img'] = calib_info['lidar2img'].tolist(
                    )
            else:
                return None
        else:
            return None

    # Ensure calibration matrices are present
    cam2_info = data_info['images']['CAM2']
    if 'cam2img' not in cam2_info or 'lidar2cam' not in cam2_info:
        calib_path = os.path.join(data_root, split, 'calib', f'{idx:06d}.txt')
        if os.path.exists(calib_path):
            calib_info = parse_kitti_calib_file(calib_path)
            if calib_info and 'cam2img' in calib_info and 'lidar2cam' in calib_info:
                cam2_info['cam2img'] = calib_info['cam2img'].tolist()
                cam2_info['lidar2cam'] = calib_info['lidar2cam'].tolist()
                if 'lidar2img' in calib_info:
                    cam2_info['lidar2img'] = calib_info['lidar2img'].tolist()
            else:
                return None
        else:
            return None

    # Convert image path to absolute
    rel_img = cam2_info.get('img_path', f'{target}.png')
    if not os.path.isabs(rel_img):
        abs_img = os.path.join(data_root, split, 'image_2', rel_img)
    else:
        abs_img = rel_img

    if not os.path.exists(abs_img):
        if verbose:
            print(f'Image file not found: {abs_img}')
        return None

    data_info['images']['CAM2']['img_path'] = abs_img

    return data_info


def create_temp_pkl(data_info: Dict[str, Any], temp_dir: str, idx: int) -> str:
    """Tạo temporary pkl file cho single frame"""
    temp_pkl = os.path.join(temp_dir, f'{idx:06d}.pkl')

    pkl_data = {
        'data_list': [data_info],
        'metainfo': {
            'dataset_name': 'KittiDataset',
            'version': '1.0'
        }
    }

    mmengine.dump(pkl_data, temp_pkl)
    return temp_pkl


def encode_image(image: np.ndarray, quality: int = 85) -> bytes:
    """Encode image to JPEG bytes"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_img = cv2.imencode('.jpg', image, encode_param)
    return encoded_img.tobytes()


def update_statistics(frame_idx: int, inference_time: float, encode_time: float):
    """Update streaming statistics"""
    global stats

    stats['frames_sent'] += 1

    # Update running averages
    alpha = 0.1  # Smoothing factor
    stats['avg_inference_time'] = (
        1 - alpha) * stats['avg_inference_time'] + alpha * inference_time
    stats['avg_encode_time'] = (1 - alpha) * \
        stats['avg_encode_time'] + alpha * encode_time

    # Update FPS
    current_time = time.time()
    if current_time - stats['last_fps_update'] >= 1.0:  # Update every second
        elapsed = current_time - stats['start_time']
        stats['fps'] = stats['frames_sent'] / elapsed if elapsed > 0 else 0.0
        stats['last_fps_update'] = current_time


def print_statistics(frame_idx: int, verbose: bool = False):
    """Print current statistics"""
    if verbose or stats['frames_sent'] % 10 == 0:
        elapsed = time.time() - stats['start_time']
        print(f"Frame {frame_idx:06d} | "
              f"Sent: {stats['frames_sent']} | "
              f"Skipped: {stats['frames_skipped']} | "
              f"FPS: {stats['fps']:.1f} | "
              f"Inference: {stats['avg_inference_time']*1000:.1f}ms | "
              f"Encode: {stats['avg_encode_time']*1000:.1f}ms | "
              f"Status: {stats['connection_status']}")


class KITTIStreamer:
    def __init__(self, args):
        self.args = args
        self.infos_map = {}
        self.inferencer = None
        self.socket = None
        self.context = None
        self.temp_dir = None
        self.frame_interval = 1.0 / args.fps if args.fps > 0 else 0.0

    def setup_zmq(self):
        """Setup ZeroMQ connection"""
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind(f"tcp://{self.args.host}:{self.args.port}")
            self.socket.setsockopt(zmq.RCVTIMEO, self.args.timeout)
            self.socket.setsockopt(zmq.SNDTIMEO, self.args.timeout)

            print(f"ZeroMQ server listening on port {self.args.port}")
            print(f"Timeout: {self.args.timeout}ms")
            stats['connection_status'] = 'listening'
            return True
        except Exception as e:
            print(f"Failed to setup ZeroMQ: {e}")
            return False

    def cleanup_zmq(self):
        """Clean up ZeroMQ resources"""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        print("🧹 ZeroMQ cleaned up")

    def load_data(self):
        """Load KITTI dataset info"""
        info_pkl_path = os.path.join(self.args.data_root, self.args.info_pkl)
        try:
            self.infos_map = load_kitti_infos_efficiently(
                info_pkl_path, self.args.verbose)
            print(
                f"Loaded {len(self.infos_map)} frame infos from {self.args.info_pkl}")
            return True
        except Exception as e:
            print(f"Error loading pkl: {e}")
            if self.args.force_files:
                print("Continuing with force_files mode...")
                self.infos_map = {}
                return True
            else:
                print("Use --force-files to continue without pkl info")
                return False

    def setup_inferencer(self):
        """Initialize MMDetection3D inferencer"""
        try:
            register_all_modules()

            print("Initializing MMDetection3D inferencer...")
            self.inferencer = MultiModalityDet3DInferencer(
                model=self.args.config,
                weights=self.args.checkpoint,
                device=self.args.device
            )

            # Setup temp directory
            self.temp_dir = os.path.join('/tmp', f'mmdet3d_{os.getpid()}')
            os.makedirs(self.temp_dir, exist_ok=True)

            print("Inferencer ready")
            return True
        except Exception as e:
            print(f"Failed to initialize inferencer: {e}")
            return False

    def cleanup_temp_files(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                print("Temporary files cleaned up")
            except Exception as e:
                print(f"Error cleaning temp files: {e}")

    def clear_cuda_cache(self):
        """Clear CUDA cache to prevent memory issues"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def wait_for_client(self):
        """Wait for client connection (REQ-REP pattern)"""
        try:
            # Wait for request from client
            message = self.socket.recv_json()

            if message.get('type') == 'request_frame':
                stats['connection_status'] = 'connected'
                return True
            else:
                # Send error response
                self.socket.send_json({
                    'type': 'error',
                    'message': 'Invalid request type'
                })
                return False

        except zmq.Again:
            # Timeout
            stats['connection_status'] = 'timeout'
            return False
        except Exception as e:
            print(f"Error waiting for client: {e}")
            stats['connection_status'] = 'error'
            return False

    def send_frame(self, image_data: bytes, frame_info: Dict[str, Any]):
        """Send frame data to client"""
        try:
            # Prepare message
            message = {
                'type': 'frame_data',
                'frame_idx': frame_info['frame_idx'],
                'timestamp': time.time(),
                'image_size': len(image_data),
                'image_data': image_data.hex(),  # Convert bytes to hex string
                'stats': {
                    'fps': stats['fps'],
                    'frames_sent': stats['frames_sent'],
                    'inference_time': frame_info.get('inference_time', 0),
                    'encode_time': frame_info.get('encode_time', 0)
                }
            }

            # Send response
            self.socket.send_json(message)
            return True

        except Exception as e:
            print(f"Error sending frame: {e}")
            stats['connection_status'] = 'error'

            self.socket.send_json({
                'type': 'error',
                'message': str(e)
            })
            return False

    def send_end_signal(self):
        """Send end of stream signal"""
        try:
            # Wait for final request
            self.socket.recv_json()

            # Send end signal
            message = {
                'type': 'end_stream',
                'total_frames': stats['frames_sent'],
                'total_skipped': stats['frames_skipped'],
                'avg_fps': stats['fps']
            }
            self.socket.send_json(message)
            print("📡 Sent end of stream signal")
            return True
        except Exception as e:
            print(f"Error sending end signal: {e}")

            self.socket.send_json({
                'type': 'error',
                'message': str(e)
            })
            return False

    def process_frame(self, idx: int) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """Process single frame and return encoded image with metadata"""
        start_time = time.time()

        # Prepare data info
        data_info = prepare_data_info(
            self.infos_map, self.args.data_root, self.args.split, idx,
            force_files=self.args.force_files, verbose=self.args.verbose
        )

        if data_info is None:
            if self.args.verbose:
                print(f"Skipped frame {idx:06d}: data not available")
            stats['frames_skipped'] += 1
            return None

        # Create temporary pkl
        temp_pkl = create_temp_pkl(data_info, self.temp_dir, idx)

        try:
            # Clear CUDA cache
            self.clear_cuda_cache()

            # Prepare inputs for inferencer
            lidar_path = data_info['lidar_points']['lidar_path']
            img_path = data_info['images']['CAM2']['img_path']

            inputs = [{
                'points': lidar_path,
                'img': img_path,
                'infos': temp_pkl
            }]
            print(">>> inputs =", inputs)
            # Run inference (in-memory, no file output)
            inference_start = time.time()

            # Since MMDet3D might not support return_vis directly,
            # we use a workaround by running inference to a temp dir
            # and immediately read the result
            temp_out_dir = os.path.join(self.temp_dir, f'vis_{idx:06d}')
            os.makedirs(temp_out_dir, exist_ok=True)

            # Run inference with temporary output
            inference_start = time.time()

            self.inferencer(
                inputs=inputs,
                out_dir=temp_out_dir,
                cam_type='CAM2',
                pred_score_thr=self.args.score_thr,
            )

            inference_time = time.time() - inference_start

            # Look for visualization result
            vis_path = os.path.join(
                temp_out_dir, 'vis_camera', 'CAM2', f'{idx:06d}.png')

            # Fallback: find any png file in the output directory
            if not os.path.exists(vis_path):
                vis_dir = os.path.join(temp_out_dir, 'vis_camera', 'CAM2')
                if os.path.exists(vis_dir):
                    png_files = [f for f in os.listdir(
                        vis_dir) if f.endswith('.png')]
                    if png_files:
                        vis_path = os.path.join(vis_dir, png_files[0])

            if not os.path.exists(vis_path):
                # Fallback: use original image with text overlay
                img = cv2.imread(img_path)
                if img is not None:
                    cv2.putText(img, f'Frame {idx:06d} - No Detection',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, f'Score Threshold: {self.args.score_thr}',
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    if self.args.verbose:
                        print(f"Cannot read original image: {img_path}")
                    return None
            else:
                # Read visualization result
                img = cv2.imread(vis_path)
                if img is None:
                    if self.args.verbose:
                        print(f"Cannot read visualization: {vis_path}")
                    return None

            # Encode image
            encode_start = time.time()
            image_data = encode_image(img, self.args.jpeg_quality)
            encode_time = time.time() - encode_start

            # Cleanup temp visualization
            if os.path.exists(temp_out_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_out_dir)
                except:
                    pass

            # Update statistics
            total_time = time.time() - start_time
            update_statistics(idx, inference_time, encode_time)

            frame_info = {
                'frame_idx': idx,
                'inference_time': inference_time,
                'encode_time': encode_time,
                'total_time': total_time,
                'image_size': len(image_data)
            }

            return image_data, frame_info

        except Exception as e:
            print(f"Error processing frame {idx:06d}: {e}")
            if self.args.verbose:
                import traceback
                traceback.print_exc()
            stats['frames_skipped'] += 1

            return None

        finally:
            # Cleanup temp pkl file
            if os.path.exists(temp_pkl):
                try:
                    os.remove(temp_pkl)
                except:
                    pass

    def run_streaming(self):
        """Main streaming loop with REQ-REP pattern"""
        print(f"🎬 Starting streaming loop")
        print(f"Frame range: {self.args.start} - {self.args.end}")
        print(f"Target FPS: {self.args.fps}")
        print(f"Score threshold: {self.args.score_thr}")
        print(f"JPEG quality: {self.args.jpeg_quality}")

        frame_indices = list(range(self.args.start, self.args.end + 1))
        current_idx = 0

        try:
            while not shutdown_event.is_set():
                loop_start_time = time.time()

                # Current frame
                frame_idx = frame_indices[current_idx]

                # Wait for client request
                if not self.wait_for_client():
                    if shutdown_event.is_set():
                        break
                    continue

                # Process frame
                result = self.process_frame(frame_idx)

                if result is not None:
                    image_data, frame_info = result

                    # Send frame to client
                    if self.send_frame(image_data, frame_info):
                        print_statistics(frame_idx, self.args.verbose)
                    else:
                        # Connection error, continue waiting
                        stats['connection_status'] = 'disconnected'
                        continue
                else:
                    # Frame processing failed, send error
                    try:
                        error_message = {
                            'type': 'error',
                            'frame_idx': frame_idx,
                            'message': f'Failed to process frame {frame_idx:06d}',
                            'timestamp': time.time()
                        }
                        self.socket.send_json(error_message)
                    except Exception as e:
                        self.socket.send_json({
                            'type': 'error',
                            'message': str(e)
                        })
                    continue

                # Move to next frame
                current_idx = (current_idx + 1) % len(frame_indices)

                # Loop restart for continuous streaming
                if current_idx == 0 and self.args.loop:
                    print("Looping back to start...")
                elif current_idx == 0 and not self.args.loop:
                    print("Reached end of sequence")
                    break

                # FPS control
                if self.frame_interval > 0:
                    elapsed = time.time() - loop_start_time
                    sleep_time = self.frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStreaming interrupted by user")
        except Exception as e:
            print(f"Streaming error: {e}")
            if self.args.verbose:
                import traceback
                traceback.print_exc()

        finally:
            # Send end signal
            self.send_end_signal()

            # Final statistics
            total_time = time.time() - stats['start_time']
            print(f"\nFinal Statistics:")
            print(f"   Total frames sent: {stats['frames_sent']}")
            print(f"   Total frames skipped: {stats['frames_skipped']}")
            print(f"   Total time: {total_time:.1f}s")
            print(f"   Average FPS: {stats['frames_sent']/total_time:.1f}")
            print(
                f"   Average inference time: {stats['avg_inference_time']*1000:.1f}ms")
            print(
                f"   Average encode time: {stats['avg_encode_time']*1000:.1f}ms")

    def run(self):
        """Main execution flow"""
        print("KITTI 3D Object Detection Real-time Sender")
        print("=" * 60)

        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Initialize components
            if not self.load_data():
                return False

            if not self.setup_inferencer():
                return False

            if not self.setup_zmq():
                return False

            print("All components initialized successfully")
            print(
                f"Waiting for receiver connection on port {self.args.port}...")

            # Start streaming
            self.run_streaming()

            return True

        except Exception as e:
            print(f"Fatal error: {e}")
            if self.args.verbose:
                import traceback
                traceback.print_exc()
            return False

        finally:
            # Cleanup
            self.cleanup_zmq()
            self.cleanup_temp_files()
            print("Sender shutdown complete")


def main():
    """Main entry point"""
    args = parse_args()

    # Validate arguments
    if not os.path.exists(args.data_root):
        print(f"Data root not found: {args.data_root}")
        return 1

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return 1

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint file not found: {args.checkpoint}")
        return 1

    # Info pkl validation
    info_pkl_path = os.path.join(args.data_root, args.info_pkl)
    if not os.path.exists(info_pkl_path) and not args.force_files:
        print(f"Info pkl not found: {info_pkl_path}")
        print("Use --force-files to run without pkl info")
        return 1

    # Frame range validation
    if args.start < 0 or args.end < args.start:
        print(f"Invalid frame range: {args.start} to {args.end}")
        return 1

    if args.fps <= 0:
        print(f"Invalid FPS: {args.fps}")
        return 1

    # Print configuration
    print(f"Data root: {args.data_root}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Info pkl: {args.info_pkl}")
    print(f"Frame range: {args.start} - {args.end}")
    print(f"Loop: {args.loop}")
    print(f"Device: {args.device}")
    print(f"Score threshold: {args.score_thr}")
    print(f"Port: {args.port}")
    print(f"FPS: {args.fps}")
    print(f"Timeout: {args.timeout}ms")
    print(f"JPEG quality: {args.jpeg_quality}")
    print(f"Force files: {args.force_files}")
    print(f"Verbose: {args.verbose}")

    # Create and run streamer
    streamer = KITTIStreamer(args)
    success = streamer.run()

    return 0 if success else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
