import ast
import configparser

import tensorflow as tf

from .front import HRFront
from .stage import HRStage
from .head import ClsHead

slim = tf.contrib.slim


w32_config = {
  'NET': {'num_stages': 3, 'num_channels': 32},
  'FRONT': {'num_channels': 64, 'bottlenect_channels': 256,
            'output_channels': 32, 'num_blocks': 4},
  'S2': {'num_modules': 1, 'num_blocks': 4, 'num_branches': 2},
  'S3': {'num_modules': 4, 'num_blocks': 4, 'num_branches': 3},
  'S4': {'num_modules': 3, 'num_blocks': 4, 'num_branches': 4},
  'HEAD': {'base_channel': 128, 'num_branches': 4, 'fc_channel': 2048}
}


w48_config = {
  'NET': {'num_stages': 3, 'num_channels': 48},
  'FRONT': {'num_channels': 64, 'bottlenect_channels': 256,
            'output_channels': 48, 'num_blocks': 4},
  'S2': {'num_modules': 1, 'num_blocks': 4, 'num_branches': 2},
  'S3': {'num_modules': 4, 'num_blocks': 4, 'num_branches': 3},
  'S4': {'num_modules': 3, 'num_blocks': 4, 'num_branches': 4},
  'HEAD': {'base_channel': 128, 'num_branches': 4, 'fc_channel': 2048}
}


def hrnet(cfg):
  stages = []
  front = HRFront(num_channels=cfg['FRONT']['num_channels'],
                  bottlenect_channels=cfg['FRONT']['bottlenect_channels'],
                  output_channels=[i * cfg['FRONT']['output_channels'] for i in range(1, 3)],
                  num_blocks=cfg['FRONT']['num_blocks'])
  stages.append(front)

  num_stages = cfg['NET']['num_stages']
  for i in range(num_stages):
    _key = 'S{}'.format(i + 2)
    _stage = HRStage(stage_id=i + 2,
                    num_modules=cfg[_key]['num_modules'],
                    num_channels=cfg['NET']['num_channels'],
                    num_blocks=cfg[_key]['num_blocks'],
                    num_branches=cfg[_key]['num_branches'],
                    last_stage=True if i == num_stages - 1 else False)
    stages.append(_stage)

  clshead = ClsHead(base_channel=cfg['HEAD']['base_channel'],
                    num_branches=cfg['HEAD']['num_branches'],
                    cls_num=num_classes,
                    fc_channel=cfg['HEAD']['fc_channel'])
  stages.append(clshead)
  return stages


def build_hrnet_w32(input, is_training=True, num_classes=None):
  cfg = w32_config
  stages = hrnet(cfg)

  with tf.variable_scope('HRNet') as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        out = input
        for stage in stages:
          out = stage.forward(out)
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return out, end_points


def build_hrnet_w48(input, is_training=True, num_classes=None):
  cfg = w48_config
  stages = hrnet(cfg)

  with tf.variable_scope('HRNet') as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        out = input
        for stage in stages:
          out = stage.forward(out)
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return out, end_points


def hrnet_arg_scope(weight_decay=0.0001,
                    batch_norm_decay=0.997,
                    batch_norm_epsilon=1e-5,
                    batch_norm_scale=True,
                    activation_fn=tf.nn.relu,
                    use_batch_norm=True,
                    batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):

  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': batch_norm_updates_collections,
      'fused': None,  # Use fused batch norm if possible.
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=activation_fn,
      normalizer_fn=slim.batch_norm if use_batch_norm else None,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
      return arg_sc
