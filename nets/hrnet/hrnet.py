import ast
import configparser

import tensorflow as tf

from .front import HRFront
from .stage import HRStage
from .head import ClsHead

slim = tf.contrib.slim


def load_net_cfg_from_file(cfgfile):
  def load_from_options(section, cfg):
    options = dict()
    xdict = dict(cfg.items(section))
    for key, value in xdict.items():
      try:
        value = ast.literal_eval(value)
      except:
        value = value
      options[key] = value
    return options

  cfg = configparser.ConfigParser()
  cfg.read(cfgfile)

  sections = cfg.sections()
  options = dict()
  for _section in sections:
    options[_section] = load_from_options(_section, cfg)

  return options


def build_hrnet(input, config_file, is_training=True, num_classes=None):
  cfg = load_net_cfg_from_file(config_file)

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
