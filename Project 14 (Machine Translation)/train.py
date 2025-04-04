import os, argparse, ast
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import Input
from tensorflow.keras.optimizers import AdamW, Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from model import Transformer
from metric_loss import MaskedCELoss, MaskedAccuracy
from utils import CustomScheduler, CreatePadMask

create_data_mask = CreatePadMask(mask_token=0)

def preprocessing_data(train_data_path: dict[str,str], val_data_path: dict[str,str]):

    train_ind_input = tf.convert_to_tensor(np.load(train_data_path['indo_input']), dtype=tf.int32)
    train_eng_input = tf.convert_to_tensor(np.load(train_data_path['eng_input']), dtype=tf.int32)
    train_eng_target = tf.convert_to_tensor(np.load(train_data_path['eng_target']), dtype=tf.float32)

    val_ind_input = tf.convert_to_tensor(np.load(val_data_path['indo_input']), dtype=tf.int32)
    val_eng_input = tf.convert_to_tensor(np.load(val_data_path['eng_input']), dtype=tf.int32)
    val_eng_target = tf.convert_to_tensor(np.load(val_data_path['eng_target']), dtype=tf.float32)

    train_ind_mask = create_data_mask(train_ind_input)
    train_eng_mask = create_data_mask(train_eng_input)

    val_ind_mask = create_data_mask(val_ind_input)
    val_eng_mask = create_data_mask(val_eng_input)

    train_dataset = tf.data.Dataset.from_tensor_slices(((train_ind_input, train_eng_input, train_ind_mask, train_eng_mask), train_eng_target))
    val_dataset = tf.data.Dataset.from_tensor_slices(((val_ind_input, val_eng_input, val_ind_mask, val_eng_mask), val_eng_target))
    
    return train_dataset, val_dataset

def create_model(**model_parameter):
  CONTEXT_NUM_VOCABS = model_parameter.get("context_num_vocabs")
  CONTEXT_MAX_LEN = model_parameter.get("context_max_len")
  TARGET_NUM_VOCABS = model_parameter.get("target_num_vocabs")
  TARGET_MAX_LEN = model_parameter.get("target_max_len")
  NUM_HEADS = model_parameter.get("num_heads")
  EMBED_DIM = model_parameter.get("embed_dim")
  INNER_DIM = model_parameter.get("inner_dim")
  N = model_parameter.get("n")
  DROPOUT_RATE = model_parameter.get("dropout_rate")
  INITIALIZER = model_parameter.get("initializer")

  transformer = Transformer(context_num_vocabs=CONTEXT_NUM_VOCABS, context_max_len=CONTEXT_MAX_LEN, target_num_vocabs=TARGET_NUM_VOCABS, target_max_len=TARGET_MAX_LEN,
                            num_heads=NUM_HEADS, embed_dim=EMBED_DIM, inner_dim=INNER_DIM, n=N, dropout_rate=DROPOUT_RATE, initializer=INITIALIZER)

  context = Input(shape=(None,), dtype=tf.int32, name="context")
  target = Input(shape=(None,), dtype=tf.int32, name="target")
  context_mask = Input(shape=(None,), dtype=tf.float32, name="context_mask")
  target_mask = Input(shape=(None,), dtype=tf.float32, name="target_mask")

  inputs = [context, target, context_mask, target_mask]
  logits = transformer(inputs)
  model = tf.keras.Model(inputs=inputs, outputs=logits, name='Transformer')

  return model

def train(train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, model_parameters: dict, optimizer: str, lr: float, epochs: int, batch_size: int, scheduler: bool, model_save_dir: str):
  strategy = tf.distribute.MirroredStrategy() if len(tf.config.list_physical_devices('GPU')) > 1 else tf.distribute.get_strategy()

  print("Creating model...", flush=True)

  with strategy.scope():

    model = create_model(**model_parameters)

    loss_fn = MaskedCELoss(from_logits=True, reduction="sum_over_batch_size")
    acc_fn = MaskedAccuracy(from_logits=True)

    if scheduler:
      lr_scheduler = CustomScheduler(embed_dim=model_parameters.get("embed_dim"))
      optimizer = AdamW(learning_rate=lr_scheduler) if optimizer=="adamw" else Adam(learning_rate=lr_scheduler)

    else:
      optimizer = AdamW(learning_rate=lr) if optimizer=="adamw" else Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[acc_fn])
    model.summary()


  train_dataset = train_dataset.shuffle(len(train_dataset)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
  val_dataset = val_dataset.batch(batch_size)

  MODEL_SAVE_DIR_ROOT = os.path.join(model_save_dir, "model_" + str(datetime.now().strftime("%Y%m%d_%H%M%S")))
  MODEL_LOGS = os.path.join(MODEL_SAVE_DIR_ROOT, "logs")

  CSV_LOGGER = os.path.join(MODEL_LOGS, "csv_logger")
  TENSORBOARD_LOGGER = os.path.join(MODEL_LOGS, "tensorboard_logger")
  MODEL_CHECKPOINTS = os.path.join(MODEL_SAVE_DIR_ROOT, "checkpoints")
  MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_DIR_ROOT, "model.keras")

  if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir, exist_ok=True)

  if not os.path.exists(MODEL_SAVE_DIR_ROOT):
    os.makedirs(MODEL_SAVE_DIR_ROOT, exist_ok=True)

  if not os.path.exists(MODEL_LOGS):
    os.makedirs(MODEL_LOGS, exist_ok=True)

  if not os.path.exists(CSV_LOGGER):
    os.makedirs(CSV_LOGGER, exist_ok=True)

  if not os.path.exists(MODEL_CHECKPOINTS):
    os.makedirs(MODEL_CHECKPOINTS, exist_ok=True)

  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOGGER)
  csv_logger_callback = tf.keras.callbacks.CSVLogger(os.path.join(CSV_LOGGER, "results.csv"), append=True)
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(MODEL_CHECKPOINTS, "checkpoint.model.keras"), save_best_only=True)
  callbacks = [tensorboard_callback, csv_logger_callback, checkpoint_callback]

  print("Training model...", flush=True)

  history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)

  print("Saving model...", flush=True)
  model.save(MODEL_SAVE_DIR)
  print("Model saved!", flush=True)

  return history


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--train_data_path", type=str, required=True)
  parser.add_argument("--val_data_path", type=str, required=True)
  parser.add_argument("--model_parameter", type=str, required=True)
  parser.add_argument("--optimizer", type=str, required=True)
  parser.add_argument("--lr", type=float, required=True)
  parser.add_argument("--scheduler", type=str, required=True)
  parser.add_argument("--batch_size", type=int, required=True)
  parser.add_argument("--epochs", type=int, required=True)
  parser.add_argument("--save_dir", type=str, required=True)

  args_dict = parser.parse_args().__dict__

  train_data_path = ast.literal_eval(args_dict.get("train_data_path"))
  val_data_path = ast.literal_eval(args_dict.get("val_data_path"))
  model_parameter = ast.literal_eval(args_dict.get("model_parameter"))
  optimizer = args_dict.get("optimizer")
  lr = args_dict.get("lr")
  scheduler = True if args_dict.get("scheduler") == "True" else False
  batch_size = args_dict.get("batch_size")
  epochs = args_dict.get("epochs")
  model_save_dir = args_dict.get("save_dir")


  print("Preprocessing data...", flush=True)

  train_dataset, val_dataset = preprocessing_data(train_data_path=train_data_path, val_data_path=val_data_path)

  print("Preprocessing is done!", flush=True)

  history = train(train_dataset=train_dataset, val_dataset=val_dataset, model_parameters=model_parameter, optimizer=optimizer, lr=lr, epochs=epochs,
                  batch_size=batch_size, scheduler=scheduler, model_save_dir=model_save_dir)
