import numpy as np
import tensorflow as tf
import os
import time

import config
from utils import logger, ensure_dir
from network import CaptionNetwork, TrainInputPipeline

CHECKPOINT_INTERVAL = 2500

if __name__ == '__main__':
    input_pipeline = TrainInputPipeline([config.train_features_file], num_epochs=5, batch_size=config.batch_size)

    session = tf.Session()
    net = CaptionNetwork(session, input_pipeline)

    current_logs_path = os.path.join(config.logs_path, str(int(time.time())))
    ensure_dir(current_logs_path)
    summary_writer = tf.train.SummaryWriter(os.path.expanduser(current_logs_path), session.graph.as_graph_def())
    merged_summary = tf.merge_all_summaries()

    session.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)

    window_loss = 20.
    processed = 0

    try:
        while not coord.should_stop():
            _, loss, summary = session.run([net.train_task, net.loss, merged_summary])
            processed += 1
            window_loss = 0.95 * window_loss + 0.05 * loss
            logger().info('Batch %d completed, smoothed loss is %f', processed, window_loss)
            summary_writer.add_summary(summary, processed)
            if processed % CHECKPOINT_INTERVAL == 0:
                epochs_completed = processed / CHECKPOINT_INTERVAL
                weights_file = config.weights_file_template % epochs_completed
                net.save(weights_file)
                logger().info('Model saved to %s', weights_file)

    except tf.errors.OutOfRangeError:
        print 'Done'
    finally:
        coord.request_stop()

    coord.join(threads)
    session.close()