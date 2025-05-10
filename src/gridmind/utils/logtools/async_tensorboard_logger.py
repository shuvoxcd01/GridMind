from torch.utils.tensorboard import SummaryWriter
from queue import Queue
import threading


class AsyncTensorboardLogger:
    def __init__(self, log_dir=None, flush_secs=10):
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
        self.log_queue = Queue()
        self._stop_signal = object()

        self.thread = threading.Thread(target=self._log_worker, daemon=True)
        self.thread.start()

    def _log_worker(self):
        while True:
            item = self.log_queue.get()
            if item is self._stop_signal:
                break
            tag, value, step = item
            self.writer.add_scalar(tag, value, step)
            self.log_queue.task_done()

    def add_scalar(self, tag, value, global_step, walltime=None):
        self.log_queue.put((tag, value, global_step))

    def close(self):
        self.log_queue.put(self._stop_signal)
        self.thread.join()
        self.writer.close()
