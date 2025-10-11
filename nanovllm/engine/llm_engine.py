import atexit
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import RunnerTask, BaseModelRunner

class LLMEngineBase:
    model_runner : BaseModelRunner
    scheduler : Scheduler
    
    def __init__(self, RunnerType : type[BaseModelRunner], config: Config, tensor_parallel_size: int):
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=RunnerType, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = RunnerType(config, 0, self.events)
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request_seq(self, seq : Sequence):
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        runner_tasks = [self.preprocess_seq(seq, is_prefill) for seq in seqs]
        outputs = self.model_runner.call("run", runner_tasks, is_prefill)
        
        for seq, output in zip(seqs, outputs):
            self.postprocess_seq(seq, output, is_prefill)
        
        for seq in seqs:
            if seq.stoped:
                self.scheduler.finish(seq)

        return seqs

    def is_finished(self):
        return self.scheduler.is_finished()
    
    def preprocess_seq(self, seq : Sequence, is_prefill: bool) -> RunnerTask:
        raise NotImplementedError()
    
    def postprocess_seq(self, seq : Sequence, outputs : dict, is_prefill: bool):
        raise NotImplementedError()
