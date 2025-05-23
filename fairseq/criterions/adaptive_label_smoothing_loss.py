from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import metrics
import torch
import torch.nn.functional as F

@register_criterion("adaptive_label_smoothing")
class AdaptiveLabelSmoothing(FairseqCriterion):
    #@staticmethod
    #def add_args(parser):
        # criterion-specific arguments
        #parser.add_argument('--initial_alpha', type=float, default=0.0, help='초기 alpha 값')
        #parser.add_argument('--max_alpha', type=float, default=0.5, help='최대 alpha 값')
        #parser.add_argument('--', type=float, default=0.5, help='최대 alpha 값')
        #parser.add_argument('--total_steps', type=int, default=100000, help='alpha를 증가시킬 총 업데이트 횟수')
        #parser.add_argument('--temperature', type=float, default=2.0, help='temperature scaling')
    @classmethod
    def build_criterion(cls, args, task):
        # args를 명시적으로 __init__에 넘겨주는 부분
        return cls(
            task,
            total_steps=args.max_update,
            update_freq=args.update_freq,
        )
    def __init__(self, task, total_steps, update_freq):
        super().__init__(task)
        self.total_steps = total_steps
        self.update_freq = update_freq[0]
        self.padding_idx = task.target_dictionary.pad()
        
        self.curr_step = 0
        self.initial_alpha = 0.0
        self.temperature = 2.0
        self.max_alpha = 0.3
        self.update_alpha = self.get_update_alpha()
    
    def get_update_alpha(self):
        _alpha_max_step = self.total_steps // 2 # max_step의 0.5 구간에서 max_alpha 도달
        update_alpha = self.max_alpha / _alpha_max_step
        return update_alpha

    def get_current_alpha(self):
        update_step = self.curr_step // self.update_freq
        update_alpha = update_step * self.update_alpha
        alpha = self.initial_alpha + update_alpha
        return min(alpha, self.max_alpha)

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)  # (B, T, V) 
        logits = net_output[0]
        target = sample["target"]
        probs = F.softmax(logits / self.temperature, dim=-1)
        # print('[+] lprobs:', lprobs.shape, lprobs)
        # print('[+] logits:', logits.shape, logits)
        # print('[+] probs:', probs.shape, probs)
        # print('[+] target:', target.shape, target)
    
        B, T, V = lprobs.size()
        lprobs = lprobs.view(-1, V)
        probs = probs.view(-1, V)
        target = target.view(-1)

        mask = target != self.padding_idx
        # print('[+] mask:', mask.shape, mask)
        target = target[mask]
        lprobs = lprobs[mask]
        probs = probs[mask]

        # create one-hot vector
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, target.unsqueeze(1), 1.0)
        # print('[+] one_hot:', one_hot.shape, one_hot)
        # compute adaptive mixed target
        alpha = self.get_current_alpha()
        #print('[+] alpha:', alpha)
        mixed_target = (1 - alpha) * one_hot + alpha * probs.detach()
        # print('[+] mixed_target:', mixed_target.shape, mixed_target)
        
        loss = -(mixed_target * lprobs).sum(dim=1)
        loss = loss.sum()
        # loss = -(one_hot * lprobs).sum(dim=1).mean()

        sample_size = mask.sum()

        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
            "alpha": alpha,
        }
        self.curr_step +=1
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs):
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 1) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)

        if "alpha" in logging_outputs[0]:
            last_alpha = logging_outputs[-1]["alpha"]
            metrics.log_scalar("alpha", last_alpha, 1, round=4)