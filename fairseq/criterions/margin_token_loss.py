from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import metrics
import torch
import torch.nn.functional as F

@register_criterion("margin_token_loss")
class MarginTokenLoss(FairseqCriterion):
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
        self.bos_idx = task.target_dictionary.bos()
        self.eos_idx = task.target_dictionary.eos()
        self.unk_idx = task.target_dictionary.unk()

        self.curr_step = 0
        self.initial_alpha = 0.0
        self.max_alpha = 0.3
        self.temperature = 2.0
        self.max_margin_lambda = 5.0
        self.initial_lambda = 0.0
        self.update_alpha = self.get_update_alpha()
        self.update_lambda = self.get_update_lambda()
    
    def get_update_alpha(self):
        _alpha_max_step = self.total_steps // 2 # max_step의 0.5 구간에서 max_alpha 도달
        update_alpha = self.max_alpha / _alpha_max_step
        return update_alpha

    def get_current_alpha(self):
        update_step = self.curr_step // self.update_freq
        update_alpha = update_step * self.update_alpha
        alpha = self.initial_alpha + update_alpha
        return min(alpha, self.max_alpha)
    
    def get_update_lambda(self):
        _lambda_max_step = self.total_steps // 2
        update_lambda = self.max_margin_lambda / _lambda_max_step
        return update_lambda
    
    def get_current_margin_lambda(self):
        update_step = self.curr_step // self.update_freq
        update_lambda = update_step * self.update_lambda
        _lambda = self.initial_lambda + update_lambda
        return min(_lambda, self.max_margin_lambda)


    def margin_loss(self, log_probs_nmt, log_probs_lm, target, sample_size):
        # log_probs_nmt, log_probs_lm: [B, T, V] softmax log prob
        # target: [B, T]
        probs_nmt = log_probs_nmt.exp()
        probs_lm = log_probs_lm.exp()

        # gather token-level prob
        target_expand = target.unsqueeze(-1)  # [B, T, 1]
        p_nmt = probs_nmt.gather(dim=-1, index=target_expand).squeeze(-1)
        p_lm = probs_lm.gather(dim=-1, index=target_expand).squeeze(-1)

        # Δ = p_nmt - p_lm
        margin = p_nmt - p_lm
        
        # M(Δ): Quintic margin function
        margin_loss = (1 - p_nmt) * (1 - margin**5) / 2
        margin_loss = margin_loss.sum()  # normalize
        return margin_loss

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)  # (B, T, V) 
        logits = net_output[0]
        target = sample["target"]
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        # === UNK_REPLACE_INPUT 만들기 ===
        unk_input = {}
        for key, val in sample['net_input'].items():
            if key in ['src_tokens']:
                val_clone = val.clone()
                mask = ~((val_clone == self.padding_idx) | (val_clone == self.bos_idx) | (val_clone == self.eos_idx))
                val_clone[mask] = self.unk_idx
                unk_input[key] = val_clone
            else:
                unk_input[key] = val  # 그대로 유지

        # UNK_REPLACE_INPUT으로 모델 추론
        unk_output = model(**unk_input)
        unk_lprobs = model.get_normalized_probs(unk_output, log_probs=True)  # (B, T, V) 

        # ======  RESHAPE ======
        B, T, V = lprobs.size()
        lprobs = lprobs.view(-1, V)
        unk_lprobs = unk_lprobs.view(-1, V)
        probs = probs.view(-1, V)
        target = target.view(-1)

        mask = target != self.padding_idx
        target = target[mask]
        lprobs = lprobs[mask]
        unk_lprobs = unk_lprobs[mask]
        probs = probs[mask]
        sample_size = mask.sum()
        # ======  RESHAPE ======

        # create one-hot vector
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, target.unsqueeze(1), 1.0)

        # compute adaptive mixed target
        alpha = self.get_current_alpha()
        mixed_target = (1 - alpha) * one_hot + alpha * probs.detach()

        # compute loss
        mt_loss = -(mixed_target * lprobs).sum(dim=1)
        mt_loss = mt_loss.sum()
        
        # compute margin loss
        _lambda = self.get_current_margin_lambda()
        margin_loss = _lambda *  self.margin_loss(lprobs, unk_lprobs, target, sample_size)
        loss = mt_loss +  margin_loss
        logging_output = {
            "loss": loss.data,
            "mt_loss": mt_loss.data,
            "margin_loss": margin_loss.data,
            "sample_size": sample_size,
            "alpha": alpha,
            "lambda": _lambda,
        }
        self.curr_step +=1
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs):
        def get_sum(key):
            return sum(log.get(key, 0) for log in logging_outputs)

        sample_size = get_sum("sample_size")
        loss_sum = get_sum("loss")
        mt_loss_sum = get_sum("mt_loss")
        margin_loss_sum = get_sum("margin_loss")

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("mt_loss", mt_loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("margin_loss", margin_loss_sum / sample_size, sample_size, round=3)

        # alpha는 마지막 step 값만 로깅
        if any("alpha" in log for log in logging_outputs):
            last_alpha = next((log["alpha"] for log in reversed(logging_outputs) if "alpha" in log), 0.0)
            metrics.log_scalar("alpha", last_alpha, 1, round=4)
        if any("lambda" in log for log in logging_outputs):
            last_lambda = next((log["lambda"] for log in reversed(logging_outputs) if "lambda" in log), 0.0)
            metrics.log_scalar("lambda", last_lambda, 1, round=4)