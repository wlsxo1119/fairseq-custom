from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import metrics
import torch
import torch.nn.functional as F

@register_criterion("margin_token_loss")
class MarginTokenLoss(FairseqCriterion):
    @staticmethod
    def add_args(parser):
        # criterion-specific arguments
        # Adaptive Label Smoothing
        parser.add_argument('--initial-alpha', type=float, default=0.0, help='initial alpha value')
        parser.add_argument('--max-alpha', type=float, default=0.5, help='max alpha value')
        parser.add_argument('--soft-target-temperature', type=float, default=2.0, help='soft-target temperature scaling')
        # Margin Weight Setting
        parser.add_argument('--max-margin-lambda', type=float, default=5.0, help='max margin lambda value')
        parser.add_argument('--initial-lambda', type=float, default=0.0, help='initial margin lambda value')
        # Train Start Step
        parser.add_argument('--init-step', type=float, default=0, help='train start step')
        
    @classmethod
    def build_criterion(cls, args, task):
        # args to __init__ method
        return cls(
            task,
            total_steps=args.max_update,
            update_freq=args.update_freq,
            init_step=args.init_step,
            initial_alpha=args.initial_alpha,
            max_alpha=args.max_alpha,
            temperature=args.soft_target_temperature,
            max_margin_lambda=args.max_margin_lambda,
            initial_lambda=args.initial_lambda,
        )
        
    def __init__(self, task, total_steps, update_freq, init_step, initial_alpha, max_alpha, temperature, max_margin_lambda, initial_lambda):
        super().__init__(task)
        self.total_steps = total_steps
        self.update_freq = update_freq[0]
        self.padding_idx = task.target_dictionary.pad()
        self.bos_idx = task.target_dictionary.bos()
        self.eos_idx = task.target_dictionary.eos()
        self.unk_idx = task.target_dictionary.unk()
        self.curr_step = init_step
        self.initial_alpha = initial_alpha
        self.max_alpha = max_alpha
        self.temperature = temperature
        self.max_margin_lambda = max_margin_lambda
        self.initial_lambda = initial_lambda
        self.update_alpha = self.get_update_alpha()
        self.update_lambda = self.get_update_lambda()
    
    # Reaches the maximum at the half of the training process
    def get_update_alpha(self):
        _alpha_max_step = self.total_steps // 2 
        update_alpha = self.max_alpha / _alpha_max_step
        return update_alpha

    # Update alpha value
    def get_current_alpha(self):
        update_step = self.curr_step // self.update_freq
        update_alpha = update_step * self.update_alpha
        alpha = self.initial_alpha + update_alpha
        return min(alpha, self.max_alpha)

    # Reaches the maximum at the half of the training process
    def get_update_lambda(self):
        _lambda_max_step = self.total_steps // 2
        update_lambda = self.max_margin_lambda / _lambda_max_step
        return update_lambda
    
    # Update lambda value
    def get_current_margin_lambda(self):
        update_step = self.curr_step // self.update_freq
        update_lambda = update_step * self.update_lambda
        _lambda = self.initial_lambda + update_lambda
        return min(_lambda, self.max_margin_lambda)

    # Margin Loss
    def margin_loss(self, log_probs_nmt, log_probs_lm, target, sample_size):
        # log_probs_nmt, log_probs_lm: [B, T, V] softmax log prob
        # target: [B, T]
        probs_nmt = log_probs_nmt.exp()
        probs_lm = log_probs_lm.exp()

        # gather token-level prob
        target_expand = target.unsqueeze(-1)  # [B, T, 1]
        p_nmt = probs_nmt.gather(dim=-1, index=target_expand).squeeze(-1)
        p_lm = probs_lm.gather(dim=-1, index=target_expand).squeeze(-1)

        # margin = p_nmt - p_lm
        margin = p_nmt - p_lm
        
        # M(margin): Quintic margin function
        margin_loss = (1 - p_nmt) * (1 - margin**5) / 2
        margin_loss = margin_loss.sum()
        return margin_loss

    def forward(self, model, sample, reduce=True):
        # Forward pass
        net_output = model(**sample["net_input"])

        # Probabilities
        lprobs = model.get_normalized_probs(net_output, log_probs=True)  # (B, T, V) 
        logits = net_output[0]
        target = sample["target"]
        #probs = F.softmax(logits / self.temperature, dim=-1)
        probs = F.softmax(logits / 0.5, dim=-1)
        
        # Make UNK Input (Margin Token Loss)
        bz = sample["net_input"]['src_tokens'].size(0)
        unk_src_tokens = torch.tensor([self.unk_idx, self.eos_idx]).unsqueeze(0).expand(bz, 2).to(sample['net_input']['src_tokens'].device)
        unk_src_lengths = torch.full((bz,), 2, dtype=sample['net_input']['src_lengths'].dtype, device=sample['net_input']['src_lengths'].device)
        unk_input = {
            'src_tokens': unk_src_tokens,  # (B, 2) - lm input = bz * [UNK, EOS]
            'src_lengths': unk_src_lengths, # (B,) - lm input length = bz * 2
            'prev_output_tokens': sample["net_input"]['prev_output_tokens'].clone(),
        }
        
        # Run LM Style Model
        unk_output = model(**unk_input)
        unk_lprobs = model.get_normalized_probs(unk_output, log_probs=True)  # (B, T, V) 

        # Reshape
        B, T, V = lprobs.size()
        lprobs = lprobs.view(-1, V)
        unk_lprobs = unk_lprobs.view(-1, V)
        probs = probs.view(-1, V)
        target = target.view(-1)

        # PAD Masking
        mask = target != self.padding_idx
        target = target[mask]
        lprobs = lprobs[mask]
        unk_lprobs = unk_lprobs[mask]
        probs = probs[mask]
        sample_size = mask.sum()
        
        # create one-hot vector
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, target.unsqueeze(1), 1.0)

        # compute adaptive mixed target
        # How to Learn in a Noisy World? Self-Correcting the Real-World Data Noise in Machine Translation
        probs = probs.detach() 
        entropy = -torch.sum(probs * lprobs.detach(), dim=1, keepdim=True)  # (B*T, 1)
        max_entropy = torch.log(torch.tensor(V, dtype=probs.dtype, device=probs.device))  # log(V)
        normalized_entropy = entropy / max_entropy  # normalize entropy

        update_step = self.curr_step // self.update_freq
        exponent =  -6. * (update_step + -0.6)
        time_scalar = 1 / (1 + torch.exp(torch.tensor(exponent, dtype=probs.dtype, device=probs.device)))

        alpha = (1-normalized_entropy) * time_scalar
        # print(alpha,time_scalar)
        mixed_target = alpha * one_hot + (1-alpha) * probs.detach()

        # compute mt loss
        mt_loss = -(mixed_target * lprobs).sum(dim=1)
        mt_loss = mt_loss.sum()
        
        # compute margin loss
        _lambda = self.get_current_margin_lambda()
        margin_loss = self.margin_loss(lprobs, unk_lprobs, target, sample_size)

        # compute total loss
        loss = mt_loss +  _lambda * margin_loss

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

        # if any("alpha" in log for log in logging_outputs):
        #     last_alpha = next((log["alpha"] for log in reversed(logging_outputs) if "alpha" in log), 0.0)
        #     metrics.log_scalar("alpha", last_alpha, 1, round=4)
        if any("lambda" in log for log in logging_outputs):
            last_lambda = next((log["lambda"] for log in reversed(logging_outputs) if "lambda" in log), 0.0)
            metrics.log_scalar("lambda", last_lambda, 1, round=4)